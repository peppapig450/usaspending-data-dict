from __future__ import annotations

from playwright.async_api import async_playwright
import pandas as pd
from io import StringIO
from pathlib import Path
from typing import NamedTuple
import argparse
import json
import shlex
import asyncio
from enum import StrEnum, auto


class Section(StrEnum):
    SCHEMA = auto()
    USA_SPENDING = auto()
    DATABASE = auto()
    LEGACY = auto()


# Section labels exactly as they appear in the MultiIndex
SECTION_LABELS: dict[Section, str] = {
    Section.SCHEMA: "Schema Data Label & Description",
    Section.USA_SPENDING: "USA Spending Downloads",
    Section.DATABASE: "Database Download",
    Section.LEGACY: "Legacy USA Spending",
}

FILENAMES: dict[Section, str] = {
    key: value.replace(" ", "_").replace("&", "") + ".csv"
    for key, value in SECTION_LABELS.items()
}

# Type aliases
type ColumnRef = tuple[str, str]  # (section_label, field_name)


class QueryAction(StrEnum):
    QUERY = auto()
    MAP = auto()


class OutputFormat(StrEnum):
    JSON = auto()
    CSV = auto()
    TEXT = auto()


# NamedTuple for structured returns
class QueryComponents(NamedTuple):
    action: QueryAction
    source: ColumnRef
    target: ColumnRef
    output_format: OutputFormat
    output_file: str | None = None


class QueryResult:
    def __init__(
        self,
        data: list[dict] | str | None = None,
        output_format: OutputFormat = OutputFormat.JSON,
        message: str | None = None,
        *,
        success: bool = True,
    ) -> None:
        self.data = data or []
        self.format = output_format
        self.message = message
        self.success = success

    def __str__(self) -> str:
        if self.message:
            return self.message

        if not self.data:
            return "No data available"

        if isinstance(self.data, str):
            return self.data

        if self.format == "json":
            return json.dumps(self.data, indent=2)
        elif self.format == "csv":
            return pd.DataFrame(self.data).to_csv(index=False)
        elif self.format == "text":
            # Assuming the data is a list of dictionaries with two items each
            # NOTE: more robust?
            if all(len(item) == 2 for item in self.data):
                return "\n".join(
                    f"{list(item.values())[0]} -> {list(item.values())[1]}"
                    for item in self.data
                )
            return "\n".join(str(item) for item in self.data)

        return str(self.data)


OUTPUT_DIR = Path("data_dicts")


def parse_query(query_string: str) -> QueryComponents:
    """
    Parse the DSL query string into components using shlex for robust tokenization.

    Args:
        query_string: DSL query (e.g., "map usa_spending.'Award Element' -> schema.'Domain Values' as json to output.json")

    Returns:
        QueryComponents with action, source, target, output_format, and output_file
    """
    if "->" not in query_string:
        raise ValueError(
            "Query must contain exactly one '->' to separate column references."
        )

    # Split into left and right parts (before and after the arrow)
    left_right_parts = query_string.split("->", 1)
    left_part = left_right_parts[0].strip()
    right_part = left_right_parts[1].strip()

    # Parse the left part using shlex
    left_tokens = shlex.split(left_part)

    # Determine action based on first token
    if left_tokens[0] in QueryAction.__members__.values():
        action = QueryAction(left_tokens[0])
        # The rest of the tokens form the column reference
        col1_ref = " ".join(left_tokens[1:])
    else:
        action = QueryAction.QUERY  # Default action
        col1_ref = left_part

    # Parse the right part (which may have 'as format' and 'to file' components)

    output_file = None
    # Check for "to file" syntax
    if " to " in right_part:
        col2_and_format, output_file_part = right_part.split(" to ", 1)
        # Use shlex to handle quoted filenames
        output_file_tokens = shlex.split(output_file_part)
        output_file = output_file_tokens[0] if output_file_tokens else None
    else:
        col2_and_format = right_part

    # Check for "as format" syntx
    if " as " in col2_and_format:
        col2_ref, format_part = col2_and_format.split(" as ", 1)
        format_tokens = shlex.split(format_part)
        output_format = (
            OutputFormat(format_tokens[0]) if format_tokens else OutputFormat.JSON
        )  # Default to JSON
    else:
        col2_ref = col2_and_format
        output_format = OutputFormat.JSON

    # Clean up whitespace
    col1_ref = col1_ref.strip()
    col2_ref = col2_ref.strip()

    # Validate output format
    if output_format not in ("json", "csv", "text"):
        error_msg = f"Invalid output format '{output_format}'. Use: json, csv, or text"
        raise ValueError(error_msg)

    # Parse column references which should be in the format section.field
    try:
        # If fields contain spaces, they might be quoted and properly handled by shlex
        col1_section_key, col1_field = col1_ref.split(".", 1)
        col2_section_key, col2_field = col2_ref.split(".", 1)
    except ValueError:
        raise ValueError(
            "Each column reference must be in 'section.field' format (e.g., 'schema.Element')."
        )

    # Convert section keys to actual section labels
    try:
        col1_section = SECTION_LABELS[Section(col1_section_key)]
        col2_section = SECTION_LABELS[Section(col2_section_key)]
    except (KeyError, ValueError) as e:
        valid_sections = ", ".join(section.value for section in Section)
        raise ValueError(f"Invalid section(s). Valid sections: {valid_sections}") from e

    # Ensure output file is specified for map action
    if action == "map" and not output_file:
        raise ValueError(
            "The 'map' action requires an output file specified with 'to filename'"
        )

    return QueryComponents(
        action=action,
        source=(col1_section, col1_field),
        target=(col2_section, col2_field),
        output_format=output_format,
        output_file=output_file,
    )


def query_data_dictionary(
    df: pd.DataFrame, query_components: QueryComponents
) -> QueryResult:
    """
    Query the DataFrame or create a mapping, based on the provided QueryComponents.

    Args:
        df: Combined MultiIndex DataFrame
        query_components: The parsed query components

    Returns:
        QueryResult containing the requested data and format
    """
    if df.empty:
        return QueryResult(message="Error: No data loaded.", success=False)

    # Unpack components for clarity
    action = query_components.action
    col1 = query_components.source
    col2 = query_components.target
    output_format = query_components.output_format
    output_file = query_components.output_file

    # Check if columns exist in the MultiIndex
    try:
        # Check if the columns exist in the DataFrame
        if col1 not in df.columns or col2 not in df.columns:
            # Create a dictionary of available fields for each section
            available_fields = {}
            for section in Section:
                section_label = SECTION_LABELS[section]
                if section_label in df.index.get_level_values("Section"):
                    available_fields[section.value] = df.xs(
                        section_label, level="Section", axis=1
                    ).columns.tolist()

            return QueryResult(
                message=f"Error: Column not found. Available fields: {json.dumps(available_fields, indent=2)}",
                output_format=output_format,
                success=False,
            )
    except Exception as e:
        return QueryResult(
            message=f"Error accessing columns: {str(e)}",
            output_format=output_format,
            success=False,
        )

    # Select columns and drop NaN rows
    df_subset = df[[col1, col2]].dropna()

    if df_subset.empty:
        return QueryResult(
            message="No data matches the query after dropping NaN values.",
            output_format=output_format,
            success=True,
        )

    if action == QueryAction.MAP:
        # Create dictionary mapping
        mapping_dict = dict(zip(df_subset[col1], df_subset[col2]))

        # Make sure output_file is passed
        if output_file is None:
            raise ValueError("No output_file specified.")

        # Write to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(mapping_dict, indent=2))

        return QueryResult(
            message=f"Mapping saved to {output_file}",
            output_format=output_format,
            success=True,
        )

    else:  # action == "query"
        # Rename columns for output clarity
        col1_name = f"{next(s.value for s in Section if SECTION_LABELS[s] == col1[0])} - {col1[1]}"
        col2_name = f"{next(s.value for s in Section if SECTION_LABELS[s] == col2[0])} - {col2[1]}"

        # Create a new DataFrame to avoid modifying the original
        result_df = df_subset.copy()
        result_df.columns = [col1_name, col2_name]

        # Format output
        list_of_dicts = result_df.to_dict(orient="records")
        return QueryResult(
            data=list_of_dicts, output_format=output_format, success=True
        )


async def scrape_data_dictionary(
    url: str = "https://www.usaspending.gov/data-dictionary",
) -> pd.DataFrame:
    """Scrape the data dictionary table from the USAspending website, clicking all 'Read More' buttons in parallel with TaskGroup."""
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="load")

        # Wait for 'Read More' buttons to load
        button_selector = "button.read-more-button:has-text('Read More')"
        await page.wait_for_selector(button_selector, timeout=10000)

        # Click all 'Read More' buttons using JavaScript with staggered timing
        await page.evaluate("""
            async () => {
                const buttons = document.querySelectorAll('button.read-more-button');
                await Promise.allSettled(
                    [...buttons].map((btn, index) =>
                        new Promise((resolve) => setTimeout(() => {
                            btn.click();
                            resolve();
                        }, Math.floor(index / 25) * 500))
                    )
                );
            }
        """)

        # Wait for the table to be fully updated after all expansions
        await page.wait_for_selector("table", state="visible", timeout=15000)

        html_content = await page.content()

        await browser.close()

    # Parse the HTML content into a DataFrame
    tables = pd.read_html(StringIO(html_content), header=None)
    df = tables[1].copy()
    df.columns = tables[0].columns
    return df


def process_data_dictionary(df: pd.DataFrame) -> dict[Section, pd.DataFrame]:
    """Clean the MultiIndex and split the DataFrame into section-specific DataFrames."""
    # Clean the second level of the multi-index columns
    df.columns = df.columns.map(
        lambda label: (label[0], label[1].split("Sort")[0].strip())
    )
    # Split into DataFrames based on section headers
    dfs = {
        key: df.loc[:, df.columns.get_level_values(0) == value].copy()
        for key, value in SECTION_LABELS.items()
    }
    # Remove the section header level from column names
    for df_val in dfs.values():
        df_val.columns = df_val.columns.droplevel(0)
    return dfs


def save_data_dictionary(
    dfs: dict[Section, pd.DataFrame], output_dir: Path = OUTPUT_DIR
):
    """Save the section-specific DataFrames to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, df_val in dfs.items():
        output_path = output_dir / FILENAMES[key]
        df_val.to_csv(output_path, index=False)
    print("Data dictionary has been scraped and saved to CSVs.")


def load_data_dictionary(input_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """
    Load the data dictionary from saved CSV files and return as a combined DataFrame.

    Returns:
        Combined MultiIndex DataFrame
    """
    # Ensure the output directory exists
    if not OUTPUT_DIR.exists():
        return pd.DataFrame()

    # Load each section's CSV file
    dfs = []
    for section, filename in FILENAMES.items():
        file_path = OUTPUT_DIR / filename
        if file_path.exists():
            df = pd.read_csv(file_path)

            # Create a MultiIndex DataFrame
            if not df.empty:
                # Convert to MultiIndex
                columns = pd.MultiIndex.from_product(
                    [[SECTION_LABELS[section]], df.columns], names=["Section", "Field"]
                )
                # Create a new DataFrame with MultiIndex columns
                df_multi = pd.DataFrame(df.values, columns=columns)
                dfs.append(df_multi)

    # Combine all DataFrames
    if dfs:
        return pd.concat(dfs, axis=1)
    return pd.DataFrame()


def parse_dsl_command(command: str) -> list[str]:
    """
    Parse a DSL command string using shlex to handle quoted arguments properly.

    Args:
        command: A DSL command string

    Returns:
        List of tokens from the command
    """
    return shlex.split(command)


async def main():
    """Main entry point for the USAspending Data Dictionary Tool."""
    parser = argparse.ArgumentParser(
        description="USAspending Data Dictionary Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_dict.py create                         # Scrape and save the data dictionary
  python data_dict.py load                           # Load and display the data dictionary
  python data_dict.py query "schema.Element -> usa_spending.'Award Element' as json"
  python data_dict.py query "map usa_spending.'Award Element' -> schema.'Domain Values' as json to output.json"
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to execute"
    )

    # Subparser for 'create'
    subparsers.add_parser("create", help="Scrape and save the data dictionary to CSVs")

    # Subparser for 'load'
    subparsers.add_parser("load", help="Load the data dictionary from existing CSVs")

    # Subparser for 'query'
    query_parser = subparsers.add_parser(
        "query", help="Query or map the data dictionary with a DSL"
    )
    query_parser.add_argument(
        "query_string",
        help="DSL query (e.g., \"map usa_spending.'Award Element' -> schema.'Domain Values' as json to output.json\")",
    )

    # Subparser for 'interactive' mode
    interactive_parser = subparsers.add_parser(
        "interactive", help="Start an interactive DSL shell"
    )

    args = parser.parse_args()

    match args.command:
        case "create":
            print("Scraping data dictionary...")
            df = await scrape_data_dictionary()
            dfs = process_data_dictionary(df)
            save_data_dictionary(dfs)

        case "load":
            print(f"Loading data dictionary from {OUTPUT_DIR}...")
            combined_df = load_data_dictionary()
            if combined_df.empty:
                print(f"No data found in {OUTPUT_DIR}")
            else:
                print(f"Loaded data dictionary with shape: {combined_df.shape}")
                print("\nFirst 5 rows:")
                print(combined_df.head())

        case "query":
            try:
                query_components = parse_query(args.query_string)
                print(f"Parsed query components: {query_components}")

                print("Loading data dictionary...")
                combined_df = load_data_dictionary()

                if combined_df.empty:
                    print(f"No data found in {OUTPUT_DIR}")
                else:
                    print("Executing query...")
                    result = query_data_dictionary(combined_df, query_components)
                    print("\nResult:")
                    print(result)

            except ValueError as e:
                print(f"Error parsing query: {e}")
                print("\nSyntax:")
                print(
                    "  query <section>.<field> -> <section>.<field> [as (json|csv|text)] [to <output_file>]"
                )
                print(
                    "  map <section>.<field> -> <section>.<field> [as (json|csv|text)] to <output_file>]"
                )
                print("\nExamples:")
                print("  query schema.Element -> usa_spending.'Award Element' as json")
                print(
                    "  map usa_spending.'Award Element' -> schema.'Domain Values' as json to output.json"
                )

        case "interactive":
            print("Starting interactive DSL shell. Type 'exit' or 'quit' to exit.")
            print("Example commands:")
            print("  query schema.Element -> usa_spending.'Award Element' as json")
            print(
                "  map usa_spending.'Award Element' -> schema.'Domain Values' as json to output.json"
            )

            # Load data dictionary once for the interactive session
            print("Loading data dictionary...")
            combined_df = load_data_dictionary()

            if combined_df.empty:
                print(f"No data found in {OUTPUT_DIR}")
                return

            print(f"Loaded data dictionary with shape: {combined_df.shape}")

            # Interactive loop
            while True:
                try:
                    user_input = input("\ndata_dict> ").strip()

                    if user_input.lower() in ("exit", "quit", "q"):
                        print("Exiting interactive shell.")
                        break

                    if not user_input:
                        continue

                    # Parse and execute the query
                    query_components = parse_query(user_input)
                    result = query_data_dictionary(combined_df, query_components)
                    print("\nResult:")
                    print(result)

                except ValueError as e:
                    print(f"Error: {e}")
                except KeyboardInterrupt:
                    print("\nExiting interactive shell.")
                    break
                except Exception as e:
                    print(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
