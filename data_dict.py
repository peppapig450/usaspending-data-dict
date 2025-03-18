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
import logging

# Module-level logger
logger = logging.getLogger(__name__)


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

    Raises:
        ValueError: If the query syntax is invalid
    """
    # Tokenize the entire query string using shlex.split
    tokens = shlex.split(query_string)
    if not tokens:
        raise ValueError("Empty query")

    # Determine the action (first token may be 'query' or 'map')
    action = QueryAction.QUERY  # Default action
    if tokens[0] in QueryAction.__members__.values():
        action = QueryAction(tokens[0])
        tokens = tokens[1:]  # Remove action token
    else:
        action = QueryAction.QUERY  # Action is optional, defaults to 'query'

    # Expect at least col1_ref, '->', col2_ref
    if len(tokens) < 3 or "->" not in tokens:
        raise ValueError(
            "Invalid query syntax. Expected: [action] col1 -> col2 [as format] [to file]"
        )

    # Find the index of '->' to separate left and right parts
    try:
        arrow_index = tokens.index("->")
    except ValueError:
        raise ValueError("Expected '->' between column references")

    # Extract col1_ref (everything before '->')
    if arrow_index < 1:
        raise ValueError("Column reference missing before '->'")
    col1_ref = " ".join(tokens[:arrow_index])  # Rejoin tokens in case of spaces

    # Extract tokens after '->' for the right side
    right_tokens = tokens[arrow_index + 1 :]
    if not right_tokens:
        raise ValueError("Column reference missing after '->'")

    # Parse the right side: col2_ref, [as format], [to file]
    output_format = OutputFormat.JSON  # Default
    output_file = None
    col2_ref = None

    # Look for 'as' and 'to' in the right side tokens
    i = 0
    while i < len(right_tokens):
        if right_tokens[i] == "as":
            if i + 1 >= len(right_tokens):
                raise ValueError("'as' must be followed by format")
            output_format = OutputFormat(right_tokens[i + 1])
            if col2_ref is None:  # 'as' appears before col2_ref is fully parsed
                col2_ref = " ".join(right_tokens[:i])
            i += 2  # Skip 'as' and format
        elif right_tokens[i] == "to":
            if i + 1 >= len(right_tokens):
                raise ValueError("'to' must be followed by output file")
            output_file = right_tokens[i + 1]
            if col2_ref is None:  # 'to' appears before col2_ref is fully parsed
                col2_ref = " ".join(right_tokens[:i])
            i += 2  # Skip 'to' and file
        else:
            i += 1  # Move to next token
            # If no 'as' or 'to' yet, keep building col2_ref
            if i == len(right_tokens) and col2_ref is None:
                col2_ref = " ".join(right_tokens)

    if not col2_ref:
        raise ValueError("Second column reference (col2) not found after '->'")

    # For 'map' action, ensure output_file is provided
    if action == QueryAction.MAP and output_file is None:
        raise ValueError(
            "The 'map' action requires an output file specified with 'to filename'"
        )

    # Function to parse column references into section and field
    def parse_column_ref(ref: str) -> tuple[str, str]:
        # Remove outer quotes if present
        ref = ref.strip()
        if ref.startswith("'") and ref.endswith("'"):
            ref = ref[1:-1]
        elif ref.startswith('"') and ref.endswith('"'):
            ref = ref[1:-1]
        # Split on the first dot
        try:
            section_key, field = ref.split(".", 1)
        except ValueError:
            raise ValueError(
                f"Invalid column reference: '{ref}'. Expected 'section.field'"
            )
        if not section_key or not field:
            raise ValueError(
                f"Invalid column reference: '{ref}'. Both section and field are required"
            )
        return section_key, field

    # Parse column references
    col1_section_key, col1_field = parse_column_ref(col1_ref)
    col2_section_key, col2_field = parse_column_ref(col2_ref)

    # Convert section keys to actual section labels
    try:
        col1_section = SECTION_LABELS[Section(col1_section_key)]
        col2_section = SECTION_LABELS[Section(col2_section_key)]
    except (KeyError, ValueError) as e:
        valid_sections = ", ".join(section.value for section in Section)
        raise ValueError(f"Invalid section(s). Valid sections: {valid_sections}") from e

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
        logging.warning("No data loaded in DataFrame")
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
        if col1 in df.columns:
            logging.info("Column: %s found in columns", col1)
        if col1 not in df.columns or col2 not in df.columns:
            # Create a dictionary of available fields for each section
            available_fields = {
                section.value: df.xs(
                    SECTION_LABELS[section], level="Section", axis=1
                ).columns.tolist()
                for section in Section
                if SECTION_LABELS[section] in df.columns.get_level_values("Section")
            }
            logger.error(
                "Column: %s or %s not found in DataFrame. Available fields: %s",
                col1,
                col2,
                json.dumps(available_fields, indent=2),
            )

            return QueryResult(
                message=f"Error: Column not found. Available fields: {json.dumps(available_fields, indent=2)}",
                output_format=output_format,
                success=False,
            )
    except Exception as e:
        logger.exception("Error accessing columns")
        return QueryResult(
            message=f"Error accessing columns: {str(e)}",
            output_format=output_format,
            success=False,
        )

    # Select columns and drop NaN rows
    df_subset = df[[col1, col2]].dropna()

    if df_subset.empty:
        logger.info("No data matches query after dropping NaN values")
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
            logger.error("No output file specified for map action")
            raise ValueError("No output_file specified.")

        # Write to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(mapping_dict, indent=2))

        logger.info("Mapping saved to %s", output_file)
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
        logger.info("Query executed successfully")
        return QueryResult(
            data=list_of_dicts, output_format=output_format, success=True
        )


async def scrape_data_dictionary(
    url: str = "https://www.usaspending.gov/data-dictionary",
) -> pd.DataFrame:
    """Scrape the data dictionary table from the USAspending website, clicking all 'Read More' buttons in parallel with TaskGroup."""
    logger.info("Starting scrape of data dictionary from %s", url)
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
    logger.info("Data dictionary scraped successfully")
    return df


def process_data_dictionary(df: pd.DataFrame) -> dict[Section, pd.DataFrame]:
    """Clean the MultiIndex and split the DataFrame into section-specific DataFrames."""
    logger.debug("Processing DataFrame with shape: %s", df.shape)
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
    logger.debug("DataFrame split into %d sections", len(dfs))
    return dfs


def save_data_dictionary(
    dfs: dict[Section, pd.DataFrame], output_dir: Path = OUTPUT_DIR
):
    """Save the section-specific DataFrames to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, df_val in dfs.items():
        output_path = output_dir / FILENAMES[key]
        df_val.to_csv(output_path, index=False)
        logger.info("Saved %s to %s", FILENAMES[key], output_path)
    logger.info("Data dictionary has been scraped and saved to CSVs.")


def load_data_dictionary(input_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """
    Load the data dictionary from saved CSV files and return as a combined DataFrame.

    Returns:
        Combined MultiIndex DataFrame
    """
    logger.info("Loading data dictionary from %s", input_dir)
    # Ensure the output directory exists
    if not input_dir.exists():
        logger.warning("Directory %s does not exist", input_dir)
        return pd.DataFrame()

    # Load each section's CSV file
    dfs = {}
    for key, filename in FILENAMES.items():
        file_path = input_dir / filename
        if file_path.is_file():
            dfs[key] = pd.read_csv(file_path)

            # Create a MultiIndex DataFrame
            if not dfs[key].empty:
                # Convert to MultiIndex
                dfs[key].columns = pd.MultiIndex.from_product(
                    [[SECTION_LABELS[key]], dfs[key].columns],
                    names=["Section", "Field"],
                )
                logger.debug("Loaded %s with shape: %s", filename, dfs[key].shape)
            else:
                logger.warning("File %s not found in %s", filename, input_dir)

    if dfs:
        combined_df = pd.concat(dfs.values(), axis=1)
        logger.info("Combined data dictionary loaded with shape: %s", combined_df.shape)
        return combined_df
    logger.warning("No CSV files found in %s", input_dir)
    return pd.DataFrame()


def parse_dsl_command(command: str) -> list[str]:
    """
    Parse a DSL command string using shlex to handle quoted arguments properly.

    Args:
        command: A DSL command string

    Returns:
        List of tokens from the command
    """
    logger.debug("Parsing DSL command: %s", command)
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
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

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("data_dict.log"),
        ],
    )
    logger.info(
        "Starting USAspending Data Dictionary Tool with log level: %s", args.log_level
    )

    match args.command:
        case "create":
            logger.info("Scraping data dictionary...")
            df = await scrape_data_dictionary()
            dfs = process_data_dictionary(df)
            save_data_dictionary(dfs)

        case "load":
            logger.info("Loading data dictionary from %s", OUTPUT_DIR)
            combined_df = load_data_dictionary()
            if combined_df.empty:
                logger.warning("No data found in %s", OUTPUT_DIR)
            else:
                logger.debug(f"Loaded data dictionary with shape: {combined_df.shape}")

        case "query":
            try:
                logger.debug("Parsing query: %s", args.query_string)
                query_components = parse_query(args.query_string)
                logger.debug("Parsed query components: %s", query_components)

                print("Loading data dictionary...")
                combined_df = load_data_dictionary()
                logger.debug(f"Loaded data dictionary with shape: {combined_df.shape}")

                logger.info("Loading data dictionary")
                if combined_df.empty:
                    logger.warning("No data found in %s", OUTPUT_DIR)
                else:
                    logger.info("Executing query")
                    result = query_data_dictionary(combined_df, query_components)
                    print("\nResult:")
                    print(result)

            except ValueError as e:
                logger.exception("Error parsing query")
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
            logger.info("Starting interactive DSL shell")

            logger.info("Loading data dictionary for interactive session")
            combined_df = load_data_dictionary()

            if combined_df.empty:
                logger.warning("No data found in %s", OUTPUT_DIR)
                return

            print(f"Loaded data dictionary with shape: {combined_df.shape}")

            # Interactive loop
            while True:
                try:
                    user_input = input("\ndata_dict> ").strip()

                    if user_input.lower() in ("exit", "quit", "q"):
                        logger.info("Exiting interactive shell")
                        break

                    if not user_input:
                        continue

                    # Parse and execute the query
                    logger.debug("User input: %s", user_input)
                    query_components = parse_query(user_input)
                    result = query_data_dictionary(combined_df, query_components)
                    print("\nResult:")
                    print(result)

                except ValueError:
                    logger.exception("Error in interactive query")
                except KeyboardInterrupt:
                    logger.info("Interactive shell interrupted by user")
                    break
                except Exception:
                    logger.exception("Unexpected error in interactive shell")


if __name__ == "__main__":
    asyncio.run(main())
