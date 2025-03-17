from playwright.async_api import async_playwright, Locator
import pandas as pd
from io import StringIO
from pathlib import Path
import argparse
import json
from textwrap import dedent
import asyncio

# Constants
SECTIONS = {
    "schema": "Schema Data Label & Description",
    "usa_spending": "USA Spending Downloads",
    "database": "Database Download",
    "legacy": "Legacy USA Spending",
}
FILENAMES = {
    key: value.replace(" ", "_").replace("&", "") + ".csv"
    for key, value in SECTIONS.items()
}
OUTPUT_DIR = Path("data_dicts")


async def click_button(button: Locator):
    """Helper function to click a single button if visible."""
    try:
        if await button.is_visible():
            await button.click()
    except Exception as e:
        print(f"Warning: Failed to click button - {e}")


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


def process_data_dictionary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Clean the MultiIndex and split the DataFrame into section-specific DataFrames."""
    # Clean the second level of the multi-index columns
    df.columns = df.columns.map(
        lambda label: (label[0], label[1].split("Sort")[0].strip())
    )
    # Split into DataFrames based on section headers
    dfs = {
        key: df.loc[:, df.columns.get_level_values(0) == value].copy()
        for key, value in SECTIONS.items()
    }
    # Remove the section header level from column names
    for df_val in dfs.values():
        df_val.columns = df_val.columns.droplevel(0)
    return dfs


def save_data_dictionary(dfs: dict[str, pd.DataFrame], output_dir: Path = OUTPUT_DIR):
    """Save the section-specific DataFrames to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, df_val in dfs.items():
        output_path = output_dir / FILENAMES[key]
        df_val.to_csv(output_path, index=False)
    print("Data dictionary has been scraped and saved to CSVs.")


def load_data_dictionary(input_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Load existing CSV files into a combined MultiIndex DataFrame."""
    dfs = {}
    for key, filename in FILENAMES.items():
        if (file_path := input_dir / filename).is_file():
            dfs[key] = pd.read_csv(file_path)
            dfs[key].columns = pd.MultiIndex.from_product(
                [[SECTIONS[key]], dfs[key].columns], names=["Section", "Field"]
            )
        else:
            print(f"Warning: {file_path} not found")

    if dfs:
        combined_df = pd.concat(dfs.values(), axis=1)
        return combined_df
    print("No CSV files found in the directory.")
    return pd.DataFrame()


async def main():
    """Main entry point for the USAspending Data Dictionary Tool."""
    parser = argparse.ArgumentParser(description="USAspending Data Dictionary Tool")
    parser.add_argument(
        "--create",
        action="store_true",
        help="Scrape and save the data dictionary to CSVs",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the data dictionary from existing CSVs",
    )
    args = parser.parse_args()

    match (args.create, args.load):
        case (True, False):
            df = await scrape_data_dictionary()
            dfs = process_data_dictionary(df)
            save_data_dictionary(dfs)
        case (False, True):
            combined_df = load_data_dictionary()
            pd.set_option('display.max_columns', None)
            pd.set_option("display.max_rows", None)  # Show all rows
            pd.set_option("display.max_columns", None)  # Show all columns
            pd.set_option("display.width", 1000)  # Increase width
            pd.set_option("display.max_colwidth", None)
            print(combined_df.columns)
            if not combined_df.empty:
                print("Loaded Data Dictionary (first 5 rows):")
                #print(combined_df.head())
        case (True, True):
            print("Error: Cannot use both --create and --load together")
        case (False, False):
            print("Please specify either --create or --load")


if __name__ == "__main__":
    asyncio.run(main())
