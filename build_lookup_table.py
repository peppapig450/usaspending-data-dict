from __future__ import annotations

import pandas as pd
import json
from typing import Annotated, Literal
from pathlib import Path

type CategoryType = Literal["award", "subaward", "account"]
ElementsDict: dict[str, str]

def export_award_elements_to_json(
    df: pd.DataFrame, output_file: str = "award_defs_lookup_table.json"
) -> None:
    """Export Award Elements and their Definitions to a JSON lookup table."""
    lookup_dict = (
        df.loc[
            :,
            [
                ("USA Spending Downloads", "Schema Data Label & Description"),
                ("Award Element", "Definition"),
            ],
        ]
        .dropna(subset=[("USA Spending Downloads", "Award Element")])
        .set_index(("USA Spending Downloads", "Award Element"))[
            ("Schema Data Label & Description", "Definition")
        ]
        .fillna("No definition available")
        .astype(str)
        .to_dict()
    )

    with open(output_file, "w") as file:
        json.dump(lookup_dict, file)
    print(f"Award elements lookup table saved to {output_file}")