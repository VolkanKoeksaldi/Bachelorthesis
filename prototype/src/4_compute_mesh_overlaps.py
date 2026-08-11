from itertools import combinations
from pathlib import Path

import pandas as pd

from clustering_utils import parse_item_ids
from experiment_config import experiment_path

INPUT_PATH = experiment_path("processed/mesh_fragments.csv")

OUTPUT_PATH = experiment_path("processed/mesh_overlaps.csv")

UPDATED_INPUT_PATH = experiment_path("reoptimization/mesh_fragments_updates.csv")
UPDATED_OUTPUT_PATH = experiment_path("reoptimization/mesh_overlaps_updates.csv")

MODES = {
    "baseline": (INPUT_PATH, OUTPUT_PATH),
    "updates": (UPDATED_INPUT_PATH, UPDATED_OUTPUT_PATH),
}

MODE = "baseline"

def compute_overlaps(fragments_df):
    """
    Computes the overlaps between fragments belonging to different schemes.
    Here only pairs with at least one shared item are included in the resulting table.
    """

    prepared = [{"fragment_id": row.fragment_id,
                 "scheme": row.scheme,
                 "value": row.value,
                 "item_ids": parse_item_ids(row.tuple_ids)}
                 for row in fragments_df.itertuples(index=False)]
    
    overlap_rows = []
    
    # Examines every unordered pair of fragments
    for f1, f2 in combinations(prepared, 2):

        # if fragments are from the same scheme, they are not compared
        if f1["scheme"] == f2["scheme"]:
            continue
        
        # determines the overlap by calculating intersection of item ids
        overlap = f1["item_ids"] & f2["item_ids"]

        if not overlap:
            continue

        # Stores only pairs with intersections that are not empty
        overlap_rows.append({
            "fragment_1": f1["fragment_id"],
            "scheme_1": f1["scheme"],
            "value_1": f1["value"],
            "fragment_2": f2["fragment_id"],
            "scheme_2": f2["scheme"],
            "value_2": f2["value"],
            "overlap_size": len(overlap),
            "overlap_tuple_ids": ",".join(sorted(overlap))
        })

    return pd.DataFrame(overlap_rows)

def process_overlaps(input_path: Path, output_path: Path):
    """
    Loads the fragment definitions and then computes their overlaps.
    The results are then stored as a CSV file.
    """

    fragments_df = pd.read_csv(input_path)

    required = {"fragment_id", "scheme", "value", "tuple_ids"}

    # checks whether every required column is included in fragments
    missing = required - set(fragments_df.columns)
    if missing:
        raise ValueError(f"Fragment file is missing: {sorted(missing)}")

    overlaps_df = compute_overlaps(fragments_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlaps_df.to_csv(output_path, index=False)

    print("Input fragments:", len(fragments_df))
    print("Computed overlaps:", len(overlaps_df))
    print(f"Sum of pairwise overlap sizes: {overlaps_df['overlap_size'].sum()}")
    print(f"Saved to; {output_path}")

    return overlaps_df

def main():

    if MODE not in MODES:
        raise ValueError(f"Unknown execution mode: {MODE}")

    input_path, output_path = MODES[MODE]
    print("\nMode: ", MODE)
    process_overlaps(input_path, output_path)

    

if __name__ == "__main__":
    main()