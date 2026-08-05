from pathlib import Path
from itertools import combinations

import pandas as pd

INPUT_PATH = Path(
    "prototype/output/processed/imdb_fragments.csv"
)

OUTPUT_PATH = Path(
    "prototype/output/processed/imdb_overlaps.csv"
)

UPDATED_INPUT_PATH = Path(
    "prototype/output/reoptimization/imdb_fragments_updates.csv"
)

UPDATED_OUTPUT_PATH = Path(
    "prototype/output/reoptimization/imdb_overlaps_updates.csv"
)

MODES = {
    "baseline": (INPUT_PATH, OUTPUT_PATH),
    "updates": (UPDATED_INPUT_PATH, UPDATED_OUTPUT_PATH),
}

MODE = "baseline"

item_ids_column = "title_ids"

# If set as True, then fragments belonging to the same scheme are compared.
# Since fragments within each IMDb scheme are disjoint, this should not produce additional overlaps.
# If set as False, then only fragments from different schemes are compared.
compare_same_scheme = False

def parse_title_ids(value):
    """
    Converts a title id string into a set
    """

    if pd.isna(value) or value == "":
        return set()

    return {
        title_id.strip() for title_id in str(value).split(",") if title_id.strip()
    }

def prepare_fragments(fragments_df):
    """
    Prepares the fragments for calculating overlaps by converting their title id strings into sets.
    """

    prepared = []

    # iterates over all fragments without including the index from the DataFrame
    for row in fragments_df.itertuples(index=False):

        # getattr() reads the attribute of item_ids_column in row.
        # it corresponds to row.title_ids
        title_ids = parse_title_ids(getattr(row, item_ids_column))

        prepared.append({
            "fragment_id": row.fragment_id,
            "scheme": row.scheme,
            "value": row.value,
            "fragment_size": row.fragment_size,
            "title_ids": title_ids
        })

    return prepared

def compute_overlaps(prepared, compare_same_scheme):
    """
    Compares fragment pairs and returns the pairs that have an overlap.
    """

    overlap_rows = []
    
    # Generates every unordered pair of distinct fragments once.
    for f1, f2 in combinations(prepared, 2):

        # Skips pairs from same fragmentation scheme
        if (not compare_same_scheme and f1["scheme"] == f2["scheme"]):
            continue
        
        # Calculates overlap of two title id sets
        overlap = f1["title_ids"].intersection(f2["title_ids"])

        if overlap:
            overlap_rows.append({
                "fragment_1": f1["fragment_id"],
                "scheme_1": f1["scheme"],
                "value_1": f1["value"],
                "fragment_2": f2["fragment_id"],
                "scheme_2": f2["scheme"],
                "value_2": f2["value"],
                "overlap_size": len(overlap),
                "overlap_title_ids": ",".join(sorted(overlap))
            })

    return pd.DataFrame(overlap_rows)


def process_imdb_overlaps(input_path, output_path, compare_same_scheme):
    """
    Loads fragments and then validates the required columns.
    Afterwards, computes their overlaps and stores the result as a CSV.
    """

    fragments_df = pd.read_csv(input_path)

    required_columns = {"fragment_id", "scheme", "value", "fragment_size", "title_ids"}

    missing = (required_columns - set(fragments_df.columns))

    if missing:
        raise ValueError(f"These columns from fragment file are still missing: {sorted(missing)}")

    prepared_fragments = prepare_fragments(fragments_df)

    overlaps_df = compute_overlaps(prepared_fragments, compare_same_scheme)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlaps_df.to_csv(output_path, index=False)

    print(f"Number of fragments: {len(fragments_df)}")
    print(f"NUmber of overlap pairs in fragments: {len(overlaps_df)}")

    if not overlaps_df.empty:
        print(f"Sum of overlap sizes: {overlaps_df['overlap_size'].sum()}")

        print("Largest overlap:")
        # sorts overlap pairs by descreasing overlap size and then prints the ten largest values
        print(overlaps_df.sort_values("overlap_size", ascending=False).head(10))

    print(f"Output saved to: {output_path}")

    return overlaps_df

def main():
    if MODE not in MODES:
        raise ValueError(f"Unknown mode: {MODE}")

    selected_input_path, selected_output_path = MODES[MODE]

    print(f"Mode: {MODE}")

    process_imdb_overlaps(
        selected_input_path,
        selected_output_path,
        compare_same_scheme
    )

if __name__ == "__main__":
    main()