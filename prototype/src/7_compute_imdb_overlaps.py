from pathlib import Path
from itertools import combinations
from experiment_config import IMDB_FRAGMENTATION_SCHEMES, experiment_path
from clustering_utils import parse_item_ids, validate_fragmentation_memberships
import pandas as pd


MODE = "baseline" # baseline or updates

item_ids_column = "title_ids"



INPUT_PATH = experiment_path("processed/imdb_fragments.csv")

OUTPUT_PATH = experiment_path("processed/imdb_overlaps.csv")

UPDATED_INPUT_PATH = experiment_path("reoptimization/imdb_fragments_updates.csv")

UPDATED_OUTPUT_PATH = experiment_path("reoptimization/imdb_overlaps_updates.csv")

MODES = {
    "baseline": (INPUT_PATH, OUTPUT_PATH),
    "updates": (UPDATED_INPUT_PATH, UPDATED_OUTPUT_PATH),
}

# Defines output columns so that the column headers are also written when no
# overlapping pairs were found.
OUTPUT_COLUMNS = ["fragment_1", "scheme_1", "value_1", "fragment_2", 
                  "scheme_2", "value_2", "overlap_size", "overlap_title_ids"]

def prepare_fragments(fragments_df):
    """
    Prepares the fragments for calculating overlaps by converting
    their comma-separated title ID strings into sets.

    Parameters:
        fragments_df: DataFrame that contains the fragments
    
    Returns:
        prepared_fragments: A list of prepared fragment dictionaries
    """

    prepared_fragments = []

    # Iterates over all fragments without including the DataFrame index.
    for row in fragments_df.itertuples(index=False):

        # Reads the comma-separated title ids from item_ids_column and converts them into a set.
        title_ids = parse_item_ids(getattr(row, item_ids_column))

        # Ensures that the stored fragment size equals 
        # the number of unique title IDs in the fragment.
        if len(title_ids) != int(row.fragment_size):
            raise ValueError(f"Fragment {row.fragment_id} "
                             f"has a fragment size of {row.fragment_size}, but contains "
                             f"{len(title_ids)} unique title IDs.")

        prepared_fragments.append({"fragment_id": str(row.fragment_id),
                                   "scheme": str(row.scheme),
                                   "value": str(row.value),
                                   "title_ids": title_ids})

    return prepared_fragments


def compute_overlaps(prepared_fragments):
    """
    Computes non-empty overlaps between fragment pairs from different schemes.
    
    Parameters:
        prepared_fragments: A list of the prepared fragment dictionaries
    
    Returns:
        A DataFrame that contains the fragment pairs with overlaps
    """

    overlap_rows = []

    # Generates every pair of distinct fragments exactly once.
    for fragment_1, fragment_2 in combinations(prepared_fragments, 2):

        # Fragments within the same fragmentation scheme are disjoint.
        # Therefore they are skipped.
        if fragment_1["scheme"] == fragment_2["scheme"]:
            continue

        # Calculates the intersection of the two title ID sets.
        overlap = fragment_1["title_ids"] & fragment_2["title_ids"]

        # Fragment pairs without shared titles are not stored.
        if not overlap:
            continue

        overlap_rows.append({
            "fragment_1": fragment_1["fragment_id"],
            "scheme_1": fragment_1["scheme"],
            "value_1": fragment_1["value"],
            "fragment_2": fragment_2["fragment_id"],
            "scheme_2": fragment_2["scheme"],
            "value_2": fragment_2["value"],
            "overlap_size": len(overlap),
            "overlap_title_ids": ",".join(sorted(overlap))
        })

    return pd.DataFrame(overlap_rows, columns=OUTPUT_COLUMNS)


def process_imdb_overlaps(input_path: Path, output_path: Path):
    """
    Loads and validates the IMDb fragments, computes their overlaps, validates
    the overlap sum, and stores the overlap results as a CSV file.

    Parameters:
        input_path: Fragment CSV file path
        output_path: Output path for the overlap CSV file

    Returns:
        overlaps_df: A DataFrame that contains the fragment overlaps
    """

    if not input_path.exists():
        raise FileNotFoundError(f"IMDb fragment file was not found: {input_path}")

    fragments_df = pd.read_csv(input_path, dtype={"fragment_id": "string", 
                                                  "title_ids": "string"})

    required_columns = {"fragment_id", "scheme", "value", "fragment_size", item_ids_column}

    missing_columns = (required_columns - set(fragments_df.columns))

    if missing_columns:
        raise ValueError(f"The IMDb fragment file is missing the following columns: "
                         f"{sorted(missing_columns)}")

    if fragments_df.empty:
        raise ValueError(f"The IMDb fragment file is empty: {input_path}")

    if not fragments_df["fragment_id"].is_unique:
        raise ValueError("The IMDb fragment file contains duplicate fragment IDs.")

    expected_schemes = set(IMDB_FRAGMENTATION_SCHEMES)

    actual_schemes = set(fragments_df["scheme"].astype(str))

    if actual_schemes != expected_schemes:
        raise ValueError(
            f"Unexpected IMDb fragmentation schemes. Expected: "
            f"{sorted(expected_schemes)}, found: {sorted(actual_schemes)}.")

    prepared_fragments = prepare_fragments(fragments_df)

    expected_title_ids = set()

    # Collects all unique title ids from all fragments
    for fragment in prepared_fragments:
        expected_title_ids.update(fragment["title_ids"])

    if not expected_title_ids:
        raise ValueError("The IMDb fragments do not contain any title IDs.")

    # Validates that each title belongs to exactly one fragment
    # in each of the fragmentation schemes.
    validate_fragmentation_memberships(fragments_df, expected_item_ids=expected_title_ids, 
                                       expected_schemes=IMDB_FRAGMENTATION_SCHEMES,
                                       item_ids_column=item_ids_column)

    # Calculates pairs with non empty overlaps.
    overlaps_df = compute_overlaps(prepared_fragments)

    # Each title contributes to exactly one overlap for each fragment pair of 
    # fragmentation schemes because every fragmentation is complete and internally disjoint.
    number_of_schemes = len(IMDB_FRAGMENTATION_SCHEMES)

    # Calculates the number of possible scheme pairs.
    scheme_pair_count = (number_of_schemes * (number_of_schemes - 1) // 2)

    # Calculates the expected sum of all overlap sizes.
    expected_overlap_sum = (len(expected_title_ids) * scheme_pair_count)

    # Calculates the actual sum of all overlap sizes.
    if not overlaps_df.empty:
        actual_overlap_sum = int(overlaps_df["overlap_size"].sum())
    else: actual_overlap_sum = 0

    if actual_overlap_sum != expected_overlap_sum:
        raise ValueError(f"The sum of overlap sizes is {actual_overlap_sum},"
                         f" but {expected_overlap_sum} was expected.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlaps_df.to_csv(output_path, index=False)

    print(f"Number of titles: {len(expected_title_ids)}")
    print(f"Number of fragments: {len(fragments_df)}")
    print(f"Number of overlap pairs: {len(overlaps_df)}")
    print(f"Sum of pairwise overlap sizes: {actual_overlap_sum}")

    if not overlaps_df.empty:
        print("Largest overlaps:")
        print(overlaps_df.sort_values("overlap_size", ascending=False).head(10))

    print(f"Output saved to: {output_path}")

    return overlaps_df

def main():
    if MODE not in MODES:
        raise ValueError(f"Unknown mode: {MODE}")

    selected_input_path, selected_output_path = MODES[MODE]

    print(f"Mode: {MODE}")

    process_imdb_overlaps(selected_input_path, selected_output_path)

if __name__ == "__main__":
    main()