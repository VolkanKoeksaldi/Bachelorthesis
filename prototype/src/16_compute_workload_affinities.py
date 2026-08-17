from experiment_config import experiment_path
from itertools import combinations
import json
import pandas as pd

DATASET = "mesh" # imdb or mesh


AFFINITY_CONFIGS = {
    "mesh": {
        "workload_path": experiment_path("workloads/mesh_workload.json"),
        "output_directory": experiment_path("workload_affinities"),
        "overlap_path": experiment_path("processed/mesh_overlaps.csv"),

        "overlap_fragment_1": "fragment_1",
        "overlap_fragment_2": "fragment_2"
    },

    "imdb": {
        "workload_path": experiment_path("workloads/imdb_workload.json"),
        "output_directory": experiment_path("workload_affinities"),
        "overlap_path": experiment_path("processed/imdb_overlaps.csv"),

        "overlap_fragment_1": "fragment_1",
        "overlap_fragment_2": "fragment_2"
    }
}

def load_workload(path):
    """
    Loads and validates the generated workload from a JSON file.

    Parameters:
        path: Path to the workload JSON file

    Returns:
        workload: A non-empty list of workload-operation dictionaries
    """

    if not path.exists():
        raise FileNotFoundError(f"Expected workload file not found: {path}.")

    with path.open("r", encoding="utf-8") as file:
        workload = json.load(file)

    # Workload must consist of a list of operation objects
    if not isinstance(workload, list):
        raise ValueError(f"Workload file must contain a JSON list: {path}")

    if not workload:
        raise ValueError(f"Workload file contains no operations: {path}")

    if not all(isinstance(operation, dict) for operation in workload):
        raise ValueError(f"Every workload operation must be an object.")

    return workload


def normalize_pair(fragment_i, fragment_j):
    """
    Returns a sorted fragment pair so that (i,j) and (j,i) are treated as the same pair.
    """
    return tuple(sorted((fragment_i, fragment_j)))

def compute_affinities(workload):
    """
    Computes workload-based affinities from FRAGMENT_SELECT operations.

    Each unordered pair of distinct fragments receives an affinity increase of one, for every
    reference in FRAGMENT_SELECT operations. The final affinity represents the number of 
    operations in which the fragment pair was accessed together.

    Parameters:
        workload: List of workload operation dictionaries

    Returns:
        A dictionary that maps fragment pairs to affinity values
    """

    affinities = {}

    for operation in workload:
        # Only considers FRAGMENT_SELECT operations for the affinity generation
        if operation["operation"] != "FRAGMENT_SELECT":
            continue

        fragment_ids = operation["fragment_ids"]

        if not isinstance(fragment_ids, list):
            raise ValueError(f"Every FRAGMENT_SELECT operation must contain "
                             "a fragment_ids list.")

        # Removes duplicate fragment ids within the same operation.
        fragment_ids = sorted(set(fragment_ids))

        if len(fragment_ids) < 2:
            raise ValueError(f"Every FRAGMENT_SELECT operation must contain"
                             f" at least two different fragment ids.")

        # Generates every unordered fragment pair contained in the operation.
        for fragment_i, fragment_j in combinations(fragment_ids, 2):
            pair = normalize_pair(fragment_i, fragment_j)
            if pair not in affinities:
                affinities[pair] = 0

            affinities[pair] += 1

    return affinities

def create_affinity_df(affinities):
    """
    Converts computed affinity dictionary into a sorted DataFrame.

    Returns:
        A DataFrame that contains fragment_i, fragment_j, and their affinity
    """

    rows = []

    for pair, affinity in affinities.items():
        fragment_i = pair[0]
        fragment_j = pair[1]

        rows.append({"fragment_i": fragment_i, "fragment_j": fragment_j, "affinity": affinity})

    affinity_df = pd.DataFrame(rows)

    if affinity_df.empty:
        raise ValueError(f"No fragment affinities could be computed from the workload.")
    else:
        # Sorts pairs by descending affinitiy. Ties are sorted by their fragment ids
        affinity_df = affinity_df.sort_values(by=["affinity", "fragment_i", "fragment_j"], 
                                              ascending=False).reset_index(drop=True)

    return affinity_df

def compare_affinity_conflict(affinity, config):
    """
    Compares affinity pairs with fragment overlap conflict pairs.

    Affinity-conflict pairs are accessed together by the workload but they cannot be
    colocated because of their overlap.

    Parameters:
        affinity: A DataFrame that contains the computed affinities
        config: Configurations for dataset-specific paths and parameters

    Returns:
        comparison: Comparison counts
        non_conflict_affinities: Set of affinity pairs that are not conflict pairs

    """

    overlap_path = config["overlap_path"]
    fragment_1 = config["overlap_fragment_1"]
    fragment_2 = config["overlap_fragment_2"]

    if not overlap_path.exists():
        raise FileNotFoundError(f"Expected file not found: {overlap_path}")

    overlap_df = pd.read_csv(overlap_path)

    required_columns = {fragment_1, fragment_2}

    missing_columns = required_columns - set(overlap_df.columns)

    if missing_columns:
        raise ValueError(f"There are missing columns in the overlap file "
                         f"{overlap_path}: {sorted(missing_columns)}")

    # Pairs that occur together in at least one FRAGMENT_SELECT operation.
    affinity_pairs = {normalize_pair(row["fragment_i"], row["fragment_j"]) 
                      for _, row in affinity.iterrows()}

    # Every pair in the overlap file represents a hard conflict -> conflict_pairs
    conflict_pairs = {normalize_pair(row[fragment_1], row[fragment_2]) 
                      for _, row in overlap_df.iterrows()}

    # Affine pairs that overlap and therefore cannot be assigned to the same node
    affinity_conflicts = affinity_pairs & conflict_pairs

    # Affine pairs that can be placed on the same node
    non_conflict_affinities = affinity_pairs - conflict_pairs

    comparison = {"amount_affinity_pairs": len(affinity_pairs),
                  "amount_conflict_affinities": len(affinity_conflicts),
                  "amount_non_conflict_affinities": len(non_conflict_affinities)}

    return comparison, non_conflict_affinities

def save(affinity, dataset, config):
    """
    Saves computed fragment affinities as CSV file.
    """

    output_directory = config["output_directory"]

    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / f"{dataset}_workload_affinities.csv"

    affinity.to_csv(output_path, index=False)

    print(f"Workload affinities saved to: {output_path}")

    return output_path

def process_compute_workload_affinities(dataset, config):
    """
    Loads workload, computes fragment affinities,
    compares them with overlap conflicts, and then saves the results.

    Returns:
        affinity_df: Affinity DataFrame
        comparison: Comparison counts
        non_conflict_affinities: Set of affinity pairs that are not conflict pairs
    """

    workload = load_workload(config["workload_path"])

    affinities = compute_affinities(workload)

    affinity_df = create_affinity_df(affinities)

    comparison, non_conflict_affinities = compare_affinity_conflict(affinity=affinity_df, 
                                                                    config=config)

    save(affinity_df, dataset, config)

    return affinity_df, comparison, non_conflict_affinities


def main():
    if DATASET not in AFFINITY_CONFIGS:
        raise ValueError(f"Unknown dataset in affinity configuration: {DATASET}")

    config = AFFINITY_CONFIGS[DATASET]

    affinity_df, comparison, non_conflict_affinities = process_compute_workload_affinities(
        dataset=DATASET, config=config)

    print("\nNumber of different affinity pairs: ", len(affinity_df))

    print("\nTop 5 fragment pairs with the highest affinity: ")
    print(affinity_df.head(5))

    print("\nAffinity pairs in total:", comparison["amount_affinity_pairs"])

    print("\nAffinity pairs that are also conflict pairs:", 
          comparison["amount_conflict_affinities"])

    print("\nNon-Conflict pairs:", comparison["amount_non_conflict_affinities"])
    
    for pair in sorted(non_conflict_affinities):
        print("\n Pair:", pair)


if __name__ == "__main__":
    main()