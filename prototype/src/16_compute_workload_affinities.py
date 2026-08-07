from pathlib import Path
from experiment_config import experiment_path
from itertools import combinations
import json
import pandas as pd

AFFINITY_CONFIGS = {
    "mesh": {
        "workload_path": experiment_path("workloads/mesh_workload.json"),
        "output_directory": experiment_path("workload_affinities"),
        "overlap_path": experiment_path("processed/mesh_overlaps_sample.csv"),

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

DATASET = "imdb"

def load_workload(path):
    """
    Loads the generated workload.
    """

    if not path.exists():
        raise FileNotFoundError(f"Expected workload file not found: {path}.")

    with path.open("r", encoding="utf-8") as file:
        workload = json.load(file)

    # workload must consist of a list of operation objects
    if not isinstance(workload, list):
        raise ValueError(f"Workload file must contain a JSON list: {path}")

    if not workload:
        raise ValueError(f"Workload file contains no operations: {path}")

    if not all(isinstance(operation, dict) for operation in workload):
        raise ValueError(f"Every workload operation must be an object.")

    return workload


def normalize_pair(fragment_i, fragment_j):
    """
    Returns a sorted fragment pair so order does not matter.
    """
    return tuple(sorted((fragment_i, fragment_j)))

def compute_affinities(workload):
    """
    Counts how often each fragment pair appears together in FRAGMENT_SELECT operations.
    Using that the affinities are computed.
    """

    affinities = {}

    for operation in workload:
        # Only considers FRAGMENT_SELECT operations for the affinity generation
        if operation["operation"] != "FRAGMENT_SELECT":
            continue

        fragment_ids = operation["fragment_ids"]

        if not isinstance(fragment_ids, list):
            raise ValueError(f"Every FRAGMENT_SELECT operation must contain a fragment_ids list.")

        # Removes duplicate fragment ids within the same query
        fragment_ids = sorted(set(fragment_ids))

        if len(fragment_ids) < 2:
            raise ValueError(f"Every FRAGMENT_SELECT operation must contain"
                             f" at least two different fragment ids.")

        # Generates every unordered fragment pair contained in the query
        for fragment_i, fragment_j in combinations(fragment_ids, 2):
            pair = normalize_pair(fragment_i, fragment_j)
            if pair not in affinities:
                affinities[pair] = 0

            affinities[pair] += 1

    return affinities

def create_affinity_df(affinities):
    """
    Creates a table containing the computed fragment affinities.
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
        # Ascending order is False, this means, that highest affinities are shown first
        affinity_df = affinity_df.sort_values(by=["affinity", "fragment_i", "fragment_j"], 
                                              ascending=False).reset_index(drop=True)

    return affinity_df

def compare_affinity_conflict(affinity, config):
    """
    Compares affinity pairs with fragment overlap conflict pairs.
    """

    overlap_path = config["overlap_path"]
    fragment_1 = config["overlap_fragment_1"]
    fragment_2 = config["overlap_fragment_2"]

    if not overlap_path.exists():
        raise FileNotFoundError(f"Expected file not found: {overlap_path}")

    overlap_df = pd.read_csv(overlap_path)

    required_columns = {fragment_1, fragment_2,}

    missing_columns = required_columns - set(overlap_df.columns)

    if missing_columns:
        raise ValueError(f"There are missing columns in the overlap file {overlap_path}: "
                         f"{sorted(missing_columns)}")

    # Fragment pairs that occur together in a FRAGMENT_SELECT query.
    affinity_pairs = {
        normalize_pair(row["fragment_i"], row["fragment_j"])
        for _, row in affinity.iterrows()
    }

    # Every fragment pair in overlap file represents a hard conflict -> conflict_pairs
    conflict_pairs = {
        normalize_pair(row[fragment_1], row[fragment_2])
        for _, row in overlap_df.iterrows()
    }

    # affine fragments that overlap but cannot be assigned to the same node
    affinity_conflicts = affinity_pairs & conflict_pairs

    # affine fragment pairs that can be placed on the same node
    non_conflict_affinities = affinity_pairs - conflict_pairs

    comparison = {
        "amount_affinity_pairs": len(affinity_pairs),
        "amount_conflict_affinities": len(affinity_conflicts),
        "amount_non_conflict_affinities": len(non_conflict_affinities)
    }

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
    Loads workload, computes and compares its affinities, and then saves the results.
    """

    workload = load_workload(config["workload_path"])

    affinities = compute_affinities(workload)

    affinity_df = create_affinity_df(affinities)

    comparison, non_conflict_affinities = compare_affinity_conflict(affinity=affinity_df, config=config)

    save(affinity_df, dataset, config)

    return affinity_df, comparison, non_conflict_affinities


def main():
    if DATASET not in AFFINITY_CONFIGS:
        raise ValueError(f"Unknown dataset in affinity configuration: {DATASET}")

    config = AFFINITY_CONFIGS[DATASET]

    affinity_df, comparison, non_conflict_affinities = process_compute_workload_affinities(dataset=DATASET, 
                                                                                           config=config)

    print("\nNumber of different affinity pairs: ", len(affinity_df))

    print("\nTop 5 fragment pairs with highest affinity: ")
    print(affinity_df.head(5))

    print("\nAffinity pairs in total:", comparison["amount_affinity_pairs"])

    print("\nAffinity pairs that are also conflict pairs:", comparison["amount_conflict_affinities"])

    print("\nNon-Conflict pairs:", comparison["amount_non_conflict_affinities"])
    
    for pair in sorted(non_conflict_affinities):
        print("\n Paare:", pair)


if __name__ == "__main__":
    main()