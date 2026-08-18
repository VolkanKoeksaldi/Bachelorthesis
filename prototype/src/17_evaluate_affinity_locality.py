from experiment_config import experiment_path
import pandas as pd

DATASET = "mesh" # imdb or mesh



AFFINITY_LOCALITY_CONFIGS = {
    "mesh": {
        "affinity_path": experiment_path("workload_affinities/mesh_workload_affinities.csv"),
        "output_directory": experiment_path("affinity_evaluation"),
        "assignment_paths": {
            "round_robin" : experiment_path(
                "processed/mesh_fragment_assignment_round_robin.csv"),

            "tuple_ilp": experiment_path(
                "processed/mesh_fragment_assignment_tuple_ilp.csv"),

            "conflict_locality_ilp": experiment_path(
                "processed/mesh_fragment_assignment_conflict_locality_ilp.csv"),

            "conflict_locality_ilp_updated": experiment_path(
                "reoptimization/mesh_fragment_assignment_conflict_locality_ilp_updated.csv"),

            "tuple_ilp_updated": experiment_path(
                "reoptimization/mesh_fragment_assignment_tuple_ilp_updated.csv")
        }
    },

    "imdb": {
        "affinity_path": experiment_path("workload_affinities/imdb_workload_affinities.csv"),
        "output_directory": experiment_path("affinity_evaluation"),
        "assignment_paths": {
            "round_robin" : experiment_path(
                "processed/imdb_fragment_assignment_round_robin.csv"),

            "tuple_ilp": experiment_path(
                "processed/imdb_fragment_assignment_tuple_ilp.csv"),

            "conflict_locality_ilp": experiment_path(
                "processed/imdb_fragment_assignment_conflict_locality_ilp.csv"),

            "conflict_locality_ilp_updated": experiment_path(
                "reoptimization/imdb_fragment_assignment_conflict_locality_ilp_updated.csv"),

            "tuple_ilp_updated": experiment_path(
                "reoptimization/imdb_fragment_assignment_tuple_ilp_updated.csv")
        }
    }
}


def load_affinities(path):
    """
    Loads and validates the computed workload-based fragment affinities.

    Parameters:
        path: The path to the affinity CSV file
    
    Returns:
        A DataFrame that contains fragment_i, fragment_j, and affinity
    """

    if not path.exists():
        raise FileNotFoundError(f"Affinity file not found: {path}")

    affinity_df = pd.read_csv(path, dtype={"fragment_i": "string", "fragment_j": "string"})

    required_columns = {"fragment_i", "fragment_j", "affinity"}

    missing_columns = required_columns - set(affinity_df.columns)

    if missing_columns:
        raise ValueError(f"There are missing columns in file {path}: {sorted(missing_columns)}")

    if affinity_df.empty:
        raise ValueError(f"Affinity file has no affinity pairs: {path}")

    # errors="raise" raises an exception if there is an invalid parsing.
    affinity_df["affinity"] = pd.to_numeric(affinity_df["affinity"], errors="raise")

    if (affinity_df["affinity"] <= 0).any():
        raise ValueError(f"Affinity file contains zero or negative affinity value: {path}")

    return affinity_df


def load_assignment(assignment_path):
    """
    Loads and validates fragment assignment of one placement type.

    Parameters:
        assignment_path: Path to fragment assignment CSV file

    Returns:
        assignment_df: A DataFrame that maps every fragment id to one node id
    """

    if not assignment_path.exists():
        raise FileNotFoundError(f"Assignment file not found: {assignment_path}")

    assignment_df = pd.read_csv(assignment_path, dtype={"fragment_id": "string"})

    required_columns = {"fragment_id", "node_id"}

    missing_columns = required_columns - set(assignment_df.columns)

    if missing_columns:
        raise ValueError(f"There are missing columns in assignment file: "
                         f"{assignment_path}: {sorted(missing_columns)}")

    # Identifies fragment ids that appear multiple times in the assignment.
    duplicate_fragments = assignment_df["fragment_id"].duplicated(keep=False)

    if duplicate_fragments.any():
        duplicated_ids = sorted(assignment_df.loc[duplicate_fragments, "fragment_id"].unique())

        raise ValueError(f"Fragments occur more than once in {assignment_path}: "
                         f"{duplicated_ids}")

    return assignment_df


def evaluate(placement_type, assignment_df, affinity_df):
    """
    Evaluates the number of affinity fragment pairs assigned to the same node of one placement.

    The locality ratio is calculated as the proportion of total affinity weight
    whose pairs are assigned to the same node.

    Parameters:
        placement_type: Selected placement method
        assignment_df: The assignment from fragment to nodes
        affinity_df: The fragment affinities

    Returns:
        summary: The summary dictionary with evaluated metrics
        details_df: The detailed evaluation DataFrame
    """

    # Maps fragment id to the node where it was assigned to
    fragment_node = dict(zip(assignment_df["fragment_id"], assignment_df["node_id"]))

    details = []

    # Affinity pairs are evaluated
    for row in affinity_df.itertuples(index=False):
        fragment_i = row.fragment_i
        fragment_j = row.fragment_j
        affinity = row.affinity

        # Both fragments must occur in the assignment
        if fragment_i not in fragment_node:
            raise ValueError(f"{fragment_i} is missing from the assignment.")

        if fragment_j not in fragment_node:
            raise ValueError(f"{fragment_j} is missing from the assignment.")

        node_i = fragment_node[fragment_i]
        node_j = fragment_node[fragment_j]

        # Checks whether fragment_i and fragment_j are assigned on the same node.
        if node_i == node_j:
            same_node = True
        else:
            same_node = False

        details.append({
            "placement_type": placement_type,
            "fragment_i": fragment_i,
            "fragment_j": fragment_j,
            "affinity": affinity,
            "node_i": node_i,
            "node_j": node_j,
            "same_node": same_node
        })

    details_df = pd.DataFrame(details)

    same_node_mask = details_df["same_node"]

    # Counts colocated and separated distinct affinity pairs.
    same_node_pairs = int(same_node_mask.sum())
    separated_pairs = (len(details_df) - same_node_pairs)

    # Sum of affinities of all pairs.
    total_affinity = details_df["affinity"].sum()

    # Sum of affinities whose fragments are placed together on the same node.
    colocated_affinity = details_df.loc[same_node_mask, "affinity"].sum()

    # Sum of affinities whose fragments are placed on different nodes.
    separated_affinity = details_df.loc[details_df["same_node"]==False, "affinity"].sum()

    # Locality ratio represents workload weighted share of affinity that is colocated.
    if total_affinity == 0:
        locality_ratio = 0
    else:
        locality_ratio = colocated_affinity / total_affinity

    summary = {
        "placement_type": placement_type,
        "number_affinity_pairs": len(details_df),
        "same_node_pairs": same_node_pairs,
        "separated_pairs": separated_pairs,
        "total_affinity": total_affinity,
        "colocated_affinity": colocated_affinity,
        "separated_affinity": separated_affinity,
        "locality_ratio": locality_ratio
    }

    return summary, details_df

def save(summary, details, dataset, config):
    """
    Saves affinity-locality summary and detail results.
    """

    output_directory = config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    summary_path = output_directory/f"{dataset}_affinity_locality_summary.csv"
    details_path = output_directory/f"{dataset}_affinity_locality_details.csv"

    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)

    print(f"Affinity locality summary saved to: {summary_path}")
    print(f"Affinity locality details saved to: {details_path}")

    return summary_path, details_path

def process_evaluate_affinity_locality(dataset, config):
    """
    Evaluates affinity locality for every placement type and saves the combined results.
    The same workload-derived affinities are used for every placement method to ensure
    comparability.

    Returns:
        summary_df: Combined summary DataFrame
        detail_df: Combined details DataFrame
    """

    affinity = load_affinities(config["affinity_path"])

    summary_array = []
    detail_array = []

    # Same workload affinities are used for every placement type.
    for placement_type, path in config["assignment_paths"].items():
        if not path.exists():
            print(f"Skipping {placement_type} because assignment file not found: {path}")
            continue

        assignment_df = load_assignment(path)

        summary, detail = evaluate(placement_type=placement_type, 
                                   assignment_df=assignment_df, affinity_df=affinity)

        summary_array.append(summary)
        detail_array.append(detail)

    if not summary_array:
        raise FileNotFoundError(f"No available assignment files were found.")

    # Creates summary row per placement type
    summary_df = pd.DataFrame(summary_array)
    # Combines results into one detail table
    detail_df = pd.concat(detail_array, ignore_index=True)

    save(summary=summary_df, details=detail_df, dataset=dataset, config=config)

    return summary_df, detail_df


def main():

    if DATASET not in AFFINITY_LOCALITY_CONFIGS:
        raise ValueError(f"Unknown dataset {DATASET}")

    config = AFFINITY_LOCALITY_CONFIGS[DATASET]

    summary_df, detail_df = process_evaluate_affinity_locality(dataset=DATASET, config=config)

    print("\nAffinity-locality Summary:")
    print(summary_df)

    print("\nNumber of evaluated detail rows:", len(detail_df))

if __name__ == "__main__":
    main()