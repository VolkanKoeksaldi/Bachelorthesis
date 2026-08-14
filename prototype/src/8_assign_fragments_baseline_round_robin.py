from pathlib import Path
import pandas as pd
from experiment_config import experiment_path, NUM_NODES

DATASET = "mesh"

CONFIGS = {
    "mesh": {
        "input_path": experiment_path("processed/mesh_fragments.csv"),
        "output_path": experiment_path("processed/mesh_fragment_assignment_round_robin.csv")
    },

    "imdb": {
        "input_path": experiment_path("processed/imdb_fragments.csv"),
        "output_path": experiment_path("processed/imdb_fragment_assignment_round_robin.csv")
    }
}



def assign_round_robin(fragments_df, num_nodes):
    """
    Assigns fragments to nodes using round-robin.
    Represents a simple deterministic baseline for comparisons with the ILP methods.

    Example with 4 nodes for cyclical assignment:
    Fragment 0 -> node_1
    Fragment 1 -> node_2
    Fragment 2 -> node_3
    Fragment 3 -> node_4
    Fragment 4 -> node_1
    """

    if num_nodes <= 0:
        raise ValueError("Number of nodes must be greater than 0.")

    if "node_id" in fragments_df.columns:
        raise ValueError("The fragment input already contains a node_id column.")

    # Creates a copy with a new index for the assignment DataFrame
    # drop=True prevents the old index from becoming an additional column.
    assignment_df = fragments_df.copy().reset_index(drop=True)

    # Assigns fragments cyclically to the available nodes.
    # The modulo operator restarts the assignment after the last node.
    assignment_df["node_id"] = [f"node_{(index % num_nodes) + 1}" 
                                for index in range(len(assignment_df))]

    return assignment_df


def process_round_robin(input_path: Path, output_path: Path, num_nodes: int):
    """
    Loads the fragments, validates the input, and assigns the fragments
    according to the round-robin strategy.

    All original fragment columns are preserved and only node_id is added.
    """

    if not input_path.exists():
        raise FileNotFoundError(f"Fragment file was not found: {input_path}")

    fragments_df = pd.read_csv(input_path)

    required_columns = {"fragment_id"}

    missing_columns = required_columns - set(fragments_df.columns)

    if missing_columns:
        raise ValueError(f"The fragment file is missing the following columns: "
                         f"{sorted(missing_columns)}")

    if fragments_df.empty:
        raise ValueError(f"The fragment file is empty: {input_path}")

    # Fragment IDs must not be missing or empty.
    fragment_ids = (fragments_df["fragment_id"].astype("string"))
    # .eq("") checks here whether the fragment_ids still contains empty strings
    invalid_fragment_ids = (fragment_ids.isna()| fragment_ids.str.strip().eq(""))

    if invalid_fragment_ids.any():
        raise ValueError("The fragment file contains missing or empty fragment IDs.")

    # Every Fragment must have a unique fragment id
    if not fragment_ids.is_unique:
        raise ValueError("The fragment file contains duplicate fragment IDs.")

    assignment_df = assign_round_robin(fragments_df, num_nodes)

    # The assignment must contain all fragments.
    if len(assignment_df) != len(fragments_df):
        raise ValueError("The number of assigned fragments does not match "
                         f"to the number of input fragments.")

    # The output consists of all input columns plus node_id.
    expected_columns = [*fragments_df.columns, "node_id"]

    if assignment_df.columns.tolist() != expected_columns:
        raise ValueError("The round-robin assignment has an unexpected column.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    assignment_df.to_csv(output_path, index=False)

    # Displays the number of assigned fragments for every node.
    node_order = [f"node_{node_number}"  for node_number in range(1, num_nodes + 1)]

    # Counts the assigned fragments per node from node_id column
    # then reindexes it to node_order and gives missing nodes the value 0.
    # Renames the result to fragment_count
    fragments_per_node = (assignment_df["node_id"].value_counts()
                          .reindex(node_order, fill_value=0).rename("fragment_count"))

    print(f"Dataset: {DATASET}")
    print(f"Input fragments: {len(fragments_df)}")
    print(f"Number of nodes: {num_nodes}")
    print()
    print(assignment_df.head(20))
    print()
    print("Fragments per node:")
    print(fragments_per_node)
    print()
    print(f"Saved to: {output_path}")

    return assignment_df


def main():
    if DATASET not in CONFIGS:
        raise ValueError(f"Unknown dataset: {DATASET}")

    config = CONFIGS[DATASET]

    process_round_robin(config["input_path"], config["output_path"], NUM_NODES)


if __name__ == "__main__":
    main()