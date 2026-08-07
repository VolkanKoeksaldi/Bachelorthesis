from pathlib import Path
import pandas as pd
from experiment_config import experiment_path


CONFIGS = {
    "mesh": {
        "input_path": experiment_path("processed/mesh_fragments_sample.csv"),
        "output_path": experiment_path("processed/mesh_fragment_assignment_round_robin.csv")
    },

    "imdb": {
        "input_path": experiment_path("processed/imdb_fragments.csv"),
        "output_path": experiment_path("processed/imdb_fragment_assignment_round_robin.csv")
    }
}

DATASET = "mesh"

NUM_NODES = 10

def assign_round_robin(fragments_df, num_nodes):
    """
    Assigns fragments to nodes using round-robin.
    Serves as a basic method for comparison with ILP-based placement methods.
    Example with 4 nodes:
    Fragment 0 -> node_1
    Fragment 1 -> node_2
    Fragment 2 -> node_3
    Fragment 3 -> node_4
    Fragment 4 -> node_1
    """

    if num_nodes <= 0:
        raise ValueError(f"Number of nodes must be greater than 0.")

    # Creates copy with new consecutive index
    # drop=True prevents then that the old index is used as an additional column
    assignment_df = fragments_df.copy().reset_index(drop=True)

    # Assigns fragments in a cycle to available nodes
    # % restarts the assignment after last node
    assignment_df["node_id"] = [f"node_{(index % num_nodes) + 1}" for index in range(len(assignment_df))]

    return assignment_df

def process_round_robin(input_path: Path, output_path: Path, num_nodes: int):
    """
    Loads fragments and then
    executes the steps of assigning fragments according to round-robin strategy.
    """

    fragments_df = pd.read_csv(input_path)

    assignment_df = assign_round_robin(fragments_df, num_nodes)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    assignment_df.to_csv(output_path, index=False)

    print("Input fragments:", len(fragments_df))
    print("Number of nodes:", num_nodes)
    print()
    print(assignment_df.head(20))
    print()
    print("Fragments per node:")
    print(assignment_df["node_id"].value_counts())
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