from pathlib import Path
import pandas as pd
import json
import random
from experiment_config import experiment_path


DATASETS = {
    "mesh": {
        "items_path": experiment_path("processed/mesh_descriptors_sample.csv"),
        "fragments_path": experiment_path("processed/mesh_fragments_sample.csv"),
        "output_path": experiment_path("workloads/mesh_workload.json"),
        "item_id_column": "descriptor_ui",
        "item_name_column": "descriptor_name",
        "new_item_name": "new_descriptor_name",
        "generated_id_prefix": "D_WORKLOAD_",
        "generated_name_prefix": "Workload Generated Descriptor",
        "updated_name_prefix": "Updated Workload Generated Descriptor",
        "fragmentation_schemes": [
            "top_category",
            "branch_code",
            "subbranch_code"
        ]
    },

    "imdb": {
        "items_path": experiment_path("processed/imdb_titles.csv"),
        "fragments_path": experiment_path("processed/imdb_fragments.csv"),
        "output_path": experiment_path("workloads/imdb_workload.json"),
        "item_id_column": "title_id",
        "item_name_column": "primary_title",
        "new_item_name": "new_primary_title",
        "generated_id_prefix": "T_WORKLOAD_",
        "generated_name_prefix": "Workload Generated Title",
        "updated_name_prefix": "Updated Workload Generated Title",
        "fragmentation_schemes": [
            "title_type",
            "decade",
            "primary_genre"
        ]
    },
}

DATASET = "mesh"

random_seed = 42
number_blocks_operations = 20

def load_workload_inputs(config):
    """
    Loads existing item ids and groups available fragments by scheme.
    """
    # usecols defines which columns are relevant to be loaded
    items_df = pd.read_csv(config["items_path"], usecols=[config["item_id_column"]])
    fragments_df = pd.read_csv(config["fragments_path"], usecols=["fragment_id", "scheme"])

    # extracts item_ids using dropna() to remove missing values and make a list of it using tolist()
    item_ids = (items_df[config["item_id_column"]].dropna().astype(str).unique().tolist())

    # groups fragments by scheme
    fragments_by_scheme = {scheme: group["fragment_id"].astype(str).tolist()
                           for scheme, group in fragments_df.groupby("scheme")}

    return item_ids, fragments_by_scheme


def generate_workload(item_ids, fragments_by_scheme, config, number_blocks_operations=number_blocks_operations, seed=random_seed):
    """
    Generates a reproducible mixed workload.
    Each workload block contains:
    SELECT, INSERT, SELECT, UPDATE, DELETE, and FRAGMENT SELECT

    Every inserted item is assigned to exactly one fragment from each configured fragmentation scheme.
    FRAGMENT_SELECT operations are used later to derive workload-based fragment affinities.
    """

    if not item_ids:
        raise ValueError(f"No item ids were provided.")
    
    rng = random.Random(seed)

    workload = []

    required_schemes = config["fragmentation_schemes"]

    if len(required_schemes) < 2:
        raise ValueError(f"At least two fragmentation schemes are required for FRAGMENT_SELECT operations.")

    missing_schemes = [scheme for scheme in required_schemes
                       if not fragments_by_scheme.get(scheme)]

    if missing_schemes:
        raise ValueError(f"No fragments were found for the required schemes: {missing_schemes}")
    
    for num in range(1, number_blocks_operations + 1):
        # Chooses a random element
        existing_item = rng.choice(item_ids)

        new_item = f"{config['generated_id_prefix']}{num:03d}"

        # Assigns the new item to exactly one fragment from each scheme.
        assigned_fragments = [rng.choice(fragments_by_scheme[scheme]) for scheme in required_schemes]

        # Selects two assigned fragments for workload-based affinity generation
        selected_fragments = rng.sample(assigned_fragments, 2)

        workload.append({"operation": "SELECT", config["item_id_column"]: existing_item})

        workload.append({"operation": "INSERT",
                         config["item_id_column"]: new_item,
                         config["item_name_column"]: f"{config['generated_name_prefix']} {num}",
                         "fragment_ids": assigned_fragments})

        workload.append({"operation": "SELECT", config["item_id_column"]: new_item})

        workload.append({"operation": "UPDATE",
                         config["item_id_column"]: new_item,
                         config["new_item_name"]: f"{config['updated_name_prefix']} {num}"})

        workload.append({"operation": "DELETE", config["item_id_column"]: new_item})

        workload.append({"operation": "FRAGMENT_SELECT", "fragment_ids": selected_fragments})

    output_path = config["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # indent=4 in order to make the file more readable
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(workload, file, indent=4)

    return workload

def main():
    if DATASET not in DATASETS:
        raise ValueError(f"Unknown dataset {DATASET}")

    config = DATASETS[DATASET]

    item_ids, fragments_by_scheme = (load_workload_inputs(config))

    workload = generate_workload(item_ids=item_ids, fragments_by_scheme=fragments_by_scheme,
                                 config=config, number_blocks_operations=number_blocks_operations,
                                 seed=random_seed)
    
    print("Number of operations:", len(workload))
    print("Saved to:", config["output_path"])

    print("First five operations:")
    for operation in workload[:5]:
        print(operation)

if __name__ == "__main__":
    main()