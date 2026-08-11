from pathlib import Path
import pandas as pd
import json
import random
from experiment_config import experiment_path
from itertools import combinations


DATASETS = {
    "mesh": {
        "items_path": experiment_path("processed/mesh_terms.csv"),
        "fragments_path": experiment_path("processed/mesh_fragments.csv"),
        "output_path": experiment_path("workloads/mesh_workload.json"),
        "item_id_column": "tuple_id",
        "item_name_column": "mesh_term",
        "new_item_name": "new_mesh_term",
        "generated_id_prefix": "MT_WORKLOAD_",
        "generated_name_prefix": "Workload Generated MeSH Term",
        "updated_name_prefix": "Updated Workload Generated MeSH Term",
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

DATASET = "imdb"

random_seed = 42
number_blocks_operations = 200
number_fragment_selects = 500

number_affinity_pairs = 30

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

def select_affinity_pairs(fragments_by_scheme, required_schemes, number_affinity_pairs, rng):
    """
    Selects a random set of fragment pairs from different schemes.
    """
    possible_pairs = []

    # considers every unordered combination of two different schemes
    for scheme_i, scheme_j in combinations(required_schemes, 2):
        # generates all possible fragment pairs for each scheme_i and scheme_j
        for fragment_i in fragments_by_scheme[scheme_i]:

            for fragment_j in fragments_by_scheme[scheme_j]:
                
                possible_pairs.append((fragment_i, fragment_j))

    # Ensures that enough distinct fragment pairs are in the array of possible_pairs
    if len(possible_pairs) < number_affinity_pairs:
        raise ValueError(f"Only {len(possible_pairs)} fragment pairs are available, but {number_affinity_pairs} are required.")

    # Then selects a random sample of unique pairs
    return rng.sample(possible_pairs, number_affinity_pairs)    

def generate_fragment_select(affinity_pairs, number_fragment_selects, rng):
    """
    Generates the random FRAGMENT_SELECT operation for calculating affinity from the selected affinity pairs.
    """

    fragment_selects = []
    for _ in range(number_fragment_selects):
        fragment_pair = rng.choice(affinity_pairs)

        fragment_selects.append({"operation": "FRAGMENT_SELECT", "fragment_ids": list(fragment_pair)})

    return fragment_selects

def generate_workload(item_ids, fragments_by_scheme, config, number_blocks_operations=number_blocks_operations,
                      number_fragment_selects=number_fragment_selects, number_affinity_pairs=number_affinity_pairs, seed=random_seed):
    """
    Generates a reproducible mixed workload.
    Each workload block contains:
    SELECT, INSERT, SELECT, UPDATE, and DELETE.

    FRAGMENT_SELECT operations are generated using a separate function and used to derive fragment affinities.
    """

    if not item_ids:
        raise ValueError(f"No item ids were provided.")
    
    block_rng = random.Random(seed)
    affinity_rng = random.Random(seed + 1)

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
        existing_item = block_rng.choice(item_ids)

        new_item = f"{config['generated_id_prefix']}{num:03d}"

        # Assigns the new item to exactly one fragment from each scheme.
        assigned_fragments = [block_rng.choice(fragments_by_scheme[scheme]) for scheme in required_schemes]

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

    # selects the ranom affinity pairs
    affinity_pairs = select_affinity_pairs(fragments_by_scheme, required_schemes, number_affinity_pairs, affinity_rng)
    # generates fragment select operations
    fragment_selects = generate_fragment_select(affinity_pairs, number_fragment_selects, affinity_rng)

    workload.extend(fragment_selects)

    expected_operations = number_blocks_operations * 5 + number_fragment_selects

    if len(workload) != expected_operations:
        raise ValueError(f"There were only generated {len(workload)} operations, however {expected_operations} were expected.")

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
                                 number_fragment_selects=number_fragment_selects,
                                 number_affinity_pairs=number_affinity_pairs,
                                 seed=random_seed)
    
    print("Number of operations:", len(workload))
    print("Saved to:", config["output_path"])

    print("First five operations:")
    for operation in workload[:5]:
        print(operation)

if __name__ == "__main__":
    main()