from pathlib import Path
import pandas as pd
import json
import random


DATASETS = {
    "mesh": {
        "items_path": Path("prototype/output/processed/mesh_descriptors_sample.csv"),
        "fragments_path": Path("prototype/output/processed/mesh_fragments_sample.csv"),
        "output_path": Path("prototype/output/workloads/mesh_workload.json"),
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
        "items_path": Path("prototype/output/processed/imdb_titles.csv"),
        "fragments_path": Path("prototype/output/processed/imdb_fragments.csv"),
        "output_path": Path("prototype/output/workloads/imdb_workload.json"),
        "item_id_column": "title_id",
        "item_name_column": "primary_title",
        "new_item_name": "new_primary_title",
        "generated_id_prefix": "D_WORKLOAD_",
        "generated_name_prefix": "Workload Generated Descriptor",
        "updated_name_prefix": "Updated Workload Generated Descriptor",
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
    items_df = pd.read_csv(config["items_path"], usecols=[config["item_id_column"]])
    fragments_df = pd.read_csv(config["fragments_path"], usecols=["fragment_id", "scheme"])

    item_ids = (items_df[config["item_id_column"]].dropna().astype(str).unique().tolist())

    fragments_by_scheme = {scheme: group["fragment_id"].astype(str).tolist()
                           for scheme, group in fragments_df.groupby("scheme")}

    return item_ids, fragments_by_scheme

# generieren der verschiedenen Workloads

def generate_workload(item_ids, fragments_by_scheme, config, number_blocks_operations=number_blocks_operations, seed=random_seed):
    """
    Hiermit werden reproduzierbare gemischte Workloads erzeugt.
    descriptor_ids:
        Vorhandene descriptor-IDs
    
    fragment_ids:
        Vorhandene fragment-ids

    Diese Operationen werden in Blöcke erstellt. Jeder Block enthält:
    SELECT, INSERT, SELECT, UPDATE, DELETE
    """

    if not item_ids:
        raise ValueError(f"Es wurden keine Item-IDs eingegeben.")
    
    rng = random.Random(seed)

    workload = []

    required_schemes = config["fragmentation_schemes"]

    for num in range(1, number_blocks_operations + 1):
        # Chooses a random element
        existing_item = rng.choice(item_ids)

        new_item = f"{config['generated_id_prefix']}{num:03d}"

        assigned_fragments = [rng.choice(fragments_by_scheme[scheme]) for scheme in required_schemes]

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

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(workload, file, indent=4)


    return workload

def main():
    if DATASET not in DATASETS:
        raise ValueError(f"Unbekannter Datensatz {DATASET}")

    config = DATASETS[DATASET]

    item_ids, fragments_by_scheme = (load_workload_inputs(config))

    workload = generate_workload(item_ids=item_ids, fragments_by_scheme=fragments_by_scheme,
                                 config=config, number_blocks_operations=number_blocks_operations,
                                 seed=random_seed)
    
    print("Anzahl Operationen:", len(workload))
    print("Die Workloads werden gespeichert unter:", config["output_path"])

    for operation in workload[:5]:
        print(operation)

if __name__ == "__main__":
    main()