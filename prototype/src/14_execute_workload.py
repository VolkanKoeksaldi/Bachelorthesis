from pathlib import Path
import json
import time

from database_operations import(
    DATASETS as CONFIGS,
    select_item,
    insert_item,
    update_item,
    delete_item,
    find_item_nodes
)

WORKLOAD_CONFIGS = {
    "mesh": {
        "workload_path": Path("prototype/output/workloads/mesh_workload.json"),
        "result_output": Path("prototype/output/mesh/workload_results"),

        "item_id_column": "descriptor_ui",
        "item_name_column": "descriptor_name",
        "new_item_name": "new_descriptor_name",
        "generated_id_prefix": "D_WORKLOAD_",
        "number_blocks_operations": 20
    },

    "imdb": {
        "workload_path": Path("prototype/output/workloads/imdb_workload.json"),
        "result_output": Path("prototype/output/imdb/workload_results"),

        "item_id_column": "title_id",
        "item_name_column": "primary_title",
        "new_item_name": "new_primary_title",
        "generated_id_prefix": "D_WORKLOAD_",
        "number_blocks_operations": 20
    }
}

DATASET = "imdb"
PLACEMENT = "round_robin"


def load_workload(workload_path):
    """
    Lädt die JSON Datei der erzeugten Workloads.
    """

    if not workload_path.exists():
        raise FileNotFoundError(f"Workload nicht gefunden im Pfad: {workload_path}")

    with workload_path.open("r", encoding="utf-8") as file:
        workload = json.load(file)

    return workload


def execute_operation(operation, placement_type, workload_config, config):
    """
    Führt eine Workload-Operation aus.
    """

    operation_type = operation["operation"]

    item_id_column = workload_config["item_id_column"]
    item_name_column = workload_config["item_name_column"]
    new_item_name = workload_config["new_item_name"]

    item_id = operation[item_id_column]

    start_time = time.perf_counter()

    if operation_type == "SELECT":
        result = select_item(
            item_id=item_id,
            placement_type=placement_type,
            config=config
        )

    elif operation_type == "INSERT":
        result = insert_item(
            item_id=item_id,
            item_name=operation[item_name_column],
            fragment_ids=operation["fragment_ids"],
            placement_type=placement_type,
            config=config
        )

    elif operation_type == "UPDATE":
        result = update_item(
            item_id=item_id,
            update_item_name=operation[new_item_name],
            placement_type=placement_type,
            config=config
        )

    elif operation_type == "DELETE":
        result = delete_item(
            item_id=item_id,
            placement_type=placement_type,
            config=config
        )

    else:
        raise ValueError(f"Unbekannte Operation im Workload: {operation_type}")

    end_time = time.perf_counter()
    runtime = end_time - start_time

    return {"operation": operation_type,
            "item_id": item_id,
            "runtime_seconds": runtime,
            "result": result}


def execute_workload(workload, placement_type, workload_config, config):
    """
    Hier werden alle Operationen aus einem Workload ausgeführt.
    """
    results = []

    executable = [operation for operation in workload if operation["operation"] != "FRAGMENT_SELECT"]

    for operation_id, operation in enumerate(executable, start=1):
        print(f"Operation {operation_id} von {len(executable)}: "
              f"{operation['operation']}")

        op_result = execute_operation(operation=operation, placement_type=placement_type, workload_config=workload_config, config=config)

        op_result["operation_id"] = operation_id

        results.append(op_result)

    return results


def save(results, placement_type, dataset, config):
    """
    Speichert die Ergebnisse einer Placement-Type im output path
    """

    result_output = config["result_output"]
    result_output.mkdir(parents=True, exist_ok=True)

    output_path = (result_output / f"{dataset}_workload_{placement_type}.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"Die Results wurden unter folgendem Pfad gespeichert: {output_path}")


def process_execute_workload(dataset, placement_type, workload_config, database_config):

    workload = load_workload(workload_config["workload_path"])

    results = execute_workload(workload=workload, placement_type=placement_type, workload_config=workload_config, config=database_config)

    save(results=results, placement_type=placement_type, dataset=dataset, config=workload_config)

    return results
        

def main():
    if DATASET not in WORKLOAD_CONFIGS:
        raise ValueError(f"Unbekannter Workload Datensatz mit {DATASET}")

    if DATASET not in CONFIGS:
        raise ValueError(f"Unbekannter Datenbank Datensatz mit {DATASET}")

    workload_config = WORKLOAD_CONFIGS[DATASET]
    database_config = CONFIGS[DATASET]

    process_execute_workload(dataset=DATASET, placement_type=PLACEMENT, workload_config=workload_config, database_config=database_config)

    for number in range(1, workload_config["number_blocks_operations"] + 1):
        item_id = (f"{workload_config['generated_id_prefix']}{number:03d}")

        remaining_nodes = find_item_nodes(item_id=item_id, placement_type=PLACEMENT, config=database_config)

        if remaining_nodes:
            print(f"{item_id} ist noch auf {remaining_nodes} vorhanden")


if __name__ == "__main__":
    main()