from pathlib import Path
import json
import time
import shutil
import tempfile
from copy import deepcopy

from experiment_config import RUN_LABEL, experiment_path

from database_operations import(
    DATASETS as CONFIGS,
    select_item,
    select_fragments,
    insert_item,
    update_item,
    delete_item,
    find_item_nodes
)

WORKLOAD_CONFIGS = {
    "mesh": {
        "workload_path": experiment_path("workloads/mesh_workload.json"),
        "result_output": experiment_path("mesh/workload_results"),

        "item_id_column": "descriptor_ui",
        "item_name_column": "descriptor_name",
        "new_item_name": "new_descriptor_name",
    },

    "imdb": {
        "workload_path": experiment_path("workloads/imdb_workload.json"),
        "result_output": experiment_path("imdb/workload_results"),

        "item_id_column": "title_id",
        "item_name_column": "primary_title",
        "new_item_name": "new_primary_title",
    }
}

DATASET = "imdb"
PLACEMENT = "round_robin"
REPETITIONS = 5
WARM_UP_RUNS = 1


def load_workload(workload_path):
    """
    Loads generated workload from a JSON file.
    """

    if not workload_path.exists():
        raise FileNotFoundError(f"Workload file not found: {workload_path}")

    with workload_path.open("r", encoding="utf-8") as file:
        workload = json.load(file)

    if not isinstance(workload, list):
        raise ValueError("Workload file must be a JSON list.")

    return workload


def execute_operation(operation, placement_type, workload_config, config):
    """
    Executes and then measures a single database operation.
    """

    operation_type = operation["operation"]

    item_id_column = workload_config["item_id_column"]
    item_name_column = workload_config["item_name_column"]
    new_item_name = workload_config["new_item_name"]

    item_id = operation.get(item_id_column)

    start_time = time.perf_counter()

    if operation_type == "SELECT":
        result = select_item(
            item_id=item_id,
            placement_type=placement_type,
            config=config
        )

    elif operation_type == "FRAGMENT_SELECT":
        result = select_fragments(
            fragment_ids=operation["fragment_ids"],
            placement_type=placement_type,
            config=config,
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
        raise ValueError(f"Unknown workload operation: {operation_type}")

    end_time = time.perf_counter()
    runtime = end_time - start_time

    return {"operation": operation_type,
            "item_id": item_id,
            "fragment_ids": operation.get("fragment_ids"),
            "runtime_seconds": runtime,
            "result": result}


def execute_workload(workload, placement_type, workload_config, config):
    """
    Executes database operations in the workload.
    """
    results = []

    # executes the operations one after another starting with index 1.
    for operation_id, operation in enumerate(workload, start=1):
        print(f"Operation {operation_id} of {len(workload)}: "
              f"{operation['operation']}")

        op_result = execute_operation(operation=operation, placement_type=placement_type, 
                                      workload_config=workload_config, config=config)

        op_result["operation_id"] = operation_id

        results.append(op_result)

    return results


def save(results, placement_type, dataset, config):
    """
    Saves workload execution results as a JSON.
    """

    result_output = config["result_output"]
    result_output.mkdir(parents=True, exist_ok=True)

    output_path = (result_output / f"{dataset}_workload_{placement_type}.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"Results saved to: {output_path}")

def copy_node_databases(node_directory, target_directory):
    """
    Copies initial nodes for one independent benchmark repitition.
    """

    node_files = sorted(node_directory.glob("node_*.db"))

    if not node_files:
        raise FileNotFoundError(f"No node databases were found in {node_directory}. "
                                f"First file 11 needs to be run before executing the workload.")

    target_directory.mkdir(parents=True, exist_ok=True)

    for node_file in node_files:
        # shutil used for file and folder operations.
        # copy2 copies a file and retains original information of the node file into target directory
        shutil.copy2(node_file, target_directory / node_file.name)

def create_repeated_config(database_config, placement_type, node_directory):
    """
    Returns the database configurations pointing to a copy of the benchmark nodes.
    """

    repeat_config = deepcopy(database_config)
    repeat_config["placements"][placement_type]["node_output"] = node_directory

    return repeat_config

def verify_deleted_items(workload, placement_type, workload_config, database_config):
    """
    Verifies every item deleted with a DELETE operation from all node databases.
    """
    item_id_column = workload_config["item_id_column"]

    deleted_item_ids = [
        operation[item_id_column]
        for operation in workload
        if operation["operation"] == "DELETE"
        ]

    remaining_items = {}

    for item_id in deleted_item_ids:
        remaining_nodes = find_item_nodes(item_id, placement_type, database_config)

        if remaining_nodes:
            remaining_items[item_id] = remaining_nodes

    if remaining_items:
        details = "; ".join(f"{item_id}: {nodes}"
                            for item_id, nodes in remaining_items.items())

        raise RuntimeError("The following items still exist in node databases, even though they "
                           f"should be deleted: {details}")

    print("All items that were supposed to be deleted using DELETE operations, were removed from "
          "every node.")


def process_execute_workload(dataset, placement_type, workload_config, database_config, 
                             repetitions=REPETITIONS, warm_up_runs=WARM_UP_RUNS):
    """
    Loads, executes, verifies and saves the workload.
    """

    workload = load_workload(workload_config["workload_path"])

    if repetitions < 1:
        raise ValueError("REPETITIONS must be at least 1.")

    if warm_up_runs < 0:
        raise ValueError("WARM_UP_RUNS cannot be negative.")

    source_node_directory = database_config["placements"][placement_type]["node_output"]

    results = []

    total_runs = warm_up_runs + repetitions

    for run_number in range(1, total_runs + 1):
        is_warm_up = run_number <= warm_up_runs
        measured_repeat = run_number - warm_up_runs
        run_kind = "warm-up" if is_warm_up else f"repeat {measured_repeat}"

        print(f"\nStarting {run_kind} of {placement_type}...")

        with tempfile.TemporaryDirectory(
            prefix=f"{dataset}_{placement_type}_") as temp_directory:
            repeat_node_directory = Path(temp_directory) / "nodes"
            copy_node_databases(source_node_directory, repeat_node_directory,)

            repeat_database_config = create_repeated_config(database_config, 
                                                            placement_type, repeat_node_directory)

            repeat_results = execute_workload(workload, placement_type, 
                                              workload_config, repeat_database_config)  

            verify_deleted_items(workload, placement_type, workload_config, repeat_database_config)

        if is_warm_up:
            continue

        for result in repeat_results:
            result["run_label"] = RUN_LABEL
            result["repeat_id"] = measured_repeat

        results.extend(repeat_results)   

    save(results=results, placement_type=placement_type, dataset=dataset, config=workload_config)

    return results
        

def main():
    if DATASET not in WORKLOAD_CONFIGS:
        raise ValueError(f"Unknown workload dataset: {DATASET}")

    if DATASET not in CONFIGS:
        raise ValueError(f"Unknown database dataset: {DATASET}")

    workload_config = WORKLOAD_CONFIGS[DATASET]
    database_config = CONFIGS[DATASET]

    results = process_execute_workload(dataset=DATASET, placement_type=PLACEMENT, 
                             workload_config=workload_config, database_config=database_config,
                             repetitions=REPETITIONS, warm_up_runs=WARM_UP_RUNS)

    print(f"Executed database operations: {len(results)}")


if __name__ == "__main__":
    main()