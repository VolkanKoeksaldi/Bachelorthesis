import json
import pandas as pd
from experiment_config import experiment_path
from database_operations import DATASETS as CONFIGS

DATASET = "imdb" # imdb or mesh


EVALUATION_CONFIGS = {
    "mesh": {
        "workload_directory": experiment_path("mesh/workload_results"),
        "output_directory": experiment_path("workload_evaluation")
    },

    "imdb": {
        "workload_directory": experiment_path("imdb/workload_results"),
        "output_directory": experiment_path("workload_evaluation")
    }
}

def load_results(placement_type, dataset, config):
    """
    Loads the workload results of one placement method.
    """
    workload_directory = config["workload_directory"]

    result_path = workload_directory / f"{dataset}_workload_{placement_type}.json"

    if not result_path.exists():
        raise FileNotFoundError(f"Expected workload result file not found: {result_path}")

    with result_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, list):
        raise ValueError(f"Workload result file must be a JSON list: {result_path}")

    if not results:
        raise ValueError(f"Workload result file contains no operations: {result_path}")
    
    return results


def get_operation_nodes(result, field_name, fallback_field=None):
    """
    Extracts distinct node ids stored in an operation-result field.

    If the requested field is unavailable, an optional fallback field is used.
    Node ids are returned in an ascending numeric order.

    Parameters:
        result: Workload-operation result
        field_name: Is the primary result field that contains the node ids
        fallback_field: optional alternative result field

    Returns:
        A sorted list of distingt node ids
    """
    operation_result = result.get("result", {})

    if not isinstance(operation_result, dict):
        raise ValueError(f"Result of a workload operation must be an dictionary.")

    node_ids = operation_result.get(field_name)

    # Some operations store relevant nodes in another result field.
    if node_ids is None and fallback_field is not None:
        node_ids = operation_result.get(fallback_field, [])

    if node_ids is None:
        return []
    
    # Builds a set of string node_ids in every contacted_nodes.
    # Removes duplicate node_ids.
    node_ids = {str(node_id) for node_id in node_ids}

    return sorted(node_ids, key=lambda node_id: int(node_id.rsplit("_", 1)[-1]))


def get_available_nodes(result):
    """
    Returns the number of nodes on which the item is available.
    """

    operation_result = result["result"]

    if not isinstance(operation_result, dict):
        raise ValueError(f"Result of a workload operation must be an object.")
    
    if "available_nodes" in operation_result:
        return len(set(operation_result["available_nodes"]))
    return None

def get_fragment_ids_amount(result):
    """
    Returns number of distinct fragments containing the item.
    """

    operation_result = result["result"]

    if "fragment_ids" in operation_result:
        return len(set(operation_result["fragment_ids"]))
    return None

def compute_stretch_jump(execution_nodes):
    """
    Computes stretch and jump metrics for execution nodes.
    Node ids are interpreted as positions in a linear node order.

    Parameters:
        execution_nodes: Node ids that are used by one FRAGMENT_SELECT operation

    Returns:
        stretch: Span from smallest to largest accessed node divided by number of 
                 distinct accessed nodes
        jump: Number of contiguous node-index blocks
    """

    node_indices = sorted({int(node_id.rsplit("_", 1)[-1]) for node_id in execution_nodes})

    if not node_indices:
        return None, None
    
    if len(node_indices) == 1:
        return 1.0, 1

    # Calculates span by subtracting biggest node id from first node id + 1
    span = node_indices[-1] - node_indices[0] + 1
    stretch = span / len(node_indices)

    # Counts the number of separate contiguous node index blocks
    jump = 1
    # calculates how often node indices only have 1 as difference between them
    for pre, cur in zip(node_indices, node_indices[1:]):
        if cur > pre + 1:
            jump += 1

    return stretch, jump

def prepare_result(results, placement_type):
    """
    Converts workload results into tabular rows.
    """
    rows = []

    for result in results:

        execution_nodes = get_operation_nodes(result, "execution_nodes", 
                                              fallback_field="contacted_nodes")

        searched_nodes = get_operation_nodes(result, "searched_nodes")

        if result["operation"] == "FRAGMENT_SELECT":

            stretch, jump = compute_stretch_jump(execution_nodes)

        else:
            stretch, jump = None, None

        rows.append({
            "run_label": result["run_label"],
            "repeat_id": result["repeat_id"],
            "placement_type": placement_type,
            "operation_id": result["operation_id"],
            "operation": result["operation"],
            "item_id": result["item_id"],
            "runtime_seconds": result["runtime_seconds"],
            "execution_node_count": len(execution_nodes),
            "execution_nodes": ",".join(execution_nodes),
            "searched_node_count": len(searched_nodes),
            "searched_nodes": ",".join(searched_nodes),
            "available_nodes": get_available_nodes(result),
            "fragment_count": get_fragment_ids_amount(result),
            "stretch": stretch,
            "jump": jump
            }
        )

    return rows

def all_results(dataset, evaluation_config, database_config):
    """
    Loads and combines results of all placement methods.

    Returns:
        A DataFrame that contains all individual workload measurements of all methods.
    """

    all_rows = []

    for placement_type in database_config["placements"]:
        try:
            results = load_results(placement_type, dataset, evaluation_config)
        except FileNotFoundError as err:
            print(f"Skipping {placement_type}: {err}")
            continue

        rows = prepare_result(results, placement_type)

        all_rows.extend(rows)

    if not all_rows:
        raise FileNotFoundError(f"The workload result files were not "
                                "available for the evaluation.")

    return pd.DataFrame(all_rows)

def compute_per_operation(results):
    """
    Aggregates repeated measurements of each individual workload operation.

    Runtime is summarized by using both median and mean across repetitions.
    Execution-nodes are averaged, while Stretch and Jump are represented by their median.

    Returns:
        A DataFrame containing one row per placement and operation id
    """

    return (results.groupby(["placement_type", "operation_id", "operation"], dropna=False)
            .agg(item_id=("item_id", "first"),
                 repeat_count=("repeat_id", "nunique"),
                 median_runtime_seconds=("runtime_seconds", "median"),
                 mean_runtime_seconds=("runtime_seconds", "mean"),
                 runtime_standard_deviation=("runtime_seconds", "std"),
                 execution_node_count=("execution_node_count", "mean"),
                 searched_node_count=("searched_node_count", "mean"),
                 available_nodes=("available_nodes", "mean"),
                 fragment_count=("fragment_count", "mean"),
                 stretch=("stretch", "median"),
                 jump=("jump", "median")).reset_index())

def compute_summary(per_operation_results):
    """
    Aggregates results for every operation by placement method and operation type.
    The average runtime reported in the summary is the mean of the median runtimes
    for every operation.

    Returns:
        A DataFrame that contains the workload-summary metrics
    """

    summary = (per_operation_results.groupby(["placement_type", "operation"]).agg(
            operation_count=("operation_id", "count"),
            repeat_count=("repeat_count", "min"),
            total_median_runtime_seconds=("median_runtime_seconds", "sum"),
            average_median_runtime_seconds=("median_runtime_seconds", "mean"),
            minimum_median_runtime_seconds=("median_runtime_seconds", "min"),
            maximum_median_runtime_seconds=("median_runtime_seconds", "max"),
            median_runtime_seconds=("median_runtime_seconds", "median"),
            average_execution_nodes=("execution_node_count", "mean"),
            average_searched_nodes=("searched_node_count", "mean"),
            average_available_nodes=("available_nodes", "mean"),
            average_amount_of_fragment_ids=("fragment_count", "mean"),
            average_stretch = ("stretch", "mean"),
            median_stretch = ("stretch", "median"),
            min_stretch = ("stretch", "min"),
            max_stretch = ("stretch", "max"),
            average_jump = ("jump", "mean"),
            median_jump = ("jump", "median"),
            min_jump = ("jump", "min"),
            max_jump = ("jump", "max")
        ).reset_index()
    )

    return summary

def save(results, per_operation_results, summary, dataset, config):
    """
    Saves the metrics as CSV files.
    """

    output_directory = config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    operations_output_path = output_directory / f"{dataset}_workload_operations.csv"

    per_operation_output_path = output_directory / f"{dataset}_workload_per_operation.csv"

    summary_output_path = output_directory/f"{dataset}_workload_summary.csv"

    results.to_csv(operations_output_path, index=False)

    per_operation_results.to_csv(per_operation_output_path, index=False)

    summary.to_csv(summary_output_path, index=False)

    print(f"Detailed workload metrics saved to: {operations_output_path}")

    print(f"Repeated operation medians saved to: {per_operation_output_path}")

    print(f"Workload summary saved to: {summary_output_path}")



def process_compute_workload_metrics(dataset, evaluation_config, database_config):
    """
    Loads workload results, computes the evaluation metrics, and saves the CSV files.

    Returns:
        results: Contains the detailed results
        per_operation_results: Results for each operation type
        summary: The summary metrics.
    """

    results = all_results(dataset, evaluation_config, database_config)

    per_operation_results = compute_per_operation(results)

    summary = compute_summary(per_operation_results)

    save(results, per_operation_results, summary, dataset, evaluation_config)

    return results, per_operation_results, summary

def main():
    if DATASET not in EVALUATION_CONFIGS:
        raise ValueError(f"Unknown dataset in evaluation configuration: {DATASET}")

    if DATASET not in CONFIGS:
        raise ValueError(f"Unknown dataset in database configuration: {DATASET}")

    evaluation_config = EVALUATION_CONFIGS[DATASET]
    database_config = CONFIGS[DATASET]

    results, per_operation_results, summary = process_compute_workload_metrics(dataset=DATASET, 
                                                        evaluation_config=evaluation_config, 
                                                        database_config=database_config)

    print("\nLoaded operations per placement method:")
    print(results.groupby("placement_type").size())

    print("\nSummary")
    print(summary)

if __name__ == "__main__":
    main()