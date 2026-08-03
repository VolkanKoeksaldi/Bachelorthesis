from pathlib import Path
import json
import pandas as pd

from database_operations import DATASETS as CONFIGS

EVALUATION_CONFIGS = {
    "mesh": {
        "workload_directory": Path("prototype/output/mesh/workload_results"),
        "output_directory": Path("prototype/output/workload_evaluation")
    },

    "imdb": {
        "workload_directory": Path("prototype/output/imdb/workload_results"),
        "output_directory": Path("prototype/output/workload_evaluation")
    }
}

DATASET = "mesh"


def load_results(placement_type, dataset, config):
    """
    Lädt die Ergebnisse einer Placement-Variante
    """
    workload_directory = config["workload_directory"]

    result_path = workload_directory / f"{dataset}_workload_{placement_type}.json"

    if not result_path.exists():
        raise FileNotFoundError(f"Die erwartete Datei wurde nicht gefunden {result_path}.")

    with result_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    return results

def get_contacted_nodes(result):
    """
    Gibt die kontaktierten Nodes zurück.
    """
    operation_result = result.get("result", {})
    
    contacted_nodes = operation_result.get("contacted_nodes", [])

    return sorted(set(contacted_nodes), key=lambda node_id: int(node_id.rsplit("_", 1)[-1]))
    
def get_contacted_nodes_amount(result):
    """
    Holt die Anzahl der kontaktierten Nodes.
    """

    return len(get_contacted_nodes(result))

def get_available_nodes(result):
    """
    Zum Messen der Replikationsverfügbarkeit von bestimmten Tupeln
    """
    operation_result = result["result"]
    if "available_nodes" in operation_result:
        return len(operation_result["available_nodes"])
    return None

def get_fragment_ids_amount(result):
    """
    Zum Zählen der Fragment_ids auf denen Descriptoren vorhanden sind.
    """
    operation_result = result["result"]
    if "fragment_ids" in operation_result:
        return len(operation_result["fragment_ids"])
    return None

def prepare_result(results, placement_type):
    """
    Wandelt Ergebnisse in tabellarische Zeilen um.
    """
    rows = []

    for result in results:

        contacted_nodes = get_contacted_nodes(result)

        stretch, jump = compute_stretch_jump(contacted_nodes)

        rows.append({
            "placement_type": placement_type,
            "operation": result["operation"],
            "runtime_seconds": result["runtime_seconds"],
            "operation_nodes": len(contacted_nodes),
            "contacted_nodes": ",".join(contacted_nodes),
            "available_nodes": get_available_nodes(result),
            "fragment_count": get_fragment_ids_amount(result),
            "stretch": stretch,
            "jump": jump

            }
        )

    return rows

def all_results(dataset, evaluation_config, database_config):
    """
    Lädt alle Ergebnisse der Placements
    """

    all_rows = []

    for placement_type in database_config["placements"].keys():
        results = load_results(placement_type, dataset, evaluation_config)

        rows = prepare_result(results, placement_type)

        all_rows.extend(rows)

    return pd.DataFrame(all_rows)

def compute_stretch_jump(contacted_nodes):

    node_indices = sorted({int(node_id.rsplit("_", 1)[-1]) for node_id in contacted_nodes})

    if not node_indices:
        return None, None
    elif len(node_indices) == 1:
        return 1.0, 1

    span = (node_indices[-1] - node_indices[0] + 1)

    stretch = span / len(node_indices)

    jump = 1

    for pre, cur in zip(node_indices, node_indices[1:]):
        if cur > pre + 1:
            jump += 1

    return stretch, jump

def compute_summary(results):
    """
    Berechnet die restlichen Metriken.
    """

    summary = (results.groupby(["placement_type", "operation"]).agg(
            operation_count=("operation", "count"),
            total_runtime_seconds=("runtime_seconds", "sum"),
            average_runtime_seconds=("runtime_seconds", "mean"),
            minimum_runtime_seconds=("runtime_seconds", "min"),
            maximum_runtime_seconds=("runtime_seconds", "max"),
            median_runtime_seconds=("runtime_seconds", "median"),
            average_operation_nodes=("operation_nodes", "mean"),
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

def save(results, summary, dataset, config):

    output_directory = config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_directory/f"{dataset}_workload_operations.csv", index=False)

    summary.to_csv(output_directory/f"{dataset}_workload_summary.csv", index=False)

def process_compute_workload_metrics(dataset, evaluation_config, database_config):

    results = all_results(dataset, evaluation_config, database_config)

    summary = compute_summary(results)

    save(results, summary, dataset, evaluation_config)

    return results, summary

def main():
    if DATASET not in EVALUATION_CONFIGS:
        raise ValueError(f"Unbekannter Datensatz in Evaluations Konfiguration: {DATASET}")

    if DATASET not in CONFIGS:
        raise ValueError(f"Unbekannter Datensatz in Datenbank Konfiguration: {DATASET}")

    evaluation_config = EVALUATION_CONFIGS[DATASET]
    database_config = CONFIGS[DATASET]

    results, summary = process_compute_workload_metrics(dataset=DATASET, evaluation_config=evaluation_config, database_config=database_config)

    print("Geladene Operationen:")
    print(results.groupby("placement_type").size())

    print("\nSummary")
    print(summary)

if __name__ == "__main__":
    main()