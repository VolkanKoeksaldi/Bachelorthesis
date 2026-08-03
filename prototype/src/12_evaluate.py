from pathlib import Path
import sqlite3

import pandas as pd

DATASETS = {
    "mesh": {
        "items_path": Path("prototype/output/processed/mesh_descriptors_sample.csv"),
        
        "overlaps_path": Path("prototype/output/processed/mesh_overlaps_sample.csv"),
        "item_table": "descriptors",
        "item_id_column": "descriptor_ui",
        "replication_factor": 3,

        "placements":{
            "round_robin": {
                    "assignment_path": Path("prototype/output/processed/mesh_fragment_assignment_round_robin.csv"),
                    "node_output": Path("prototype/output/nodes/mesh/round_robin"),
                    "results_output": Path("prototype/output/results/mesh/round_robin")
            },

            "tuple_ilp": {
                "assignment_path": Path("prototype/output/processed/mesh_fragment_assignment_tuple_ilp.csv"),
                "node_output": Path("prototype/output/nodes/mesh/tuple_ilp"),
                "results_output": Path("prototype/output/results/mesh/tuple_ilp")
            },

            "conflict_locality_ilp": {
                "assignment_path": Path("prototype/output/processed/mesh_fragment_assignment_conflict_locality_ilp.csv"),
                "node_output": Path("prototype/output/nodes/mesh/conflict_locality_ilp"),
                "results_output": Path("prototype/output/results/mesh/conflict_locality_ilp")
            }
        }
    },

    "imdb": {
        "items_path": Path("prototype/output/processed/imdb_titles.csv"),
        "overlaps_path": Path("prototype/output/processed/imdb_overlaps.csv"),
        "item_table": "title",
        "item_id_column": "title_id",
        "item_name_column": "primary_title",
        "fragment_item_ids_column": "title_ids",
        "membership_item_column": "title_id",

        "additional_item_columns": {
            "metadata_json": "TEXT",
            "item_size_bytes": "INTEGER"
        },

        "replication_factor": 3,

        "placements": {
            "round_robin": {
                "assignment_path": Path("prototype/output/processed/imdb_fragment_assignment_round_robin.csv"),
                "node_output": Path("prototype/output/nodes/imdb/round_robin"),
                "results_output": Path("prototype/output/results/imdb/round_robin")
            },

            "tuple_ilp": {
                "assignment_path": Path("prototype/output/processed/imdb_fragment_assignment_tuple_ilp.csv"),
                "node_output": Path("prototype/output/nodes/imdb/tuple_ilp"),
                "results_output": Path("prototype/output/results/imdb/tuple_ilp")
            },

            "conflict_locality_ilp": {
                "assignment_path": Path("prototype/output/processed/imdb_fragment_assignment_conflict_locality_ilp.csv"),
                "node_output": Path("prototype/output/nodes/imdb/conflict_locality_ilp"),
                "results_output": Path("prototype/output/results/imdb/conflict_locality_ilp")

            },
        }
    }
}

DATASET = "imdb"
PLACEMENT = "round_robin"

def count_table_rows(db_path, table_name):
    """
    Zählt Anzahl der Zeilen in einer Tabelle einer SQLite-Datei
    """

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]

    connection.close()

    return count

def collect_node_metrics(node_output, item_table):
    """
    Liest alle SQLite-Node-Dateien aus und berechnet pro Node:
    - Anzahl gespeicherter Fragmente
    - Anzahl eindeutiger Descriptors
    - Anzahl Fragment-Membership-Einträge
    
    Memberships zeigen, wie viele Descriptor-zu-Fragment-Zuordnungen tatsächlich gespeichert wurden.
    """

    node_rows = []

    for db_path in sorted(node_output.glob("node_*.db")):
        node_id = db_path.stem

        fragment_count = count_table_rows(db_path, "fragments")
        item_count = count_table_rows(db_path, item_table)
        membership_count = count_table_rows(db_path, "fragment_members")
        
        node_rows.append({
            "node_id": node_id,
            "fragments": fragment_count,
            "unique_items": item_count,
            "fragment_memberships": membership_count,
        })

    return pd.DataFrame(node_rows)

def compute_overlap_assignment_metrics(overlaps_df, assignment_df):
    """
    Prüft, auf welchen Nodes überlappende Fragmente gespeichert wurden.

    Dafür wird für jedes Overlap-Paar nachgeschaut:
    - auf welcher node liegt fragment_1?
    - Auf welcher Node liegt fragment_2?
    - Liegen beide auf derselben Node oder auf unterschiedlichen Nodes?
    """

    # Lookup: fragment_id -> node_id
    fragment_to_node = assignment_df.set_index("fragment_id")["node_id"].to_dict()

    overlap_rows = []

    for _, row in overlaps_df.iterrows():
        fragment_1 = row["fragment_1"]
        fragment_2 = row["fragment_2"]

        node_1 = fragment_to_node.get(fragment_1)
        node_2 = fragment_to_node.get(fragment_2)

        if node_1 is None or node_2 is None:
            raise ValueError(f"Es gibt keine Zuweisung für das Paar ({fragment_1}, {fragment_2})")

        same_node = node_1 == node_2

        overlap_rows.append({
            "fragment_1": fragment_1,
            "fragment_2": fragment_2,
            "overlap_size": int(row["overlap_size"]),
            "node_1": node_1,
            "node_2": node_2,
            "same_node": same_node,
        })

    return pd.DataFrame(overlap_rows)

def replication_metrics(items_df, node_output, item_table, item_id_column):
    """
    Funktion berechnet für jeden Descriptor auf wievielen verschiedenen Nodes er verfügbar ist.
    Dadurch wird geprüft, ob der Replikationsfaktor tatsächlich umgesetzt wurde.
    """

    # Alle ursprünglichen Descriptoren werden mit leerem Set angelegt
    item_to_nodes = {
        item_id: set()
        for item_id
        in items_df[item_id_column].dropna().unique()
    }

    for db in sorted(node_output.glob("node_*.db")):
        node_id = db.stem
        connection = sqlite3.connect(db)

        node_items_df = pd.read_sql_query(f"SELECT {item_id_column} FROM {item_table}",
                                                connection)
        
        connection.close()

        for item_id in node_items_df[item_id_column]:
            item_to_nodes.setdefault(item_id, set()).add(node_id)

    replication_rows = []

    for item_id, node_ids in item_to_nodes.items():
        replication_rows.append({
            item_id_column: item_id,
            "replication_count": len(node_ids),
            "nodes": ",".join(sorted(node_ids))
        })
    
    return pd.DataFrame(replication_rows)

def compute_summary_metrics(node_metrics_df, overlap_assignment_df, items_df, item_id_column, replication_metrics_df, replication_factor):
    """
    Berechnet globale Zusammenfassungsmetriken für die Round-Robin-Baseline.
    """

    global_unique_items = items_df[item_id_column].nunique()

    total_node_item_copies = node_metrics_df["unique_items"].sum()
    total_fragment_memberships = node_metrics_df["fragment_memberships"].sum()

    used_nodes = len(node_metrics_df)

    same_node_overlaps = overlap_assignment_df["same_node"].sum()
    total_overlaps = len(overlap_assignment_df)

    different_node_overlaps = total_overlaps - same_node_overlaps

    weighted_same_node_overlap = overlap_assignment_df.loc[
        overlap_assignment_df["same_node"],
        "overlap_size"
    ].sum()

    weighted_total_overlap = overlap_assignment_df["overlap_size"].sum()

    if weighted_total_overlap <= 0:
        weighted_same_node_overlap_ratio = 0
    else:
        weighted_same_node_overlap_ratio = weighted_same_node_overlap / weighted_total_overlap

    items_below_replication = (replication_metrics_df["replication_count"] < replication_factor).sum()
    
    items_equal_replication = (replication_metrics_df["replication_count"] == replication_factor).sum()
    
    items_above_replication = (replication_metrics_df["replication_count"] > replication_factor).sum()

    minimum_item_replication = replication_metrics_df["replication_count"].min()

    maximum_item_replication = replication_metrics_df["replication_count"].max()

    average_item_replication = replication_metrics_df["replication_count"].mean()
    
    maximum_replication_deficit = max(0, replication_factor - replication_metrics_df["replication_count"].min())
    
    total_replication_deficit = (replication_factor - replication_metrics_df["replication_count"]).clip(lower=0).sum()
    
    all_items_meet_replication = items_below_replication == 0

    if global_unique_items <= 0:
        item_redundancy_factor = 0
    else:
        item_redundancy_factor = total_node_item_copies / global_unique_items

    if total_overlaps <= 0:
        same_node_overlap_ratio = 0
    else:
        same_node_overlap_ratio = same_node_overlaps / total_overlaps

    summary = {
        "used_nodes": used_nodes,
        "global_unique_items": global_unique_items,
        "total_node_item_copies": total_node_item_copies,
        "item_redundancy_factor": item_redundancy_factor,
        "total_fragment_memberships": total_fragment_memberships,
        "min_items_per_node": node_metrics_df["unique_items"].min(),
        "max_items_per_node": node_metrics_df["unique_items"].max(),
        "avg_items_per_node": node_metrics_df["unique_items"].mean(),
        "item_load_imbalance": node_metrics_df["unique_items"].max() - node_metrics_df["unique_items"].min(),
        "total_overlap_pairs": total_overlaps,
        "same_node_overlap_pairs": same_node_overlaps,
        "different_node_overlap_pairs": different_node_overlaps,
        "same_node_overlap_ratio": same_node_overlap_ratio,
        "weighted_total_overlap": weighted_total_overlap,
        "weighted_same_node_overlap": weighted_same_node_overlap,
        "weighted_same_node_overlap_ratio": weighted_same_node_overlap_ratio,
        "minimum_fragment_memberships_per_node": node_metrics_df["fragment_memberships"].min(),
        "maximum_fragment_memberships_per_node": node_metrics_df["fragment_memberships"].max(),
        "average_fragment_memberships": node_metrics_df["fragment_memberships"].mean(),
        "fragment_membership_load_difference": node_metrics_df["fragment_memberships"].max() - node_metrics_df["fragment_memberships"].min(),
        "items_count_below_replication": items_below_replication,
        "items_count_equal_replication": items_equal_replication,
        "items_count_above_replication": items_above_replication,
        "minimum_item_replication": minimum_item_replication,
        "maximum_item_replication": maximum_item_replication,
        "average_item_replication": average_item_replication,
        "maximum_replication_deficit": maximum_replication_deficit,
        "total_replication_deficit": total_replication_deficit,
        "all_items_meet_replication": all_items_meet_replication
    }

    return pd.DataFrame([
        {"metric": metric, "value": value}
        for metric, value in summary.items()
    ])



def process_evaluation(config, placement):
    results_output = placement["results_output"]
    results_output.mkdir(parents=True, exist_ok=True)

    items_df = pd.read_csv(config["items_path"])
    assignment_df = pd.read_csv(placement["assignment_path"])
    overlaps_df = pd.read_csv(config["overlaps_path"])

    node_metrics_df = collect_node_metrics(placement["node_output"], config["item_table"])
    overlap_assignment_df = compute_overlap_assignment_metrics(overlaps_df, assignment_df)
    replication_metrics_df = replication_metrics(items_df, placement["node_output"], config["item_table"], config["item_id_column"])
    summary_metrics_df = compute_summary_metrics(node_metrics_df, overlap_assignment_df, 
                                                 items_df, config["item_id_column"], replication_metrics_df, config["replication_factor"])
    
    node_metrics_df.to_csv(results_output / "node_metrics.csv", index=False)
    overlap_assignment_df.to_csv(results_output / "overlap_assignment_metrics.csv", index=False)
    replication_metrics_df.to_csv(results_output / "item_replication_metrics.csv", index=False)
    summary_metrics_df.to_csv(results_output / "summary_metrics.csv", index=False)

    print("Node metrics:")
    print(node_metrics_df)
    print()

    print("Summary metrics:")
    print(summary_metrics_df)
    print()

    print(f"Saved results to: {results_output}")

    return node_metrics_df, overlap_assignment_df, summary_metrics_df, replication_metrics_df



def main():
    if DATASET not in DATASETS:
        raise ValueError(f"Unbekannter Datensatz wurde ausgewählt: {DATASET}")

    config = DATASETS[DATASET]

    if PLACEMENT not in config["placements"]:
        raise ValueError(f"Unbekanntes Placement wurde ausgewählt: {PLACEMENT}")

    placement = config["placements"][PLACEMENT]

    process_evaluation(config, placement)


if __name__ == "__main__":
    main()