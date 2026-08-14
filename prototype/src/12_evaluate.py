from pathlib import Path
import sqlite3
from contextlib import closing
from experiment_config import CAPACITY_BUFFER, NUM_NODES, REPLICATION_FACTOR, experiment_path
import pandas as pd
from placement_capacity import calculate_node_capacity

DATASETS = {
    "mesh": {
        "items_path": experiment_path("processed/mesh_terms.csv"),
        
        "overlaps_path": experiment_path("processed/mesh_overlaps.csv"),
        "item_table": "ill",
        "item_id_column": "tuple_id",
        "fragment_item_ids_column": "tuple_ids",
        "capacity_reference_path": experiment_path("processed/mesh_fragments.csv"),
        "num_nodes": NUM_NODES,
        "capacity_buffer": CAPACITY_BUFFER,
        "replication_factor": REPLICATION_FACTOR,

        "placements":{
            "round_robin": {
                    "assignment_path": experiment_path("processed/mesh_fragment_assignment_round_robin.csv"),
                    "node_output": experiment_path("nodes/mesh/round_robin"),
                    "results_output": experiment_path("results/mesh/round_robin")
            },

            "tuple_ilp": {
                "assignment_path": experiment_path("processed/mesh_fragment_assignment_tuple_ilp.csv"),
                "node_output": experiment_path("nodes/mesh/tuple_ilp"),
                "results_output": experiment_path("results/mesh/tuple_ilp")
            },

            "conflict_locality_ilp": {
                "assignment_path": experiment_path("processed/mesh_fragment_assignment_conflict_locality_ilp.csv"),
                "node_output": experiment_path("nodes/mesh/conflict_locality_ilp"),
                "results_output": experiment_path("results/mesh/conflict_locality_ilp")
            }
        }
    },

    "imdb": {
        "items_path": experiment_path("processed/imdb_titles.csv"),
        "overlaps_path": experiment_path("processed/imdb_overlaps.csv"),
        "item_table": "title",
        "item_id_column": "title_id",
        "item_name_column": "primary_title",
        "fragment_item_ids_column": "title_ids",
        "capacity_reference_path": experiment_path("processed/imdb_fragments.csv"),
        "num_nodes": NUM_NODES,
        "capacity_buffer": CAPACITY_BUFFER,
        "membership_item_column": "title_id",

        "additional_item_columns": {
            "metadata_json": "TEXT",
            "item_size_bytes": "INTEGER"
        },

        "replication_factor": REPLICATION_FACTOR,

        "placements": {
            "round_robin": {
                "assignment_path": experiment_path("processed/imdb_fragment_assignment_round_robin.csv"),
                "node_output": experiment_path("nodes/imdb/round_robin"),
                "results_output": experiment_path("results/imdb/round_robin")
            },

            "tuple_ilp": {
                "assignment_path": experiment_path("processed/imdb_fragment_assignment_tuple_ilp.csv"),
                "node_output": experiment_path("nodes/imdb/tuple_ilp"),
                "results_output": experiment_path("results/imdb/tuple_ilp")
            },

            "conflict_locality_ilp": {
                "assignment_path": experiment_path("processed/imdb_fragment_assignment_conflict_locality_ilp.csv"),
                "node_output": experiment_path("nodes/imdb/conflict_locality_ilp"),
                "results_output": experiment_path("results/imdb/conflict_locality_ilp")

            },
        }
    }
}

DATASET = "mesh"
PLACEMENT = "conflict_locality_ilp"

def count_table_rows(db_path, table_name):
    """
    Returns number of rows in a table of a SQLite node database.
    """

    with sqlite3.connect(db_path) as conn:
        with closing(conn.cursor()) as cur:
        
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
    return count

def collect_node_metrics(node_output, item_table):
    """
    Collects storage metrics from every database in a node directory.
    For each node, result contains:
    - Number of assigned fragments
    - Number of unique items
    - Number of fragment-to-item memberships
    """

    database_paths = sorted(node_output.glob("node_*.db"))

    if not database_paths:
        raise FileNotFoundError(f"No node databases were found in: {node_output}")
    
    node_rows = []

    for db_path in database_paths:
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
    Determines whether each pair of overlapping fragments share a node.
    Returned rows retain overlap size, that way both unweighted and weighted
    placement metrics can be calculated.
    """

    if not assignment_df["fragment_id"].is_unique:
        raise ValueError(f"Assignment contains more than one node assignment for the same fragment_id.")

    # Maps every fragment to its assigned node for efficient pair lookups
    fragment_to_node = assignment_df.set_index("fragment_id")["node_id"].to_dict()

    overlap_rows = []

    for _, row in overlaps_df.iterrows():
        fragment_1 = row["fragment_1"]
        fragment_2 = row["fragment_2"]

        node_1 = fragment_to_node.get(fragment_1)
        node_2 = fragment_to_node.get(fragment_2)

        if node_1 is None or node_2 is None:
            raise ValueError(f"No complete assignment exists for overlap pair ({fragment_1}, {fragment_2})")

        same_node = node_1 == node_2

        overlap_rows.append({
            "fragment_1": fragment_1,
            "fragment_2": fragment_2,
            "overlap_size": int(row["overlap_size"]),
            "node_1": node_1,
            "node_2": node_2,
            "same_node": same_node
        })

    # If overlap table is empty, then columns are generated with empty values
    return pd.DataFrame(overlap_rows,
                        columns=["fragment_1",
                                 "fragment_2",
                                 "overlap_size",
                                 "node_1",
                                 "node_2",
                                 "same_node"])

def replication_metrics(items_df, node_output, item_table, item_id_column):
    """
    Calculates number of distinct nodes storing each original item.
    """

    # Original items are initialized as sets
    item_to_nodes = {item_id: set() for item_id in items_df[item_id_column].dropna().unique()}

    for db in sorted(node_output.glob("node_*.db")):
        # extracts file name
        node_id = db.stem

        # SQL query is executed. Result is a node_items DataFrame with the item_id_column from item_table
        with sqlite3.connect(db) as conn:
            node_items_df = pd.read_sql_query(f"SELECT {item_id_column} FROM {item_table}", conn)

        conn.close()
        # Creates for every item a set
        for item_id in node_items_df[item_id_column]:
            if item_id not in item_to_nodes:
                raise ValueError(f"Node {node_id} contains unknown item_id: {item_id}")
            
            item_to_nodes[item_id].add(node_id)

    replication_rows = []

    for item_id, node_ids in item_to_nodes.items():
        replication_rows.append({item_id_column: item_id, "replication_count": len(node_ids), "nodes": ",".join(sorted(node_ids))})
    
    return pd.DataFrame(replication_rows)

def compute_summary_metrics(node_metrics_df, overlap_assignment_df, items_df, item_id_column, replication_metrics_df,
                            replication_factor, node_capacity):
    """
    Aggregates node, overlaps, storage, and replication metrics for a placement.
    """

    # .nunique() counts unique values
    global_unique_items = items_df[item_id_column].nunique()

    total_node_item_copies = node_metrics_df["unique_items"].sum()
    total_fragment_memberships = node_metrics_df["fragment_memberships"].sum()

    used_nodes = len(node_metrics_df)

    same_node_overlaps = overlap_assignment_df["same_node"].sum()
    total_overlaps = len(overlap_assignment_df)

    different_node_overlaps = total_overlaps - same_node_overlaps

    # Sums the overlap size values over filtered overlaps with loc, where same_node = True
    # and only extracts overlap_size.
    weighted_same_node_overlap = overlap_assignment_df.loc[overlap_assignment_df["same_node"], "overlap_size"].sum()

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

    # clip limits that values below 0 are fixed as value 0.
    total_replication_deficit = (replication_factor - replication_metrics_df["replication_count"]).clip(lower=0).sum()
    
    all_items_meet_replication = items_below_replication == 0

    capacity_excess = (node_metrics_df["unique_items"] - node_capacity).clip(lower=0)

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
        "item_load_imbalance": node_metrics_df["unique_items"].max() - 
                               node_metrics_df["unique_items"].min(),
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
        "fragment_membership_load_difference": node_metrics_df["fragment_memberships"].max() - 
                                               node_metrics_df["fragment_memberships"].min(),
        "items_count_below_replication": items_below_replication,
        "items_count_equal_replication": items_equal_replication,
        "items_count_above_replication": items_above_replication,
        "minimum_item_replication": minimum_item_replication,
        "maximum_item_replication": maximum_item_replication,
        "average_item_replication": average_item_replication,
        "maximum_replication_deficit": maximum_replication_deficit,
        "total_replication_deficit": total_replication_deficit,
        "all_items_meet_replication": all_items_meet_replication,
        "node_capacity": node_capacity,
        "nodes_exceeding_capacity": int((capacity_excess > 0).sum()),
        "maximum_capacity_excess": int(capacity_excess.max()),
        "total_capacity_excess": int(capacity_excess.sum()),
        "all_nodes_within_capacity": bool((capacity_excess == 0).all()),
    }

    return pd.DataFrame([
        {"metric": metric, "value": value}
        for metric, value in summary.items()
    ])


def process_evaluation(config, placement):
    """
    Loads placement, computes metrics and then writes a CSV with the results.
    """

    results_output = placement["results_output"]
    results_output.mkdir(parents=True, exist_ok=True)

    items_df = pd.read_csv(config["items_path"])
    assignment_df = pd.read_csv(placement["assignment_path"])
    overlaps_df = pd.read_csv(config["overlaps_path"])

    node_metrics_df = collect_node_metrics(placement["node_output"], config["item_table"])
    overlap_assignment_df = compute_overlap_assignment_metrics(overlaps_df, assignment_df)
    replication_metrics_df = replication_metrics(items_df, placement["node_output"], 
                                                 config["item_table"], config["item_id_column"])

    node_capacity = calculate_node_capacity(
        reference_fragments_path=config["capacity_reference_path"],
        item_ids_column=config["fragment_item_ids_column"],
        num_nodes=config["num_nodes"],
        capacity_buffer=config["capacity_buffer"],
    )

    summary_metrics_df = compute_summary_metrics(node_metrics_df, overlap_assignment_df, 
                                                 items_df, config["item_id_column"], replication_metrics_df,
                                                 config["replication_factor"], node_capacity)
    
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
        raise ValueError(f"Unknown dataset: {DATASET}")

    config = DATASETS[DATASET]

    if PLACEMENT not in config["placements"]:
        raise ValueError(f"Unknown placement method: {PLACEMENT}")

    placement = config["placements"][PLACEMENT]

    process_evaluation(config, placement)


if __name__ == "__main__":
    main()