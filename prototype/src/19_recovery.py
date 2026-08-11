from pathlib import Path
from experiment_config import experiment_path
import sqlite3
import time
import pandas as pd
from contextlib import closing

from database_operations import DATASETS as DATABASE_CONFIGS

RECOVERY_CONFIGS = {
    "mesh": {
        "failed_node_id": "node_1",

        "baseline": {
            "fragments_path": experiment_path("processed/mesh_fragments.csv"),
            "assignment_paths": {
                        "round_robin": experiment_path("processed/mesh_fragment_assignment_round_robin.csv"),
                        "tuple_ilp": experiment_path("processed/mesh_fragment_assignment_tuple_ilp.csv"),
                        "conflict_locality_ilp": experiment_path("processed/mesh_fragment_assignment_conflict_locality_ilp.csv")
            }
        },

        "updates": {
            "fragments_path": experiment_path("reoptimization/mesh_fragments_updates.csv"),
            "assignment_paths": {
                        "round_robin": experiment_path("processed/mesh_fragment_assignment_round_robin.csv"),
                        "tuple_ilp": experiment_path("reoptimization/mesh_fragment_assignment_tuple_ilp_updated.csv"),
                        "conflict_locality_ilp": experiment_path("reoptimization/mesh_fragment_assignment_conflict_locality_ilp_updated.csv")
            }
        },
        
        "fragment_item_ids": "tuple_ids",
        "recovery_directory": experiment_path("recovery"),
        "recovery_output_path": experiment_path("recovery/mesh_recovery_results.csv")
    },

    "imdb": {
        "failed_node_id": "node_1",

        "baseline": {
            "fragments_path": experiment_path("processed/imdb_fragments.csv"),
            "assignment_paths": {
                        "round_robin": experiment_path("processed/imdb_fragment_assignment_round_robin.csv"),
                        "tuple_ilp": experiment_path("processed/imdb_fragment_assignment_tuple_ilp.csv"),
                        "conflict_locality_ilp": experiment_path("processed/imdb_fragment_assignment_conflict_locality_ilp.csv")
            }
        },

        "updates": {
            "fragments_path": experiment_path("reoptimization/imdb_fragments_updates.csv"),
            "assignment_paths": {
                        "round_robin": experiment_path("processed/imdb_fragment_assignment_round_robin.csv"),
                        "tuple_ilp": experiment_path("reoptimization/imdb_fragment_assignment_tuple_ilp_updated.csv"),
                        "conflict_locality_ilp": experiment_path("reoptimization/imdb_fragment_assignment_conflict_locality_ilp_updated.csv")
            }
        },
        
        "fragment_item_ids": "title_ids",
        "recovery_directory": experiment_path("recovery"),
        "recovery_output_path": experiment_path("recovery/imdb_recovery_results.csv")
    }
}

DATASET = "imdb"
MODE = "updates" # baseline or updates


def find_nodes(placement_type, database_config, mode):
    """
    Finds node databases for one placement type and execution mode.
    """

    if placement_type not in database_config["placements"]:
        raise ValueError(f"Unknown placement type: {placement_type}")

    baseline_directory = (database_config["placements"][placement_type]["node_output"])

    if mode == "baseline":
        placement_directory = baseline_directory
    elif mode == "updates":
        placement_directory = (baseline_directory.parent / "updates" / placement_type)
    else:
        raise ValueError(f"Unknown recovery mode: {mode}")

    node_files = sorted(placement_directory.glob("node_*.db"))

    if not node_files:
        raise FileNotFoundError(f"No SQLite nodes were found in {placement_directory}. First run file 11 in {mode}.")

    return node_files

def find_failed_node(node_files, failed_node_id):
    """
    Finds database file representing failed node.
    """

    for node_file in node_files:
        if node_file.stem == failed_node_id:
            return node_file
        
    raise FileNotFoundError(f"Failed node was not found: {failed_node_id}.")

def load_failed_items(node_file, database_config):
    """
    Loads item ids of all items affected by node failure.
    """

    item_table = database_config["item_table"]
    item_id = database_config["item_id_column"]

    with sqlite3.connect(node_file) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute(f"""
                SELECT {item_id}
                FROM {item_table}
            """)

            rows = cur.fetchall()

    return {row[0] for row in rows}


def find_copies(node_files, failed_node, failed_items, database_config):
    """
    Finds surviving nodes containing copies of items from the failed node.
    """
    item_table = database_config["item_table"]
    item_id_column = database_config["item_id_column"]

    recovery_items = {item_id: [] for item_id in failed_items}

    for node_file in node_files:
        # Excludes failed node from possible recovery nodes
        if node_file == failed_node:
            continue

        with sqlite3.connect(node_file) as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(f"""
                    SELECT {item_id_column}
                    FROM {item_table}
                """)

                available_items = {row[0] for row in cur.fetchall()}

        found_items = (failed_items & available_items)

        for item_id in found_items:
            recovery_items[item_id].append(node_file.stem)

    return recovery_items

def get_recovery_path(placement_type, failed_node_id, recovery_config, mode):
    """
    Creates and returns output path of the recovered node
    """
    recovery_directory = recovery_config["recovery_directory"] / mode / placement_type

    recovery_directory.mkdir(parents=True, exist_ok = True)

    return recovery_directory / f"{failed_node_id}_recovered.db"

def parse_item_ids(value):
    """
    Converts item id string into a list.
    """

    if pd.isna(value) or str(value).strip() == "":
        return []

    return [item_id.strip() for item_id in str(value).split(",") if item_id.strip()]

def load_failed_fragment_info(placement_type, failed_node_id, recovery_config, mode_config):
    """
    Loads fragments and fragment memberships assigned to failed node.
    """

    assignment = pd.read_csv(mode_config["assignment_paths"][placement_type], 
                             dtype={"fragment_id": "string", "node_id": "string"})

    fragments_df = pd.read_csv(mode_config["fragments_path"], dtype={"fragment_id": "string"})

    item_ids_column = recovery_config["fragment_item_ids"]

    required_assignment_columns = {"fragment_id", "node_id"}

    missing_assignment_columns = (required_assignment_columns - set(assignment.columns))

    if missing_assignment_columns:
        raise ValueError(f"Missing columns in assignment file: {sorted(missing_assignment_columns)}")

    required_fragment_columns = {"fragment_id", "scheme", "value", "fragment_size", item_ids_column}

    missing_fragments_columns = (required_fragment_columns - set(fragments_df.columns))

    if missing_fragments_columns:
        raise ValueError(f"Missing columns in fragment file: {sorted(missing_fragments_columns)}")

    # retains fragments assigned to the failed node
    failed_fragment_ids = set(assignment.loc[assignment["node_id"] == failed_node_id, "fragment_id"])

    missing_fragment_ids = (failed_fragment_ids - set(fragments_df["fragment_id"]))

    if missing_fragment_ids:
        raise ValueError(f"Fragments assigned to {failed_node_id} are missing from the fragments path. "
                         f"{sorted(missing_fragment_ids)}")
    
    failed_fragments = fragments_df.loc[fragments_df["fragment_id"].isin(failed_fragment_ids)].copy()

    fragment_rows = list(failed_fragments[["fragment_id", "scheme", "value", 
                                           "fragment_size"]].itertuples(index=False, name=None))

    # reconstructs fragment member relationships of failed node
    fragment_membership_rows = []

    for row in failed_fragments.itertuples(index=False):
        item_ids = parse_item_ids(getattr(row, item_ids_column))

        for item_id in item_ids:
            fragment_membership_rows.append((row.fragment_id, item_id))

    return fragment_rows, fragment_membership_rows


def recover(placement_type, recover_path, node_files, selected_nodes, failed_node_id, recovery_config, database_config, mode_config):
    """
    Reconstructs recoverable information of the failed node from remaining nodes.
    """

    item_table = database_config["item_table"]
    item_id_column = database_config["item_id_column"]
    item_name_column = database_config["item_name_column"]
    membership_item_column = database_config["membership_item_column"]

    additional_item_columns = database_config.get("additional_item_columns", {})

    item_columns = [item_id_column, item_name_column, *additional_item_columns.keys()]

    node_file_id = {node_file.stem: node_file for node_file in node_files}

    # groups recoverable items by their selected source node, thus reducing the number of database
    # connections and queries needed for recovery.
    items_recoverable = {}

    for item_id, source_id in selected_nodes.items():
        items_recoverable.setdefault(source_id, []).append(item_id)

    recovered_rows = []
    fragment_rows, fragment_membership_rows = load_failed_fragment_info(placement_type, 
                                                                        failed_node_id=failed_node_id, 
                                                                        recovery_config=recovery_config, 
                                                                        mode_config=mode_config)

    print(f"Number of fragments to be recovered: {len(fragment_rows)}")
    print(f"Number of fragment memberships to be recovered: {len(fragment_membership_rows)}")

    for node_id, item_ids in items_recoverable.items():
        path = node_file_id[node_id]

        with sqlite3.connect(path) as conn:
            # SQLite limits the number of bound variables in one SQL statement.
            # For large recovery requests these are divided into smaller batch sizes of 10.000 size.
            sql_variable_limit = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
            batch_size = min(sql_variable_limit, 10_000)

            with closing(conn.cursor()) as cur:
                for start in range(0, len(item_ids), batch_size):
                    item_batch = item_ids[start:start + batch_size]
                    placeholders = ", ".join("?" for _ in item_batch)

                    cur.execute(f"""
                        SELECT {", ".join(item_columns)}
                        FROM {item_table}
                        WHERE {item_id_column} IN ({placeholders})
                    """, item_batch)

                    recovered_rows.extend(cur.fetchall())

    recovered_item_ids = {row[0] for row in recovered_rows}

    # recalculates fragment sizes in case some items were unrecoverable
    recovered_fragment_sizes = {}

    # retains memberships whose items were successfully recovered
    fragment_membership_rows = [membership for membership in fragment_membership_rows if membership[1] in recovered_item_ids]

    for fragment_id, item_id in fragment_membership_rows:
        recovered_fragment_sizes[fragment_id] = recovered_fragment_sizes.get(fragment_id, 0) + 1

    fragment_rows = [(fragment_id, scheme, value, recovered_fragment_sizes.get(fragment_id, 0))
                      for fragment_id, scheme, value, fragment_size in fragment_rows]
    
    if recover_path.exists():
        recover_path.unlink()

    item_column_def = [f"{item_id_column} TEXT PRIMARY KEY",
                       f"{item_name_column} TEXT NOT NULL"]

    for column_name, column_type in additional_item_columns.items():
        item_column_def.append(f"{column_name} {column_type}")

    item_placeholders = ", ".join("?" for _ in item_columns)

    with sqlite3.connect(recover_path) as conn:
        # Enables foreign keys to match the node database schema from file 11
        conn.execute("PRAGMA foreign_keys=ON")
        with closing(conn.cursor()) as cur:
            cur = conn.cursor()

            # Creates the recovered item table
            cur.execute(f"""
                CREATE TABLE {item_table} (
                    {", ".join(item_column_def)}
                    )
            """)

            # Inserts recovered item rows into the table
            cur.executemany(f"""
                INSERT INTO {item_table} ({", ".join(item_columns)}) VALUES({item_placeholders})
            """, recovered_rows)

            conn.commit()

            # Creates the fragments table
            cur.execute("""
                CREATE TABLE fragments(
                    fragment_id TEXT PRIMARY KEY,
                    scheme TEXT NOT NULL,
                    value TEXT NOT NULL,
                    fragment_size INTEGER NOT NULL
                    )
            """)

            # Inserts recovered fragment metadata
            cur.executemany("""
                INSERT INTO fragments(fragment_id, scheme, value, fragment_size) VALUES(?, ?, ?, ?)
            """, fragment_rows)

            conn.commit()

            # Creates the fragment-membership table
            cur.execute(f"""
                CREATE TABLE fragment_members(
                    fragment_id TEXT NOT NULL,
                    {membership_item_column} TEXT NOT NULL,
                    PRIMARY KEY (fragment_id, {membership_item_column}))
            """)

            # Inserts recovered fragment-member relationships
            cur.executemany(f"""
                INSERT INTO fragment_members(fragment_id, {membership_item_column}) VALUES(?, ?)
            """, fragment_membership_rows)
            conn.commit()

    return recovered_rows, fragment_rows, fragment_membership_rows


def process_recovery(dataset, recovery_config, database_config, mode_config, mode):
    """
    Simulates and evaluates a node failure for every configured placement type.
    """

    failed_node_id = recovery_config["failed_node_id"]
    recovery_results = []
    for placement_type in mode_config["assignment_paths"]:
        assignment_path = mode_config["assignment_paths"][placement_type]

        if not assignment_path.exists():
            print(
                f"Skipping {placement_type}: assignment file not found: "
                f"{assignment_path}"
            )
            continue

        try:
            node_files = find_nodes(
                placement_type,
                database_config,
                mode,
            )
            failed_node = find_failed_node(
                node_files,
                failed_node_id,
            )
        except FileNotFoundError as error:
            print(f"Skipping {placement_type}: {error}")
            continue

        # Loads all affected items by the node failure
        failed_items = load_failed_items(failed_node, database_config)
        recovery_time_start = time.perf_counter()
        recovery = find_copies(node_files, failed_node, failed_items, database_config)
        # distinguishes between recoverable items from failed node and unrecoverable nodes
        recoverable = {item_id: nodes for item_id, nodes in recovery.items() if nodes}
        unrecoverable = {item_id: nodes for item_id, nodes in recovery.items() if not nodes}
        source_nodes = {node_id for nodes in recoverable.values() for node_id in nodes}
        # Uses first available copy of every item
        selected_sources = {item_id: nodes[0] for item_id, nodes in recoverable.items()}
        recovery_node = set(selected_sources.values())

        recover_path = get_recovery_path(placement_type, failed_node_id, recovery_config, mode)
        (recovered_items, recovered_fragments, recovered_memberships) = recover(placement_type, 
                                                                              recover_path, 
                                                                              node_files, 
                                                                              selected_sources, 
                                                                              failed_node_id, 
                                                                              recovery_config, 
                                                                              database_config,
                                                                              mode_config)
        recovery_time = time.perf_counter() - recovery_time_start

        if failed_items:
            recovery_rate = len(recovered_items) / len(failed_items)
        else:
            recovery_rate = 0.0

        recovery_results.append({
            "placement_type": placement_type,
            "mode": mode,
            "failed_node_id": failed_node_id,
            "amount_affected_items": len(failed_items),
            "amount_recoverable_items": len(recoverable),
            "amount_recovered_items": len(recovered_items),
            "amount_unrecoverable_items": len(unrecoverable),
            "recovery_rate": recovery_rate,
            "available_recovery_nodes": len(source_nodes),
            "selected_recovery_nodes": len(recovery_node),
            "recovery_time_sec": recovery_time,
            "amount_affected_fragments": len(recovered_fragments),
            "amount_recovered_fragment_memberships": len(recovered_memberships)
            })

    if not recovery_results:
        raise FileNotFoundError(
            "No placement had both an assignment and usable node databases."
        )

    recovery_df = pd.DataFrame(recovery_results)
    base_output_path = recovery_config["recovery_output_path"]

    output_path = base_output_path.with_name(f"{base_output_path.stem}_{mode}{base_output_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_df.to_csv(output_path, index=False)
    print(f"Recovery results saved to: {output_path}")

    return recovery_df
        


def main():
    if DATASET not in RECOVERY_CONFIGS:
        raise ValueError(f"Unknown recovery dataset: {DATASET}")

    if DATASET not in DATABASE_CONFIGS:
        raise ValueError(f"Unknown database dataset: {DATASET}")

    recovery_config = RECOVERY_CONFIGS[DATASET]
    database_config = DATABASE_CONFIGS[DATASET]

    if MODE not in {"baseline", "updates"}:
        raise ValueError(f"Unknown recovery mode: {MODE}")

    if MODE not in recovery_config:
        raise ValueError(f"Unknown mode {MODE} for dataset {DATASET}")

    mode_config = recovery_config[MODE]

    recovery_df = process_recovery(dataset=DATASET, recovery_config=recovery_config, 
                                   database_config=database_config, mode_config=mode_config, mode=MODE)

    print(f"\nRecovery Summary {MODE}:")
    print(recovery_df.to_string(index=False))


if __name__ == "__main__":
    main()