from pathlib import Path
import sqlite3
import time
import pandas as pd

from database_operations import DATASETS as DATABASE_CONFIGS

RECOVERY_CONFIGS = {
    "mesh": {
        "failed_node_id": "node_1",

        "baseline": {
            "fragments_path": Path("prototype/output/processed/mesh_fragments_sample.csv"),
            "assignment_paths": {
                        "round_robin": Path("prototype/output/processed/mesh_fragment_assignment_round_robin.csv"),
                        "tuple_ilp": Path("prototype/output/processed/mesh_fragment_assignment_tuple_ilp.csv"),
                        "conflict_locality_ilp": Path("prototype/output/processed/mesh_fragment_assignment_conflict_locality_ilp.csv")
            }
        },

        "updates": {
            "fragments_path": Path("prototype/output/reoptimization/mesh_fragments_sample_updates.csv"),
            "assignment_paths": {
                        "round_robin": Path("prototype/output/processed/mesh_fragment_assignment_round_robin.csv"),
                        "tuple_ilp": Path("prototype/output/reoptimization/mesh_fragment_assignment_tuple_ilp_updated.csv"),
                        "conflict_locality_ilp": Path("prototype/output/reoptimization/mesh_fragment_assignment_conflict_locality_ilp_updated.csv")
            }
        },
        
        "fragment_item_ids": "descriptor_ids",
        "recovery_directory": Path("prototype/output/recovery"),
        "recovery_output_path": Path("prototype/output/recovery/mesh_recovery_results.csv")
    },

    "imdb": {
        "failed_node_id": "node_1",

        "baseline": {
            "fragments_path": Path("prototype/output/processed/imdb_fragments.csv"),
            "assignment_paths": {
                        "round_robin": Path("prototype/output/processed/imdb_fragment_assignment_round_robin.csv"),
                        "tuple_ilp": Path("prototype/output/processed/imdb_fragment_assignment_tuple_ilp.csv"),
                        "conflict_locality_ilp": Path("prototype/output/processed/imdb_fragment_assignment_conflict_locality_ilp.csv")
            }
        },

        "updates": {
            "fragments_path": Path("prototype/output/reoptimization/imdb_fragments_updates.csv"),
            "assignment_paths": {
                        "round_robin": Path("prototype/output/processed/imdb_fragment_assignment_round_robin.csv"),
                        "tuple_ilp": Path("prototype/output/reoptimization/imdb_fragment_assignment_tuple_ilp_updated.csv"),
                        "conflict_locality_ilp": Path("prototype/output/reoptimization/imdb_fragment_assignment_conflict_locality_ilp_updated.csv")
            }
        },
        
        "fragment_item_ids": "title_ids",
        "recovery_directory": Path("prototype/output/recovery"),
        "recovery_output_path": Path("prototype/output/recovery/imdb_recovery_results.csv")
    }
}

DATASET = "imdb"
MODE = "baseline" # baseline oder updates


def find_nodes(placement_type, database_config, mode):
    """
    Lädt je nach Modus die Baseline- oder Update-Nodes.
    """

    if placement_type not in database_config["placements"]:
        raise ValueError(
            f"Den Placement-Typ gibt es nicht: {placement_type}"
        )

    baseline_directory = (
        database_config["placements"][placement_type]["node_output"]
    )

    if mode == "baseline":
        placement_directory = baseline_directory
    elif mode == "updates":
        placement_directory = (
            baseline_directory.parent
            / "updates"
            / placement_type
        )
    else:
        raise ValueError(f"Unbekannter Recovery-Modus: {mode}")

    node_files = sorted(
        placement_directory.glob("node_*.db")
    )

    if not node_files:
        raise FileNotFoundError(
            f"Keine SQLite-Nodes gefunden in "
            f"{placement_directory}. Führe zuerst Skript 11 "
            f"im Modus {mode} aus."
        )

    return node_files

def find_failed_node(node_files, failed_node_id):
    """
    Sucht nach der ausgefallenen Node.
    """
    for node_file in node_files:
        if node_file.stem == failed_node_id:
            return node_file
        
    raise FileNotFoundError(f"Die 'ausgefallene' Node wurde nicht gefunden: {failed_node_id}.")

def search_db(node_file):
    """
    Zeigt die Informationen der ausgefallenen Node.
    """

    with sqlite3.connect(node_file) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """)
        tables = [row[0] for row in cur.fetchall()]

        print(f"\nAusgefallene Datenbank: {node_file}")
        print(f"Tabellen der Datenbank: {tables}")

        for table_name in tables:
            cur.execute(f"PRAGMA table_info({table_name})")
            columns = cur.fetchall()
            print(f"\nSpalten der Tabelle {table_name}")
            for column in columns:
                print(column)

def load_failed_items(node_file, database_config):
    """
    Hiermit werden die Descriptoren einer ausgefallenen Node geladen.
    """

    item_table = database_config["item_table"]
    item_id = database_config["item_id_column"]
    item_name = database_config["item_name_column"]

    with sqlite3.connect(node_file) as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT {item_id}, {item_name}
            FROM {item_table}
        """)

        rows = cur.fetchall()
    return {item_id: item_name for item_id, item_name in rows}


def find_copies(node_files, failed_node, failed_items, database_config):
    """
    Sucht auf restlichen Nodes nach ausgefallenen/verlorenen Items.
    """
    item_table = database_config["item_table"]
    item_id_column = database_config["item_id_column"]

    recovery_items = {item_id: [] for item_id in failed_items}

    for node_file in node_files:
        if node_file == failed_node:
            continue
        with sqlite3.connect(node_file) as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT {item_id_column}
                FROM {item_table}
            """)

            available = {row[0] for row in cur.fetchall()}

        found_items = set(failed_items) & available

        for item_id in found_items:
            recovery_items[item_id].append(node_file.stem)

    return recovery_items

def get_recovery_path(placement_type, failed_node_id, recovery_config, mode):
    recovery_directory = recovery_config["recovery_directory"] / mode / placement_type

    recovery_directory.mkdir(parents=True, exist_ok = True)

    return recovery_directory / f"{failed_node_id}_recovered.db"


def load_failed_fragment_info(placement_type, failed_node_id, recovery_config, mode_config):
    """
    Entnimmt Informationen des Fragments und ihre Fragmentmitgliedschaften, der ausgefallenen Node.
    """

    assignment = pd.read_csv(mode_config["assignment_paths"][placement_type])

    fragments_df = pd.read_csv(mode_config["fragments_path"])

    item_ids_column = recovery_config["fragment_item_ids"]

    # Fragment_IDs auf der failed node werden entnommen
    failed_fragments_ids = set(assignment.loc[assignment["node_id"] == failed_node_id, "fragment_id"])

    failed_fragments = fragments_df[fragments_df["fragment_id"].isin(failed_fragments_ids)].copy()

    fragment_rows = list(failed_fragments[["fragment_id", "scheme", "value", "fragment_size"]].itertuples(index=False, name=None))

    fragment_membership_rows = []

    for row in failed_fragments.itertuples(index=False):
        item_ids = getattr(row, item_ids_column)

        if pd.isna(item_ids) or item_ids == "":
            item_ids_value = []
        else:
            item_ids_value = [item_id.strip() for item_id in str(item_ids).split(",") if item_id.strip()]

        for item_id in item_ids_value:
            fragment_membership_rows.append((row.fragment_id, item_id))

    return fragment_rows, fragment_membership_rows


def recover(placement_type, recover_path, node_files, selected_node, failed_node_id, recovery_config, database_config, mode_config):
    """
    Liest die ausgefallenen Descriptoren und speichert die Descriptoren die aus anderen Nodes recovered werden können in eine Node.
    """

    item_table = database_config["item_table"]
    item_id_column = database_config["item_id_column"]
    item_name_column = database_config["item_name_column"]
    membership_item_column = database_config["membership_item_column"]

    additional_item_columns = database_config.get("additional_item_columns", {})

    item_columns = [item_id_column, item_name_column, *additional_item_columns.keys()]

    node_file_id = {node_file.stem: node_file for node_file in node_files}

    items_recoverable = {}

    for item_id, source_id in selected_node.items():
        items_recoverable.setdefault(source_id, []).append(item_id)

    recovered_rows = []
    fragment_rows, fragment_membership_rows = load_failed_fragment_info(placement_type, failed_node_id=failed_node_id, recovery_config=recovery_config, mode_config=mode_config)

    print(f"Anzahl Fragmente die zu recovern sind: {len(fragment_rows)}")
    print(f"Anzahl Fragmentmitgliedschaften die zu recovern sind: {len(fragment_membership_rows)}")

    for node_id, item_ids in items_recoverable.items():
        path = node_file_id[node_id]

        placeholders = ",".join("?" for _ in item_ids)

        with sqlite3.connect(path) as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT {", ".join(item_columns)}
                FROM {item_table}
                WHERE {item_id_column} IN ({placeholders})
            """, item_ids)

            recovered_rows.extend(cur.fetchall())

    recovered_item_ids = {row[0] for row in recovered_rows}

    recovered_fragment_sizes = {}

    fragment_membership_rows = [membership
                                for membership in fragment_membership_rows
                                if membership[1] in recovered_item_ids]

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
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE {item_table} (
                {", ".join(item_column_def)}
                )
        """)

        cur.executemany(f"""
            INSERT INTO {item_table} ({", ".join(item_columns)}) VALUES({item_placeholders})
        """, recovered_rows)

        conn.commit()

        cur.execute("""
            CREATE TABLE fragments(
                fragment_id TEXT PRIMARY KEY,
                scheme TEXT NOT NULL,
                value TEXT NOT NULL,
                fragment_size INTEGER NOT NULL
                )
        """)

        cur.executemany("""
            INSERT INTO fragments(fragment_id, scheme, value, fragment_size) VALUES(?, ?, ?, ?)
        """, fragment_rows)

        conn.commit()

        cur.execute(f"""
            CREATE TABLE fragment_members(
                fragment_id TEXT NOT NULL,
                {membership_item_column} TEXT NOT NULL,
                PRIMARY KEY (fragment_id, {membership_item_column}))
        """)

        cur.executemany(f"""
            INSERT INTO fragment_members(fragment_id, {membership_item_column}) VALUES(?, ?)
        """, fragment_membership_rows)
        conn.commit()

    return recovered_rows, fragment_rows, fragment_membership_rows

def count_recovered_node_rows(path, table_name):
    with sqlite3.connect(path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def process_recovery(dataset, recovery_config, database_config, mode_config, mode):
    failed_node_id = recovery_config["failed_node_id"]
    recovery_results = []
    for placement_type in mode_config["assignment_paths"]:
        node_files = find_nodes(placement_type, database_config, mode)
        failed_node = find_failed_node(node_files, failed_node_id)
        failed_items = load_failed_items(failed_node, database_config)
        recovery_time_start = time.perf_counter()
        recovery = find_copies(node_files, failed_node, failed_items, database_config)
        recoverable = {item_id: nodes for item_id, nodes in recovery.items() if nodes}
        unrecoverable = {item_id: nodes for item_id, nodes in recovery.items() if not nodes}
        source_nodes = {node_id for nodes in recoverable.values() for node_id in nodes}
        selected_source = {item_id: nodes[0] for item_id, nodes in recoverable.items()}
        recovery_node = set(selected_source.values())

        recover_path = get_recovery_path(placement_type, failed_node_id, recovery_config, mode)
        recovered_items, recovered_fragments, recovered_memberships = recover(placement_type, 
                                                                              recover_path, 
                                                                              node_files, 
                                                                              selected_source, 
                                                                              failed_node_id, 
                                                                              recovery_config, 
                                                                              database_config,
                                                                              mode_config)
        recovery_time = time.perf_counter() - recovery_time_start

        if len(failed_items) > 0:
            recovery_rate = len(recovered_items) / len(failed_items)
        else:
            recovery_rate = 0

        recovery_results.append({
            "placement_type": placement_type,
            "mode": mode,
            "failed_node_id": failed_node_id,
            "amount_affected_items": len(failed_items),
            "amount_recoverable_items": len(recoverable),
            "amount_recovered_items": len(recovered_items),
            "amount_unrecoverable_items": len(unrecoverable),
            "recovery_rate": recovery_rate,
            "available_nodes": len(source_nodes),
            "selected_recovery_nodes": len(recovery_node),
            "recovery_time_sec": recovery_time,
            "amount_recovered_fragments": len(recovered_fragments),
            "amount_recovered_fragment_memberships": len(recovered_memberships)
            })

    recovery_df = pd.DataFrame(recovery_results)
    base_output_path = recovery_config["recovery_output_path"]

    output_path = base_output_path.with_name(f"{base_output_path.stem}_{mode}{base_output_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_df.to_csv(output_path, index=False)

    return recovery_df
        


def main():
    if DATASET not in RECOVERY_CONFIGS:
        raise ValueError(f"Unbekannter Datensatz für Recovery: {DATASET}")

    if DATASET not in DATABASE_CONFIGS:
        raise ValueError(f"Unbekannter Datensatz für Datenbank: {DATASET}")

    recovery_config = RECOVERY_CONFIGS[DATASET]
    database_config = DATABASE_CONFIGS[DATASET]

    if MODE not in {"baseline", "updates"}:
        raise ValueError(f"Unbekannter Recovery Modus mit {MODE}")

    if MODE not in recovery_config:
        raise ValueError(f"Unbekannter MODE {MODE} für Datensatz {DATASET}")

    mode_config = recovery_config[MODE]

    recovery_df = process_recovery(dataset=DATASET, recovery_config=recovery_config, 
                                   database_config=database_config, mode_config=mode_config, mode=MODE)

    print(f"\nRecovery Summary {MODE}:")
    print(recovery_df.to_string(index=False))


if __name__ == "__main__":
    main()