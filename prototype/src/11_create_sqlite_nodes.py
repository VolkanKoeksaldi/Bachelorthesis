from experiment_config import experiment_path
import sqlite3
import pandas as pd

DATASET = "imdb" # imdb or mesh
PLACEMENT = "round_robin" # tuple_ilp, round_robin, or conflict_locality_ilp
MODE = "updates" # baseline or updates


MESH_ADDITIONAL_COLUMNS = {
    "patient_id": "TEXT",
    "term_ui": "TEXT",
    "concept_ui": "TEXT",
    "descriptor_ui": "TEXT",
    "descriptor_name": "TEXT",
    "tree_number": "TEXT",
    "all_tree_numbers": "TEXT",
    "top_category": "TEXT",
    "branch_code": "TEXT",
    "subbranch_code": "TEXT",
    "metadata_json": "TEXT",
    "item_size_bytes": "INTEGER",
    "copy_number": "INTEGER"
}

IMDB_ADDITIONAL_COLUMNS = {
    "source_title_id": "TEXT",
    "title_type": "TEXT",
    "decade": "TEXT",
    "primary_genre": "TEXT",
    "genres": "TEXT",
    "metadata_json": "TEXT",
    "item_size_bytes": "INTEGER",
    "copy_number": "INTEGER"
}

DATASETS = {
    "mesh": {
        "items_path": experiment_path("processed/mesh_terms.csv"),
        "item_table": "ill",
        "item_id_column": "tuple_id",
        "item_name_column": "mesh_term",
        "fragment_item_ids_column": "tuple_ids",
        "membership_item_column": "tuple_id",
        "additional_item_columns": MESH_ADDITIONAL_COLUMNS,
        "placements": {
            "round_robin": {
                "fragments_path": experiment_path("processed/mesh_fragments.csv"),
                "assignment_path": experiment_path(
                    "processed/mesh_fragment_assignment_round_robin.csv"),
                "node_output": experiment_path("nodes/mesh/round_robin")
            },
            "tuple_ilp": {
                "fragments_path": experiment_path("processed/mesh_fragments.csv"),
                "assignment_path": experiment_path(
                    "processed/mesh_fragment_assignment_tuple_ilp.csv"),
                "node_output": experiment_path("nodes/mesh/tuple_ilp")
            },
            "conflict_locality_ilp": {
                "fragments_path": experiment_path("processed/mesh_fragments.csv"),
                "assignment_path": experiment_path(
                    "processed/mesh_fragment_assignment_conflict_locality_ilp.csv"),
                "node_output": experiment_path("nodes/mesh/conflict_locality_ilp")
            }
        }
    },

    "imdb": {
        "items_path": experiment_path("processed/imdb_titles.csv"),
        "item_table": "title",
        "item_id_column": "title_id",
        "item_name_column": "primary_title",
        "fragment_item_ids_column": "title_ids",
        "membership_item_column": "title_id",
        "additional_item_columns": IMDB_ADDITIONAL_COLUMNS,
        "placements": {
            "round_robin": {
                "fragments_path": experiment_path("processed/imdb_fragments.csv"),
                "assignment_path": experiment_path(
                    "processed/imdb_fragment_assignment_round_robin.csv"),
                "node_output": experiment_path("nodes/imdb/round_robin")
            },
            "tuple_ilp": {
                "fragments_path": experiment_path("processed/imdb_fragments.csv"),
                "assignment_path": experiment_path(
                    "processed/imdb_fragment_assignment_tuple_ilp.csv"),
                "node_output": experiment_path("nodes/imdb/tuple_ilp")
            },
            "conflict_locality_ilp": {
                "fragments_path": experiment_path("processed/imdb_fragments.csv"),
                "assignment_path": experiment_path(
                    "processed/imdb_fragment_assignment_conflict_locality_ilp.csv"),
                "node_output": experiment_path("nodes/imdb/conflict_locality_ilp")
            }
        }
    }
}

UPDATE_PLACEMENTS = {
    "mesh": {
        "round_robin": {
            "items_path": experiment_path("reoptimization/mesh_terms_updates.csv"),
            "fragments_path": experiment_path("reoptimization/mesh_fragments_updates.csv"),
            "assignment_path": experiment_path(
                "processed/mesh_fragment_assignment_round_robin.csv"),
            "node_output": experiment_path("nodes/mesh/updates/round_robin")
        },
        "tuple_ilp": {
            "items_path": experiment_path("reoptimization/mesh_terms_updates.csv"),
            "fragments_path": experiment_path("reoptimization/mesh_fragments_updates.csv"),
            "assignment_path": experiment_path(
                "reoptimization/mesh_fragment_assignment_tuple_ilp_updated.csv"),
            "node_output": experiment_path("nodes/mesh/updates/tuple_ilp")
        },
        "conflict_locality_ilp": {
            "items_path": experiment_path("reoptimization/mesh_terms_updates.csv"),
            "fragments_path": experiment_path("reoptimization/mesh_fragments_updates.csv"),
            "assignment_path": experiment_path(
                "reoptimization/mesh_fragment_assignment_conflict_locality_ilp_updated.csv"),
            "node_output": experiment_path("nodes/mesh/updates/conflict_locality_ilp")
        }
    },
    "imdb": {
        "round_robin": {
            "items_path": experiment_path("reoptimization/imdb_titles_updates.csv"),
            "fragments_path": experiment_path("reoptimization/imdb_fragments_updates.csv"),
            "assignment_path": experiment_path(
                "processed/imdb_fragment_assignment_round_robin.csv"),
            "node_output": experiment_path("nodes/imdb/updates/round_robin")
        },
        "tuple_ilp": {
            "items_path": experiment_path("reoptimization/imdb_titles_updates.csv"),
            "fragments_path": experiment_path("reoptimization/imdb_fragments_updates.csv"),
            "assignment_path": experiment_path(
                "reoptimization/imdb_fragment_assignment_tuple_ilp_updated.csv"),
            "node_output": experiment_path("nodes/imdb/updates/tuple_ilp")
        },
        "conflict_locality_ilp": {
            "items_path": experiment_path("reoptimization/imdb_titles_updates.csv"),
            "fragments_path": experiment_path("reoptimization/imdb_fragments_updates.csv"),
            "assignment_path": experiment_path(
                "reoptimization/imdb_fragment_assignment_conflict_locality_ilp_updated.csv"),
            "node_output": experiment_path("nodes/imdb/updates/conflict_locality_ilp")
        }
    }
}


def parse_item_ids(item_ids_string):
    """
    Converts comma-separated item ids stored in a CSV field into a set.

    Parameters:
        item_ids_string: item ids field from CSV

    Returns:
        Set of item id Strings
    """

    if pd.isna(item_ids_string) or item_ids_string == "":
        return []
    
    return [item_id.strip() for item_id in item_ids_string.split(",") if item_id.strip()]


def create_tables(connection, config):
    """
    Creates required tables in a SQLite node databases.

    Each node database stores:
        1. fragments: fragment metadata of fragments stored on a node
        2. dataset-specific item records
        3. fragment memberships: fragment to item information

    Dataset-specific tables and column names are obtained from the config.
    """

    item_columns = [f"{config['item_id_column']} TEXT PRIMARY KEY", 
                    f"{config['item_name_column']} TEXT NOT NULL"]
    item_columns.extend(f"{name} {column_type}" for name, column_type 
                        in config["additional_item_columns"].items())

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS fragments (
            fragment_id TEXT PRIMARY KEY,
            scheme TEXT NOT NULL,
            relaxation_attribute TEXT NOT NULL,
            value TEXT NOT NULL,
            cluster_head TEXT NOT NULL,
            cluster_head_source_id TEXT,
            cluster_method TEXT NOT NULL,
            fragment_size INTEGER NOT NULL
        )
        """
    )

    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {config['item_table']} "
        f"({', '.join(item_columns)})"
    )

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS fragment_members (
            fragment_id TEXT NOT NULL,
            {config['membership_item_column']} TEXT NOT NULL,
            PRIMARY KEY (fragment_id, {config['membership_item_column']}),
            FOREIGN KEY (fragment_id) REFERENCES fragments(fragment_id),
            FOREIGN KEY ({config['membership_item_column']})
                REFERENCES {config['item_table']}({config['item_id_column']})
        )
        """
    )

    connection.commit()


def insert_fragment(connection, fragment_row, item_lookup, config):
    """
    Inserts a fragment and its item memberships into a node.
    Items that appear multiple times in fragments on the same node are only stored once.
    Every fragment membership is retained in "fragment_members".
    """

    fragment_columns = [
        "fragment_id",
        "scheme",
        "relaxation_attribute",
        "value",
        "cluster_head",
        "cluster_head_source_id",
        "cluster_method",
        "fragment_size"
    ]

    fragment_values = [fragment_row.get(column) for column in fragment_columns]

    # Inserts fragment information into the fragments table.
    connection.execute(
        f"INSERT OR REPLACE INTO fragments ({', '.join(fragment_columns)}) "
        f"VALUES ({', '.join('?' for _ in fragment_columns)})",
        fragment_values)

    item_id_column = config["item_id_column"]
    item_name_column = config["item_name_column"]
    additional_columns = list(config["additional_item_columns"])
    item_columns = [item_id_column, item_name_column, *additional_columns]
    placeholders = ", ".join("?" for _ in item_columns)

    # Iterates through every item in fragment
    for item_id in parse_item_ids(fragment_row[config["fragment_item_ids_column"]]):

        # Retrieves item_id for the current item.
        # In case the item is missing from the lookup, it gets the value "UNKNOWN"
        item_data = item_lookup.get(item_id, {item_name_column: "UNKNOWN"})

        item_values = [item_id, item_data.get(item_name_column, "UNKNOWN"),
                       *[item_data.get(column) for column in additional_columns]]

        connection.execute(
            f"INSERT OR IGNORE INTO {config['item_table']} "
            f"({', '.join(item_columns)}) VALUES ({placeholders})",
            item_values)

        connection.execute(
            f"INSERT OR IGNORE INTO fragment_members "
            f"(fragment_id, {config['membership_item_column']}) VALUES (?, ?)",
            (fragment_row["fragment_id"], item_id))


def create_sqlite_nodes(items_df, assignment_df, config):
    """
    Builds a SQLite database file for every node in the assignment.
    Existing node database files in the selected output directory 
    are removed before creating nodes for the current assignment.
    """
    
    node_output = config["node_output"]
    node_output.mkdir(parents=True, exist_ok=True)

    # Existing nodes are removed, and new nodes are created for the current assignment
    for old_db in node_output.glob("node_*.db"):
        old_db.unlink()

    item_data_columns = [config["item_name_column"], *config["additional_item_columns"].keys()]

    # Creates a lookup dictionary that maps each unique item id to its item data.
    item_lookup = (items_df.drop_duplicates(config["item_id_column"])
                   .set_index(config["item_id_column"])[item_data_columns]
                   .to_dict(orient="index"))

    # Groups the assigned fragments by node_id
    for node_id, node_fragments in assignment_df.groupby("node_id"):
        # gets the database paths
        database_path = node_output / f"{node_id}.db"
        # Enables foreign-keys and creates the required tables.
        with sqlite3.connect(database_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            create_tables(connection, config)
            for _, fragment_row in node_fragments.iterrows():
                # Inserts the fragment metadata, item records, and memberships
                insert_fragment(connection, fragment_row, item_lookup, config)
            connection.commit()

        print(f"Created {database_path} with {len(node_fragments)} fragments")


def verify_nodes(config):
    """
    Reports the numbers of fragments, distinct items, and memberships on each node.
    """

    print("\nVerification:")
    for database_path in sorted(config["node_output"].glob("node_*.db")):
        with sqlite3.connect(database_path) as connection:

            fragment_count = connection.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]

            item_count = (connection.execute(f"SELECT COUNT(*) FROM {config['item_table']}")
                          .fetchone()[0])

            membership_count = (connection.execute("SELECT COUNT(*) FROM fragment_members")
                                .fetchone()[0])

        print(f"{database_path.name} | fragments: {fragment_count} | "
              f"items: {item_count} | memberships: {membership_count}")
        connection.close()

def add_fragment_information(assignment_df, fragments_df, item_ids_column):
    """
    Replaces potentially outdates fragment information in the assignment with information
    from the current fragment file.
    
    This prevents updates mode from accidentally reusing outdated item ids or fragment sizes
    that may still be from the baseline assignment.

    Returns:
        Assignment DataFrame that contains the current fragment information
    """

    if not assignment_df["fragment_id"].is_unique:
        raise ValueError("The assignment contains more than one node assignment "
        "for the same fragment_id")

    if not fragments_df["fragment_id"].is_unique:
        raise ValueError("The fragment file contains duplicate fragment_id values.")

    missing = set(fragments_df["fragment_id"]) - set(assignment_df["fragment_id"])

    if missing:
        raise ValueError(f"{len(missing)} fragments have no node assignment.")

    metadata_columns = [
        "scheme",
        "relaxation_attribute",
        "value",
        "cluster_head",
        "cluster_head_source_id",
        "cluster_method",
        "fragment_size",
        item_ids_column,
    ]

    # Discards outdated fragment data from assignment file
    # Required columns are dropped from the placement DataFrame and if one or multiple of these
    # do not exist, then it ignores the errors.
    # Otherwise an error message would interrupt the program.
    placement_df = assignment_df.drop(columns=metadata_columns, errors="ignore")

    # Collects all the fragment information that is available using 
    # fragment_id and then the elements of the list required_columns
    current_fragment_information = fragments_df[["fragment_id", *metadata_columns]]

    # merges the current fragment information with placement_df on the fragment_id
    # validate="one_to_one" checks whether every fragment_id appears exactly once in both tables
    return placement_df.merge(current_fragment_information, on="fragment_id", 
                              how="left", validate="one_to_one")

def process_create_sqlite_nodes(config, placement):
    """
    Loads item, fragment, and assignment data. Creates the SQLite node databases, and
    reports their stored record counts.

    Parameters:
        config: Configurations for dataset-specific paths and parameters
        placement: Paths for the selected placement method and execution mode
    """

    items_path = placement.get("items_path", config["items_path"])
    items_df = pd.read_csv(items_path, dtype={config["item_id_column"]: "string"})
    fragments_df = pd.read_csv(placement["fragments_path"])
    assignment_df = pd.read_csv(placement["assignment_path"])

    # refreshes assignment with newest fragment information
    assignment_df = add_fragment_information(assignment_df, fragments_df, 
                                             config["fragment_item_ids_column"])

    # ** extracts a dictionary and overwrites the value for node_output
    node_config = {**config, "node_output": placement["node_output"]}

    create_sqlite_nodes(items_df, assignment_df, node_config)

    verify_nodes(node_config)

def main():
    if DATASET not in DATASETS:
        raise ValueError(f"Unknown dataset: {DATASET}")

    config = DATASETS[DATASET]

    if PLACEMENT not in config["placements"]:
        raise ValueError(f"Unknown placement: {PLACEMENT}")

    if MODE == "baseline":
        placement = config["placements"][PLACEMENT]
    elif MODE == "updates":
        placement = UPDATE_PLACEMENTS[DATASET][PLACEMENT]
    else:
        raise ValueError(f"Unknown mode: {MODE}")
    
    process_create_sqlite_nodes(config, placement)


if __name__ == "__main__":
    main()