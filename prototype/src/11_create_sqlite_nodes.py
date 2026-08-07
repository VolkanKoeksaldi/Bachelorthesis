from pathlib import Path
from experiment_config import experiment_path
import sqlite3
import pandas as pd

DATASETS = {
    "mesh": {
        "items_path": experiment_path("processed/mesh_descriptors_sample.csv"),

        "item_table": "descriptors",
        "item_id_column": "descriptor_ui",
        "item_name_column": "descriptor_name",
        "fragment_item_ids_column": "descriptor_ids",
        "membership_item_column": "descriptor_ui",

        "additional_item_columns": {
            "metadata_json": "TEXT",
            "item_size_bytes": "INTEGER"
        },

        "placements": {
            "round_robin": {
                "fragments_path": experiment_path("processed/mesh_fragments_sample.csv"),
                "assignment_path": experiment_path("processed/mesh_fragment_assignment_round_robin.csv"),
                "node_output": experiment_path("nodes/mesh/round_robin")
            },

            "tuple_ilp": {
                "fragments_path": experiment_path("processed/mesh_fragments_sample.csv"),
                "assignment_path": experiment_path("processed/mesh_fragment_assignment_tuple_ilp.csv"),
                "node_output": experiment_path("nodes/mesh/tuple_ilp")
            },

            "conflict_locality_ilp": {
                "fragments_path": experiment_path("processed/mesh_fragments_sample.csv"),
                "assignment_path": experiment_path("processed/mesh_fragment_assignment_conflict_locality_ilp.csv"),
                "node_output": experiment_path("nodes/mesh/conflict_locality_ilp")
            },
        },
    },

    "imdb": {
        "items_path": experiment_path("processed/imdb_titles.csv"),

        "placements": {
            "round_robin": {
                    "fragments_path": experiment_path("processed/imdb_fragments.csv"),
                    "assignment_path": experiment_path("processed/imdb_fragment_assignment_round_robin.csv"),
                    "node_output": experiment_path("nodes/imdb/round_robin")
                },

                "tuple_ilp": {
                    "fragments_path": experiment_path("processed/imdb_fragments.csv"),
                    "assignment_path": experiment_path("processed/imdb_fragment_assignment_tuple_ilp.csv"),
                    "node_output": experiment_path("nodes/imdb/tuple_ilp")
                },

                "conflict_locality_ilp": {
                    "fragments_path": experiment_path("processed/imdb_fragments.csv"),
                    "assignment_path": experiment_path("processed/imdb_fragment_assignment_conflict_locality_ilp.csv"),
                    "node_output": experiment_path("nodes/imdb/conflict_locality_ilp")
                },
        },

        "item_table": "title",
        "item_id_column": "title_id",
        "item_name_column": "primary_title",
        "fragment_item_ids_column": "title_ids",
        "membership_item_column": "title_id",

        "additional_item_columns": {
            "metadata_json": "TEXT",
            "item_size_bytes": "INTEGER"
        }
    }
}

UPDATE_PLACEMENTS = {
    "mesh": {
        "round_robin": {
            "items_path": experiment_path("reoptimization/mesh_descriptors_sample_updates.csv"),
            "fragments_path": experiment_path("reoptimization/mesh_fragments_sample_updates.csv"),
            "assignment_path": experiment_path("processed/mesh_fragment_assignment_round_robin.csv"),
            "node_output": experiment_path("nodes/mesh/updates/round_robin")
        },
        "tuple_ilp": {
            "items_path": experiment_path("reoptimization/mesh_descriptors_sample_updates.csv"),
            "fragments_path": experiment_path("reoptimization/mesh_fragments_sample_updates.csv"),
            "assignment_path": experiment_path("reoptimization/mesh_fragment_assignment_tuple_ilp_updated.csv"),
            "node_output": experiment_path("nodes/mesh/updates/tuple_ilp")
        },
        "conflict_locality_ilp": {
            "items_path": experiment_path("reoptimization/mesh_descriptors_sample_updates.csv"),
            "fragments_path": experiment_path("reoptimization/mesh_fragments_sample_updates.csv"),
            "assignment_path": experiment_path("reoptimization/mesh_fragment_assignment_conflict_locality_ilp_updated.csv"),
            "node_output": experiment_path("nodes/mesh/updates/conflict_locality_ilp")
        }
    },

    "imdb": {
        "round_robin": {
            "items_path": experiment_path("reoptimization/imdb_titles_updates.csv"),
            "fragments_path": experiment_path("reoptimization/imdb_fragments_updates.csv"),
            "assignment_path": experiment_path("processed/imdb_fragment_assignment_round_robin.csv"),
            "node_output": experiment_path("nodes/imdb/updates/round_robin")
        },
        "tuple_ilp": {
            "items_path": experiment_path("reoptimization/imdb_titles_updates.csv"),
            "fragments_path": experiment_path("reoptimization/imdb_fragments_updates.csv"),
            "assignment_path": experiment_path("reoptimization/imdb_fragment_assignment_tuple_ilp_updated.csv"),
            "node_output": experiment_path("nodes/imdb/updates/tuple_ilp")
        },
        "conflict_locality_ilp": {
            "items_path": experiment_path("reoptimization/imdb_titles_updates.csv"),
            "fragments_path": experiment_path("reoptimization/imdb_fragments_updates.csv"),
            "assignment_path": experiment_path("reoptimization/imdb_fragment_assignment_conflict_locality_ilp_updated.csv"),
            "node_output": experiment_path("nodes/imdb/updates/conflict_locality_ilp")
        }
    }
}

DATASET = "imdb"
PLACEMENT = "round_robin"
MODE = "baseline" # baseline or updates

def parse_item_ids(item_ids_string):
    """
    Parses item ids field from a fragment csv file into a list.
    Empty and missing fields are converted to an empty list.
    """

    if pd.isna(item_ids_string) or item_ids_string == "":
        return []
    
    return [item_id.strip() for item_id in item_ids_string.split(",") if item_id.strip()]

def create_tables(connection, config):
    """
    Creates tables in node databases.
    Each node database stores following information:
    1. fragments: fragment metadata of fragments stored on a node
    2. dataset-specific item records
    3. fragment memberships: fragment to item information
    Dataset-specific tables and column names are obtained from the configurations.
    """

    item_table = config["item_table"]
    item_id_column = config["item_id_column"]
    item_name_column = config["item_name_column"]
    membership_item_column = config["membership_item_column"]
    additional_item_columns = config.get("additional_item_columns", {})

    item_column_def = [f"{item_id_column} TEXT PRIMARY KEY",
                       f"{item_name_column} TEXT NOT NULL"]

    for column_name, column_type in additional_item_columns.items():
        item_column_def.append(f"{column_name} {column_type}")

    cursor = connection.cursor()

    # Stores metadata for every fragment assigned to a specific node
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fragments (
            fragment_id TEXT PRIMARY KEY,
            scheme TEXT NOT NULL,
            value TEXT NOT NULL,
            fragment_size INTEGER NOT NULL
        )
    """)

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {item_table} (
                   {", ".join(item_column_def)}
                )
        """)

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS fragment_members (
            fragment_id TEXT NOT NULL,
            {membership_item_column} TEXT NOT NULL,
            PRIMARY KEY (fragment_id, {membership_item_column}),
            FOREIGN KEY (fragment_id) REFERENCES fragments(fragment_id),
            FOREIGN KEY ({membership_item_column}) REFERENCES {item_table}({item_id_column})
        )
    """)

    connection.commit()


def insert_fragment(connection, fragment_row, item_lookup, config):
    """
    Inserts a fragment and its item memberships into a node.

    Items shared by several fragments on the same node are stored only once
    in the dataset-specific item table.
    Every fragment membership is retained in "fragment_members".
    """

    item_table = config["item_table"]
    item_id_column = config["item_id_column"]
    item_name_column = config["item_name_column"]
    membership_item_column = config["membership_item_column"]
    additional_item_columns = list(config.get("additional_item_columns", {}).keys())

    cursor = connection.cursor()

    fragment_id = fragment_row["fragment_id"]
    scheme = fragment_row["scheme"]
    value = fragment_row["value"]
    fragment_size = int(fragment_row["fragment_size"])

    item_ids = parse_item_ids(fragment_row[config["fragment_item_ids_column"]])

    # Inserts fragment metadata before its foreign-key memberships
    cursor.execute("""
        INSERT OR REPLACE INTO fragments (
            fragment_id,
            scheme,
            value,
            fragment_size
        )
        VALUES (?, ?, ?, ?)
    """, (
        fragment_id,
        scheme,
        value,
        fragment_size
    ))

    # Inserts item data and fragment memberships
    for item_id in item_ids:
        item_data = item_lookup.get(item_id)

        if item_data is None:
            item_data = {item_name_column: "UNKNOWN"}

        # * used to get the elements of a list
        item_columns = [item_id_column, item_name_column, *additional_item_columns]

        # * used to unpack the created the list of additional item column values
        item_values = [item_id, item_data.get(item_name_column, "UNKNOWN"), 
                       *[item_data.get(column_name) for column_name in additional_item_columns]]

        placeholders = ", ".join("?" for _ in item_columns)

        cursor.execute(f"""
            INSERT OR IGNORE INTO {item_table} (
                {", ".join(item_columns)}
            )
            VALUES ({placeholders})
        """, (
            item_values
        ))

        cursor.execute(f"""
            INSERT OR IGNORE INTO fragment_members (
                fragment_id,
                {membership_item_column}
            )
            VALUES (?, ?)
        """, (
            fragment_id,
            item_id
        ))


def create_sqlite_nodes(items_df, assignment_df, config):
    """
    Builds a SQLite database file for every node in the assignment.
    Existing node database files in the selected output directory are removed before creating nodes
    for the current assignment.
    """
    
    node_output = config["node_output"]
    node_output.mkdir(parents=True, exist_ok=True)

    # Removes stale node databases from a previous run
    for old_db in node_output.glob("node_*.db"):
        old_db.unlink()

    item_id_column = config["item_id_column"]
    item_name_column = config["item_name_column"]
    additional_item_columns = list(config.get("additional_item_columns", {}).keys())
    item_data_columns = [item_name_column, *additional_item_columns]

    item_lookup = (
        items_df
        .drop_duplicates(subset=[item_id_column])
        .set_index(item_id_column)[item_data_columns]
        .to_dict(orient="index")
    )

    for node_id, node_fragments in assignment_df.groupby("node_id"):
        db_path = node_output / f"{node_id}.db"

        with sqlite3.connect(db_path) as conn:
            # The schema must exist before fragments and memberships are inserted
            create_tables(conn, config)

            for _, fragment_row in node_fragments.iterrows():
                insert_fragment(conn, fragment_row, item_lookup, config)

        print(f"Created {db_path} with {len(node_fragments)} fragments")


def verify_nodes(config):
    """
    Reports number of fragments, items, and memberships on each node.
    Verifies the information and gives out a summary after all nodes have been created.
    """

    print("\nVerification:")
    node_output = config["node_output"]
    item_table = config["item_table"]

    for db_path in sorted(node_output.glob("node_*.db")):
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM fragments")
        fragment_count = cursor.fetchone()[0]

        cursor.execute(f"SELECT COUNT(*) FROM {item_table}")
        item_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM fragment_members")
        membership_count = cursor.fetchone()[0]

        connection.close()

        print(
            db_path.name,
            "| fragments:",
            fragment_count,
            "| items:",
            item_count,
            "| memberships:",
            membership_count
        )

def add_fragment_information(assignment_df, fragments_df, fragment_item_ids_column):
    """
    Refreshes assignment rows with information from current fragment file.
    Thus prevents updates mode from accidentally reusing outdated item ids or fragment sizes
    that may still be present in the baseline assignment file.
    """

    required_columns = [
        "scheme",
        "value",
        "fragment_size",
        fragment_item_ids_column
    ]

    if not assignment_df["fragment_id"].is_unique:
        raise ValueError("The assignment contains more than one node assignment for the same fragment_id")

    if not fragments_df["fragment_id"].is_unique:
        raise ValueError("The fragment file contains duplicate fragment_id values.")

    missing_fragment_ids = (
        set(assignment_df["fragment_id"]) - set(fragments_df["fragment_id"])
    )

    if missing_fragment_ids:
        raise ValueError(
            "Following top 10 assigned fragments are missing from the fragment file "
            f"{sorted(missing_fragment_ids)[:10]}"
        )

    unassigned_fragment_ids = (set(fragments_df["fragment_id"]) - set(assignment_df["fragment_id"]))

    if unassigned_fragment_ids:
        raise ValueError(f"{len(unassigned_fragment_ids)} fragments are not assigned on a node.")

    # Discards outdated fragment data from assignment file
    # Required columns are dropped from the placement DataFrame and if one or multiple of these
    # do not exist, then it ignores the errors. Otherwise an error message would interrupt the program.
    placement_df = assignment_df.drop(columns=required_columns, errors="ignore")

    # Collects all the fragment information that is available using fragment_id and then the elements of
    # the list required_columns
    current_fragment_information = fragments_df[["fragment_id", *required_columns]]

    # merges the current fragment information with placement_df on the fragment_id
    # validate="one_to_one" checks whether every fragment_id appears exactly once in both tables
    return placement_df.merge(current_fragment_information, on="fragment_id", how="left", validate="one_to_one")

def process_create_sqlite_nodes(config, placement):
    """
    Loads placement data.
    Creates node databases and verifies them.
    """

    items_path = placement.get("items_path", config["items_path"])
    items_df = pd.read_csv(items_path)
    fragments_df = pd.read_csv(placement["fragments_path"])
    assignment_df = pd.read_csv(placement["assignment_path"])

    # refreshes assignment with newest fragment information
    assignment_df = add_fragment_information(assignment_df, fragments_df, config["fragment_item_ids_column"])

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