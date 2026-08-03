from pathlib import Path
import sqlite3
import pandas as pd
import json

DATASETS = {
    "mesh": {
        "item_table": "descriptors",
        "item_id_column": "descriptor_ui",
        "item_name_column": "descriptor_name",
        "membership_item_column": "descriptor_ui",

        "additional_item_columns": {
            "metadata_json": "TEXT",
            "item_size_bytes": "INTEGER"
        },

        "placements": {
            "round_robin": {
                "node_output": Path("prototype/output/nodes/mesh/round_robin")
            },

            "tuple_ilp": {
                "node_output": Path("prototype/output/nodes/mesh/tuple_ilp")
            },

            "conflict_locality_ilp": {
                "node_output": Path("prototype/output/nodes/mesh/conflict_locality_ilp")
            }
        }
    },

    "imdb": {
        "item_table": "title",
        "item_id_column": "title_id",
        "item_name_column": "primary_title",
        "membership_item_column": "title_id",

        "additional_item_columns": {
            "metadata_json": "TEXT",
            "item_size_bytes": "INTEGER"
        },

        "placements": {
            "round_robin": {
                "node_output": Path("prototype/output/nodes/imdb/round_robin")
            },

            "tuple_ilp": {
                "node_output": Path("prototype/output/nodes/imdb/tuple_ilp")
            },

            "conflict_locality_ilp": {
                "node_output": Path("prototype/output/nodes/imdb/conflict_locality_ilp")
            }
        }
    }
}


DATASET = "imdb"


def search_array(search_term):
    """
    Wandelt einen kommagetrennten String in eine Liste von Such-Termen/Spalten auf

    Beispiel:
    "descriptor_ui, fragment_id" -> ["descriptor_ui", "fragment_id"]
    """

    if pd.isna(search_term) or search_term == "":
        return []
    
    return [col.strip() for col in search_term.split(",") if col.strip()]

def create_generated_metadata(item_id, item_name):
    """
    Erstellt Zusatzinformationen für ein künstliches Workload-Item.
    """

    metadata_json = json.dumps({}, ensure_ascii=False, separators=(",", ":"))

    item_size_bytes = len((str(item_id) + str(item_name) + metadata_json).encode("utf-8"))

    return {"metadata_json": metadata_json, "item_size_bytes": item_size_bytes}

def select(search, config, node_id, placement_type, table, where = None, parameters = ()):
    """
    Funktion um eine SELECT-Abfrage auf einer Node-Datenbank auszuführen:

    search:
        Spalten die hinter SELECT definiert werden, zum Beispiel "descriptor_ui" oder "*"
    
    node_id:
        Ist die Node der Datenbank zum Beispiel "node_1"
    
    placement_type:
        "round_robin", "tuple_ilp" oder "conflict_locality_ilp"
    
    table:
        "fragments", "descriptors" oder "fragment_members"
    
    where:
        Bestimmt optionale WHERE-Bedingungen mit "?" als Platzhalter. Zum Beispiel "descriptor_ui = ?"
    
    parameters:
        Werte für die Platzhalter im WHERE
    """

    valid_tables = get_valid_tables(config)
    node_output = get_node_output(config, placement_type)

    if table not in valid_tables:
        raise ValueError(f"Unbekannte Tabelle: {table}")

    search_query = search_array(search)

    if not search_query:
        raise ValueError(f"Es wurde nichts angegeben.")
    
    search_sql = ", ".join(search_query)

    db_path = node_output / f"{node_id}.db"

    if not db_path.exists():
        raise FileNotFoundError(f"Die Datenbank konnte nicht gefunden werden: {db_path}")

    query = f"SELECT {search_sql} FROM {table}"

    if where:
        query += f" WHERE {where}"

    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, parameters)

            rows = cur.fetchall()

    except sqlite3.Error as error:
        raise RuntimeError(f"Fehler beim Ausführen der Query: {query}") from error

    return rows


def insert_into_node(item_id, item_name, fragment_ids, node_id, placement_type, config):
    """
    Speichert Descriptor und seine Fragmentzugehörigkeiten in einer bestimmten Node.
    """

    item_table = config["item_table"]
    item_id_column = config["item_id_column"]
    item_name_column = config["item_name_column"]
    membership_item_column = config["membership_item_column"]

    additional_item_columns = list(config.get("additional_item_columns", {}).keys())

    generated_item_data = create_generated_metadata(item_id, item_name)

    item_columns = [item_id_column, item_name_column, *additional_item_columns]

    item_values = [item_id, item_name, *[generated_item_data.get(column_name) for column_name in additional_item_columns]]

    placeholders = ", ".join("?" for _ in item_columns)

    node_output = get_node_output(config, placement_type)
    
    db_path = node_output / f"{node_id}.db"

    if not db_path.exists():
        raise FileNotFoundError(f"Datenbank konnte nicht gefunden werden: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # Descriptor wird auf dieser Node einmal gespeichert
        cur.execute(
            f"""
            INSERT INTO {item_table} (
                {", ".join(item_columns)}
            )
            VALUES ({placeholders})
            """,
            item_values
        )

        inserted_membership = 0

        for fragment_id in fragment_ids:
            # Hier wird geprüft, ob das Fragment dann auch wirklich auf der Node liegt
            cur.execute(
                """
                SELECT *
                FROM fragments
                WHERE fragment_id = ?
                """,
                (fragment_id,)
            )

            if cur.fetchone() is None:
                raise ValueError(
                    f"Fragment {fragment_id} is nicht auf Node {node_id} gespeichert."
                )

            cur.execute(
                f"""
                INSERT INTO fragment_members(
                    fragment_id,
                    {membership_item_column}
                )
                VALUES (?, ?)
                """,
                (fragment_id, item_id)
            )

            # Falls neue Membership entsteht wird hochgezählt:
            if cur.rowcount == 1:
                inserted_membership += 1
                cur.execute(
                    """
                    UPDATE fragments
                    SET fragment_size = fragment_size + 1
                    WHERE fragment_id = ?
                    """,
                    (fragment_id,)
                )

    return inserted_membership


def validate_updates(table, updates, config):
    """
    Prüft den SET Ausdruck im UPDATE.
    Beispiel:
    "descriptor_name = ?"
    "scheme = ?, value = ?"
    """

    if not updates or not updates.strip():
        raise ValueError("Es wurden keine SET-Bedingungen gefunden.")

    assignment_array = [
        assignment.strip()
        for assignment in updates.split(",")
    ]

    validated = []

    valid_columns = get_valid_columns(config)

    for assignment in assignment_array:
        assignment_part = assignment.split("=")

        if len(assignment_part) != 2:
            raise ValueError(f"Es gibt keinen Platzhalter für die SET-Zuweisung: {assignment}")

        column = assignment_part[0].strip()
        placeholders = assignment_part[1].strip()

        if column not in valid_columns[table]:
            raise ValueError(f"Spalte {column} existiert nicht in Tabelle {table}.")

        if placeholders != "?":
            raise ValueError(f"In der SET-Zuweisung wurde für {column} kein ?-Platzhalter verwendet.")

        if column in validated:
            raise ValueError(f"Spalte {column} wurde mehrfach angegeben.")

        validated.append(column)

    normalized_set = ", ".join(f"{column} = ?"
                               for column in validated)

    return normalized_set, len(validated)


def update(node_id, config, placement_type, table, updates, where = None, parameters = ()):
    """
    Aktualisiert Datensätze in einer Node-Datenbank.
    
    updates:
        String mit Spaltenname und den Platzhaltern.
        Bsp: "descriptor_name = ?, fragment_size = ?"
    
    where:
        Optionale WHERE-Bedingungen mit Parameter Platzhaltern.
        Bsp: "descriptor_ui = ?"
    
    parameters:
        Werte für Platzhalter.
    """

    valid_tables = get_valid_tables(config)

    if table not in valid_tables:
        raise ValueError(f"Unbekannte Tabelle: {table}")

    if not updates:
        raise ValueError(f"Es wurde nichts zum aktualisieren eingegeben.")

    validated_set, set_parameter_count = (validate_updates(table, updates, config))

    if where:
        where_param_count = where.count("?")
    else:
        where_param_count = 0

    expected_param_count = (set_parameter_count + where_param_count)

    if len(parameters) != expected_param_count:
        raise ValueError(f"Es wurden zwar {len(parameters)} eingegeben, aber es wurden {expected_param_count} erwartet.")

    node_output = get_node_output(config, placement_type)
    db_path = node_output / f"{node_id}.db"

    if not db_path.exists():
        raise FileNotFoundError(f"Datenbank konnte nicht gefunden werden: {db_path}")

    query = f"UPDATE {table} SET {validated_set}"

    if where:
        query += f" WHERE {where}"

    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, parameters)

            updated_rows = cur.rowcount

    except sqlite3.Error as error:
        raise RuntimeError(f"Fehler beim Ausführen der Query: {query}") from error

    return updated_rows

def validate_delete(table, where, config):
    """
    Prüft die WHERE-Bedingungen vom DELETE.
    Beispiel:
    "descriptor_ui = ?"
    "fragment_id = ? AND descriptor_ui = ?"
    """

    if not where or not where.strip():
        raise ValueError(f"Keine WHERE-Bedingung wurde angegeben.")

    condition_array = [
        condition.strip()
        for condition in where.split("AND")
    ]

    validated = []

    valid_columns = get_valid_columns(config)

    for condition in condition_array:
        condition_part = condition.split("=")
        if len(condition_part) != 2:
            raise ValueError(f"Ungültige WHERE-Bedingung, da keine Platzhalter vorhanden sind: {condition}")

        column = condition_part[0].strip()
        placeholder = condition_part[1].strip()

        if column not in valid_columns[table]:
            raise ValueError(f"Unbekannte Spalte {column} in Tabelle {table}")

        if placeholder != "?":
            raise ValueError(f"Für Spalte {column} wurde kein ?-Platzhalter verwendet.")

        validated.append(column)

    normalized_where = " AND ".join(f"{column} = ?"
                                    for column in validated)

    return normalized_where, len(validated)


def delete(node_id, config, placement_type, table, where=None, parameters=(), delete_all=False):
    """
    Löscht rows in einer Node-Datenbank.
    
    where:
        WHERE-Bedingungen mit Parameter Platzhaltern.
        Bsp: "descriptor_ui = ?"
    
    parameters:
        Werte für Platzhalter.

    delete_all:
        Muss True sein, falls man die gesamte Tabelle löschen will.
    """

    valid_tables = get_valid_tables(config)

    if table not in valid_tables:
        raise ValueError(f"Unbekannte Tabelle: {table}")

    if where:
        validated_where, expected_param_count = (validate_delete(table, where, config))
    else:
        if not delete_all:
            raise ValueError("Ohne jegliche WHERE-Bedingung würde hier die Tabelle gelöscht werden." \
            "Falls dies erwünscht ist, so muss explizit im Aufruf, 'delete_all=True' gesetzt werden.")
        
        validated_where = None
        expected_param_count = 0

    if len(parameters) != expected_param_count:
        raise ValueError(f"Es wurden zwar {len(parameters)} eingegeben, aber es wurden {expected_param_count} erwartet.")

    node_output = get_node_output(config, placement_type)
    db_path = node_output / f"{node_id}.db"

    if not db_path.exists():
        raise FileNotFoundError(f"Datenbank konnte nicht gefunden werden: {db_path}")

    query = f"DELETE FROM {table}"

    if validated_where:
        query += f" WHERE {validated_where}"

    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, parameters)

            deleted_rows = cur.rowcount

    except sqlite3.Error as error:
        raise RuntimeError(f"Fehler beim Ausführen der Query: {query}") from error

    return deleted_rows

def delete_membership_from_node(fragment_id, item_id, node_id, placement_type, config):
    """
    Löscht die Fragmentzugehörigkeit und reduziert diese mit einer gespeicherten Fragmentgröße
    """

    membership_item_column = config["membership_item_column"]

    node_output = get_node_output(config, placement_type)
    db_path = node_output / f"{node_id}.db"

    if not db_path.exists():
        raise FileNotFoundError(f"Datenbank konnte nicht gefunden werden: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            DELETE FROM fragment_members
            WHERE fragment_id = ?
            AND {membership_item_column} = ?
            """,
            (fragment_id, item_id)
        )

        deleted_rows = cur.rowcount

        if deleted_rows == 1:
            cur.execute(
                """
                UPDATE fragments
                SET fragment_size = fragment_size - 1
                WHERE fragment_id = ?
                """,
                (fragment_id,)
            )
    return deleted_rows


# ----------------------------------------------------------------
# Hilfsfunktionen für distributed Operationen
# ----------------------------------------------------------------
def get_node_output(config, placement_type):
    """
    Gibt das node-Verzeichnis für ein Placement Typ zurück
    """
    if placement_type not in config["placements"]:
        raise ValueError(f"Unbekannter Placement Typ {placement_type}")

    return config["placements"][placement_type]["node_output"]

def get_nodes(config, placement_type):
    """
    Gibt alle node-IDs einer Placement-Type zurück.
    """

    node_output = get_node_output(config, placement_type)

    if not node_output.exists():
        raise FileNotFoundError(f"Das Verzeichnis wurde nicht gefunden: {node_output}")

    node_ids = [
        db_path.stem
        for db_path in node_output.glob("node_*.db")
        # isdigit überprüft hierbei die node gefolgt wird von einer Zahl, also dass node_test.db nicht berücksichtigt wird
        if db_path.stem.replace("node_", "").isdigit()
    ]

    return node_ids


def get_valid_tables(config):
    """
    Gibt die Tabellen für ein Dataset zurück.
    """
    return {"fragments", config["item_table"], "fragment_members"}

def get_valid_columns(config):
    """
    Gibt die Spalten der Tabellen zurück.
    """

    additional_item_columns = set(config.get("additional_item_columns", {}).keys())

    return {"fragments": {
            "fragment_id",
            "scheme",
            "value",
            "fragment_size"
        },

        config["item_table"]: {
            config["item_id_column"],
            config["item_name_column"],
            *additional_item_columns
        },

        "fragment_members": {
            "fragment_id",
            config["membership_item_column"]
        }
    }

def find_item_nodes(item_id, placement_type, config):
    """
    Gibt alle Nodes zurück, worauf eine bestimmte descriptor_ui gespeichert ist.
    """

    item_table = config["item_table"]
    item_id_column = config["item_id_column"]

    nodes_array = []

    all_nodes = get_nodes(config, placement_type)

    for node_id in all_nodes:
        result = select(
            search=item_id_column,
            config=config,
            node_id=node_id, 
            placement_type=placement_type,
            table=item_table,
            where=f"{item_id_column}=?",
            parameters=(item_id,)
            )

        if result:
            nodes_array.append(node_id)

    return nodes_array

def find_fragment_nodes(fragment_id, placement_type, config):
    """
    Gibt alle Nodes zurück, worauf eine bestimmte fragment_id gespeichert ist.
    """
    nodes_array = []

    all_nodes = get_nodes(config, placement_type)

    for node_id in all_nodes:
        result = select(
            search="fragment_id",
            config=config,
            node_id=node_id,
            placement_type=placement_type,
            table="fragments",
            where="fragment_id=?",
            parameters=(fragment_id,)
        )

        if result:
            nodes_array.append(node_id)

    return nodes_array

def find_item_fragments(item_id, placement_type, config):
    """
    Gibt alle Fragmente eines Descriptors zurück.
    """
    membership_item_column = config["membership_item_column"]

    item_nodes = find_item_nodes(item_id, placement_type, config)

    fragment_ids = set()

    for node_id in item_nodes:
        result = select(
            search="fragment_id",
            config=config,
            node_id=node_id,
            placement_type=placement_type,
            table="fragment_members",
            where=f"{membership_item_column} = ?",
            parameters=(item_id,)
        )

        for row in result:
            fragment_ids.add(row[0])

    return list(fragment_ids)


# ---------------------------------------------------------------
# Distributed Operationen
# ---------------------------------------------------------------
def select_item(item_id, placement_type, config):
    """
    Sucht nach item_id im gesamten verteilten System
    """
    item_table = config["item_table"]
    item_id_column = config["item_id_column"]

    item_nodes = find_item_nodes(item_id, placement_type, config)

    fragment_ids = find_item_fragments(item_id, placement_type, config)

    if not item_nodes:
        return {
            "rows": [],
            "contacted_nodes": [],
            "available_nodes": [],
            "fragment_ids": []
        }

    selected_node = item_nodes[0]

    result = select(
        search="*",
        config=config,
        node_id=selected_node,
        placement_type=placement_type,
        table=item_table,
        where=f"{item_id_column} = ?",
        parameters=(item_id,)
    )

    return {"rows": result, "contacted_nodes": [selected_node], 
            "available_nodes": item_nodes, "fragment_ids": fragment_ids}


def insert_item(item_id, item_name, fragment_ids, placement_type, config):
    """
    Fügt einen Item auf Nodes ein, auf denen seine Fragmente gespeichert sind.
    """

    fragments_in_node = {}

    for fragment_id in fragment_ids:
        fragment_nodes = find_fragment_nodes(fragment_id, placement_type, config)

        if not fragment_nodes:
            raise ValueError(f"Fragment {fragment_id} wurde auf keiner der Nodes gefunden.")

        node_id = fragment_nodes[0]

        if node_id not in fragments_in_node:
            fragments_in_node[node_id] = []

        fragments_in_node[node_id].append(fragment_id)

    count_inserted_memberships = 0

    for node_id, node_fragments in fragments_in_node.items():
        count_inserted_memberships += insert_into_node(
            item_id=item_id,
            config=config,
            item_name=item_name,
            fragment_ids=node_fragments,
            node_id=node_id,
            placement_type=placement_type
        )

    return {"inserted_item_id": item_id, "item_name": item_name
            , "inserted_memberships": count_inserted_memberships, "contacted_nodes": list(fragments_in_node.keys())}


def update_item(item_id, update_item_name, placement_type, config):
    """
    Updaten von einem Item auf allen Nodes, die diesen enthalten.
    """

    item_table = config["item_table"]
    item_id_column = config["item_id_column"]
    item_name_column = config["item_name_column"]

    item_nodes = find_item_nodes(item_id, placement_type, config)

    updated_rows = 0

    for node_id in item_nodes:
        curr_item = select(
            search="metadata_json",
            config=config,
            node_id=node_id,
            placement_type=placement_type,
            table=item_table,
            where=f"{item_id_column} = ?",
            parameters=(item_id,)
        )

        metadata_json = (curr_item[0][0] if curr_item and curr_item[0][0] else "{}")

        item_size_bytes = len((str(item_id) + str(update_item_name) + metadata_json).encode("utf-8"))

        result = update(
            node_id=node_id,
            config=config,
            placement_type=placement_type,
            table = item_table,
            updates = f"{item_name_column} = ?, "
                        "item_size_bytes = ?",
            where = f"{item_id_column} = ?",
            parameters = (update_item_name, item_size_bytes, item_id)
        )

        updated_rows += result

    return {"updated_rows": updated_rows, "contacted_nodes": item_nodes}


def delete_item(item_id, placement_type, config):
    """
    Löscht einen item mit ihren Memberships auf allen Nodes, die diesen enthalten.
    """

    item_table = config["item_table"]
    item_id_column = config["item_id_column"]
    membership_item_column = config["membership_item_column"]

    item_nodes = find_item_nodes(item_id, placement_type, config)

    deleted_rows = 0
    deleted_memberships = 0

    for node_id in item_nodes:
        memberships = select(
            search="fragment_id",
            config=config,
            node_id=node_id,
            placement_type=placement_type,
            table="fragment_members",
            where=f"{membership_item_column} = ?",
            parameters=(item_id,)
        )

        for membership in memberships:
            fragment_id = membership[0]

            deleted_memberships += delete_membership_from_node(
                fragment_id=membership[0],
                config=config,
                item_id=item_id,
                node_id=node_id,
                placement_type=placement_type
            )

        deleted_rows += delete(
            node_id=node_id,
            config=config,
            placement_type=placement_type,
            table=item_table,
            where= f"{item_id_column} = ?",
            parameters=(item_id,),
        )

    return {"deleted_rows": deleted_rows, "deleted_memberships": deleted_memberships, "contacted_nodes": item_nodes}

    