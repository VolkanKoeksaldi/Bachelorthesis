import re
import pandas as pd


def expand_base_table(base_df, copy_factor, id_column, id_prefix):
    """
    Duplicates a base DataFrame according to the copy factor 
    and assigns a unique row id to every row.

    Source identifiers and fragmentation attributes are preserved.
    Therefore duplicated rows remain in the same fragment-memberships as their original source.

    Parameters:
        base_df: The base DataFrame that is duplicated
        copy_factor: Duplication number for the base DataFrame
        id_column: Name of the generated unique-id column
        id_prefix: Sets a prefix for every generated ids.

    Returns:
        result: The expanded DataFrame with new unique physical item ids
    """

    if copy_factor < 1:
        raise ValueError("Copy factor must be at least 1.")

    copies = []

    # Creates the configured number of copies while preserving all source attributes
    for copy_number in range(copy_factor):
        current_copy = base_df.copy()
        current_copy["copy_number"] = copy_number
        copies.append(current_copy)

    result = pd.concat(copies, ignore_index=True)

    # Assigns zero-padded sequential ids to all rows.
    width = max(9, len(str(len(result))))
    result[id_column] = [f"{id_prefix}{row_number:0{width}d}" 
                         for row_number in range(1, len(result) + 1)]

    if not result[id_column].is_unique:
        raise ValueError(f"Generated {id_column} values are not unique.")

    return result


def parse_item_ids(value):
    """
    Converts comma-separated item ids stored in a CSV field into a set.

    Parameters:
        value: item ids field from CSV

    Returns:
        Set of item id Strings
    """

    if pd.isna(value) or str(value).strip() == "":
        return set()

    return {item_id.strip() for item_id in str(value).split(",") if item_id.strip()}


def safe_fragment_component(value):
    """
    Converts a value into a deterministic string which can be used as a part of a fragment id.
    """

    text = str(value).strip() or "UNKNOWN"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def validate_fragmentation_memberships(fragments_df, expected_item_ids, expected_schemes, 
                                       item_ids_column):
    """
    Validates that every expected item occurs in exactly one fragment of every scheme.
    Overlaps only allowed between different schemes.

    Parameters:
        fragments_df: A DataFrame that contains the fragment definitions
        expected_item_ids: The expected item ids in every fragmentation
        expected_schemes: The required fragmentation schemes
        item_ids_column: The column that contains comma-separated item ids
    """

    expected_item_ids = {str(item_id) for item_id in expected_item_ids}
    expected_schemes = tuple(expected_schemes)

    # Initializes one fragment-membership set for every expected item-scheme combination
    memberships = {item_id: {scheme: set() for scheme in expected_schemes} 
                   for item_id in expected_item_ids}

    # Iterates over all fragment definitions
    for row in fragments_df.itertuples(index=False):
        if row.scheme not in expected_schemes:
            continue

        # gets item id for every row
        for item_id in parse_item_ids(getattr(row, item_ids_column)):
            # Adds fragment_id to the item's membership set for this scheme
            memberships.setdefault(item_id, {scheme: set() 
                                             for scheme 
                                             in expected_schemes}
                                             )[row.scheme].add(row.fragment_id)

    violations = []

    # Every item must belong to exactly one fragment per scheme
    for item_id, scheme_memberships in memberships.items():
        for scheme in expected_schemes:
            fragment_ids = scheme_memberships[scheme]
            if len(fragment_ids) != 1:
                violations.append((item_id, scheme, sorted(fragment_ids)))

    unexpected_items = set(memberships) - expected_item_ids

    if violations or unexpected_items:
        raise ValueError(f"Invalid fragmentation memberships.")

    # Compares expected and observed numbers of memberships
    expected_memberships = len(expected_item_ids) * len(expected_schemes)
    actual_memberships = sum(len(parse_item_ids(value)) 
                             for value in fragments_df[item_ids_column])
    if actual_memberships != expected_memberships:
        raise ValueError(f"Expected {expected_memberships} memberships, "
                         f"but a total of {actual_memberships} were found.")

    print(f"Validated a total of {len(expected_item_ids)} items belong to exactly "
          f"one fragment in each of {len(expected_schemes)} schemes.")