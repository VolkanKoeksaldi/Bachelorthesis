import re

import pandas as pd


def expand_base_table(base_df, copy_factor, id_column, id_prefix):
    """
    Duplicates base table with copy factor while assigning a unique row ID to every copy.

    Source identifiers and fragmentation attributes are preserved.
    Therefore duplicated rows remain in the same fragments.
    """

    if copy_factor < 1:
        raise ValueError("copy_factor must be at least 1.")

    copies = []

    # Preserves all source attributes and records the coüy associated with each row
    for copy_number in range(copy_factor):
        current_copy = base_df.copy()
        current_copy["copy_number"] = copy_number
        copies.append(current_copy)

    result = pd.concat(copies, ignore_index=True)

    # uses zero padded sequential ids for all rows
    width = max(9, len(str(len(result))))
    result[id_column] = [
        f"{id_prefix}{row_number:0{width}d}"
        for row_number in range(1, len(result) + 1)
    ]

    if not result[id_column].is_unique:
        raise ValueError(f"Generated {id_column} values are not unique.")

    return result


def parse_item_ids(value):
    """
    Parses item ids and convers it into a set of non empty ids.
    """

    if pd.isna(value) or str(value).strip() == "":
        return set()

    return {item_id.strip() for item_id in str(value).split(",") if item_id.strip()}


def safe_fragment_component(value):
    """
    Creates a deterministic component for use in fragment ids.
    """

    text = str(value).strip() or "UNKNOWN"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def validate_fragmentation_memberships(fragments_df, expected_item_ids, expected_schemes, item_ids_column):
    """
    Requires every expected item to occur in exactly one fragment of every scheme.
    Overlap is permitted only between different schemes.
    """

    expected_item_ids = {str(item_id) for item_id in expected_item_ids}
    expected_schemes = tuple(expected_schemes)

    # Initializes one membership set for every expected item-scheme combination
    memberships = {item_id: {scheme: set() for scheme in expected_schemes} for item_id in expected_item_ids}

    for row in fragments_df.itertuples(index=False):
        if row.scheme not in expected_schemes:
            continue

        for item_id in parse_item_ids(getattr(row, item_ids_column)):
            # Reains unexpected item ids
            memberships.setdefault(item_id, {scheme: set() for scheme in expected_schemes})[row.scheme].add(row.fragment_id)

    violations = []

    # Every item must belong to one fragment per scheme
    for item_id, scheme_memberships in memberships.items():
        for scheme in expected_schemes:
            fragment_ids = scheme_memberships[scheme]
            if len(fragment_ids) != 1:
                violations.append((item_id, scheme, sorted(fragment_ids)))

    unexpected_items = set(memberships) - expected_item_ids

    if violations or unexpected_items:
        raise ValueError(
            f"Invalid fragmentation memberships. First violations: {violations[:10]}; unexpected items: {sorted(unexpected_items)[:10]}"
        )

    # checks total amount of expected memberships and detected memberships
    expected_memberships = len(expected_item_ids) * len(expected_schemes)
    actual_memberships = sum(len(parse_item_ids(value)) for value in fragments_df[item_ids_column])

    if actual_memberships != expected_memberships:
        raise ValueError(f"Expected {expected_memberships} memberships, but {actual_memberships} found.")

    print(f"Validated {len(expected_item_ids)} items: exactly one fragment in each of {len(expected_schemes)} schemes.")
