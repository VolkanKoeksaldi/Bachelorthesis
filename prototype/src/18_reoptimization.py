from pathlib import Path
import json
from itertools import combinations, product

import pandas as pd

from experiment_config import REOPTIMIZATION_INSERT_COUNT, experiment_path

CONFIGS = {
    "mesh": {
        "dataset": "mesh",
        "fragments_path": experiment_path("processed/mesh_fragments.csv"),
        "updated_fragments_path": experiment_path("reoptimization/mesh_fragments_updates.csv"),
        "items_path": experiment_path("processed/mesh_terms.csv"),
        "updated_items_path": experiment_path("reoptimization/mesh_terms_updates.csv"),

        "fragment_id": "fragment_id",
        "item_ids": "tuple_ids",
        "fragment_size": "fragment_size",
        "item_id": "tuple_id",
        "item_name": "mesh_term",

        "new_item_prefix": "MT_RE_",
        "new_item_name_prefix": "Reoptimization MeSH Term",
        "insert_count": REOPTIMIZATION_INSERT_COUNT,
        "prefer_new_conflicts": False, # new overlaps from items cannot be generated, only already known overlaps are increased

        "expected_schemes": [
            "top_category",
            "branch_code",
            "subbranch_code"
        ],

        "target_fragments": [
            "top_category_A",
            "branch_code_A05",
            "subbranch_code_A05"
        ],

        "assignment_paths": {
            "tuple_ilp":{
                "baseline": experiment_path("processed/mesh_fragment_assignment_tuple_ilp.csv"),
                "updates": experiment_path("reoptimization/mesh_fragment_assignment_tuple_ilp_updated.csv")
            },

            "conflict_locality_ilp":{
                "baseline": experiment_path("processed/mesh_fragment_assignment_conflict_locality_ilp.csv"),
                "updates": experiment_path("reoptimization/mesh_fragment_assignment_conflict_locality_ilp_updated.csv")
            }
        },

        "comparison_path": experiment_path("reoptimization/mesh_reoptimization_compared.csv"),
        "changed_comparison_path": experiment_path("reoptimization/mesh_reoptimization_changed.csv"),
        "summary_comparison_path": experiment_path("reoptimization/mesh_reoptimization_summary.csv"),
        "node_load_summary_path": experiment_path("reoptimization/mesh_reoptimization_node_load_summary.csv"),

        "node_load_paths": {
            "tuple_ilp":{
                "baseline": experiment_path("results/mesh/tuple_ilp/node_loads.csv"),
                "updates": experiment_path("reoptimization/mesh/tuple_ilp/node_loads_updated.csv")
            },
            "conflict_locality_ilp": {
                "baseline": experiment_path("results/mesh/conflict_locality_ilp/node_loads.csv"),
                "updates": experiment_path("reoptimization/mesh/conflict_locality_ilp/node_loads_updated.csv")
            }
        }
    },

    "imdb": {
        "dataset": "imdb",
        "fragments_path": experiment_path("processed/imdb_fragments.csv"),
        "updated_fragments_path": experiment_path("reoptimization/imdb_fragments_updates.csv"),
        "items_path": experiment_path("processed/imdb_titles.csv"),
        "updated_items_path": experiment_path("reoptimization/imdb_titles_updates.csv"),

        "fragment_id": "fragment_id",
        "item_ids": "title_ids",
        "fragment_size": "fragment_size",
        "item_id": "title_id",
        "item_name": "primary_title",

        "new_item_prefix": "T_RE_",
        "new_item_name_prefix": "Reoptimization Title",
        "insert_count": REOPTIMIZATION_INSERT_COUNT,
        "prefer_new_conflicts": True, # new overlaps can be generated

        "expected_schemes": [
            "title_type",
            "decade",
            "primary_genre"
        ],
        
        "target_fragments": [
            "title_type_movie",
            "decade_2000s",
            "primary_genre_Drama"
        ],

        "assignment_paths": {
            "tuple_ilp":{
                "baseline": experiment_path("processed/imdb_fragment_assignment_tuple_ilp.csv"),
                "updates": experiment_path("reoptimization/imdb_fragment_assignment_tuple_ilp_updated.csv")
            },

            "conflict_locality_ilp":{
                "baseline": experiment_path("processed/imdb_fragment_assignment_conflict_locality_ilp.csv"),
                "updates": experiment_path("reoptimization/imdb_fragment_assignment_conflict_locality_ilp_updated.csv")
            }
        },

        "comparison_path": experiment_path("reoptimization/imdb_reoptimization_compared.csv"),
        "changed_comparison_path": experiment_path("reoptimization/imdb_reoptimization_changed.csv"),
        "summary_comparison_path": experiment_path("reoptimization/imdb_reoptimization_summary.csv"),
        "node_load_summary_path": experiment_path("reoptimization/imdb_reoptimization_node_load_summary.csv"),

        "node_load_paths": {
            "tuple_ilp":{
                "baseline": experiment_path("results/imdb/tuple_ilp/node_loads.csv"),
                "updates": experiment_path("reoptimization/imdb/tuple_ilp/node_loads_updated.csv")
            },
            "conflict_locality_ilp": {
                "baseline": experiment_path("results/imdb/conflict_locality_ilp/node_loads.csv"),
                "updates": experiment_path("reoptimization/imdb/conflict_locality_ilp/node_loads_updated.csv")
            }
        }
    }
}


DATASET = "imdb"
MODE = "prepare" # prepare or evaluate

def load_fragments(path, config):
    """
    Loads fragment data used for preparing the update
    """

    if not path.exists():
        raise FileNotFoundError(f"Fragment file not found: {path}")

    fragments_df = pd.read_csv(path)

    fragment_id_column = config["fragment_id"]
    item_ids_column = config["item_ids"]
    fragment_size_column = config["fragment_size"]

    required_columns = {fragment_id_column, item_ids_column, fragment_size_column, "scheme"}

    missing_columns = required_columns - set(fragments_df.columns)

    if missing_columns:
        raise ValueError(f"The missing columns in fragment file {path} are: {sorted(missing_columns)}")

    # reads fragment ids as string
    fragments_df[fragment_id_column] = fragments_df[fragment_id_column].astype("string")

    return fragments_df

def parse_item_ids(value):
    """
    Converts a comma-separated item-ID field into a list.
    """

    if pd.isna(value) or str(value).strip() == "":
        return []

    return [item_id.strip() for item_id in str(value).split(",") if item_id.strip()]

def validate_target_fragments(fragments_df, target_fragments, config):
    """
    Verifies that exactly one target fragment is selected per scheme.
    """

    fragment_id_column = config["fragment_id"]
    expected_schemes = set(config["expected_schemes"])
    target_fragments = list(dict.fromkeys(target_fragments))

    existing_fragments = set(fragments_df[fragment_id_column])
    target_fragment_set = set(target_fragments)

    missing_fragments = target_fragment_set - existing_fragments

    if missing_fragments:
        raise ValueError(f"Following fragments were not found: {sorted(missing_fragments)}")

    target_rows = fragments_df[fragments_df[fragment_id_column].isin(target_fragment_set)]

    actual_schemes = set(target_rows["scheme"])

    if actual_schemes != expected_schemes or len(target_rows) != len(expected_schemes):
        raise ValueError(
            f"The new item must be assigned to exactly one fragment of every scheme. Expected schemes: {sorted(expected_schemes)}, "
            f"Found schemes: {sorted(actual_schemes)}."
        )

    return target_fragments

def choose_target_fragments(fragments_df, config):
    """
    Selects update targets.

    For IMDb, a deterministic combination with no previous pairwise overlap is
    preferred.
    If the baseline conflict-locality assignment is available, a combination
    containing a colocated pair is preferred because the new conflicts force
    a placement change.
    """

    configured_targets = config.get("target_fragments", [])

    # Tries fragment combination specified in configuration, in case dataset does not have conflict-oriented target selections.
    if not config.get("prefer_new_conflicts", False):
        try:
            return validate_target_fragments(fragments_df, configured_targets, config)
        except ValueError:
            item_memberships = {}

            # Reconstructs fragment combination of every existing item
            for _, row in fragments_df.iterrows():
                for item_id in parse_item_ids(row[config["item_ids"]]):
                    # maps item id to scheme: fragment_id, thereby collecting item memberships for every scheme
                    item_memberships.setdefault(item_id, {})[row["scheme"]] = row[config["fragment_id"]]

            # Retains combinations containing one fragment from every expected scheme
            candidates = [[memberships[scheme] for scheme in config["expected_schemes"]]
                          for memberships in item_memberships.values()
                          if all(scheme in memberships for scheme in config["expected_schemes"])]

            if not candidates:
                raise ValueError("No valid existing fragment combination was found.")

            fallback = sorted(candidates)[0]
            print(f"Configured MeSH targets are unavailable in this instance; using {fallback}.")
            return validate_target_fragments(fragments_df, fallback, config)

    fragment_id_column = config["fragment_id"]
    item_ids_column = config["item_ids"]
    expected_schemes = config["expected_schemes"]

    # Stores memberships as sets for pairwise overlap checks
    fragment_sets = {row[fragment_id_column]: set(parse_item_ids(row[item_ids_column])) for _, row in fragments_df.iterrows()}

    # builds dictionary that maps from every scheme to the fragment_ids contained in scheme
    fragments_by_scheme = {scheme: sorted(fragments_df.loc[fragments_df["scheme"] == scheme, fragment_id_column].astype(str))
                           for scheme in expected_schemes}

    # loads baseline placement so that colocated fragment pairs can be preferred as targets for new conflicts
    baseline_path = config["assignment_paths"]["conflict_locality_ilp"]["baseline"]
    baseline_nodes = {}

    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        baseline_nodes = dict(zip(baseline_df["fragment_id"].astype(str), baseline_df["node_id"].astype(str)))

    best_candidate = None
    best_score = None

    # examines candidate combinations containing one fragment from each scheme.
    for candidate in product(*(fragments_by_scheme[scheme] for scheme in expected_schemes)):
        # Skips combinations from fragments whose overlap is already in the baseline assignment.
        if any(fragment_sets[fragment_i] & fragment_sets[fragment_j] for fragment_i, fragment_j in combinations(candidate, 2)):
            continue

        candidate_nodes = [baseline_nodes.get(fragment_id) for fragment_id in candidate]
        known_nodes = [node_id for node_id in candidate_nodes if node_id]

        # calculates how many candidate fragments are on the same baseline node.
        # Meaning it calculates how redundant fragments are copied
        colocation = len(known_nodes) - len(set(known_nodes)) if known_nodes else 0

        # calculates negative sum of fragment weight. Therefore smaller fragments are prefered later on.
        size_score = -sum(len(fragment_sets[fragment_id]) for fragment_id in candidate)

        # fragment id tuple provides tie breaker, by comparing tuples lexicographically.
        # In case of a tie breaker the fragnet id tuple id decides.
        score = (colocation, size_score, tuple(candidate))

        # calculates best candidate
        if best_score is None or score > best_score:
            best_candidate = list(candidate)
            best_score = score

    if best_candidate is None:
        print("No previously non-overlapping target combination was found; using the configured target fragments.")

        best_candidate = configured_targets

    return validate_target_fragments(fragments_df, best_candidate, config)

def create_new_item_ids(config):
    """
    Creates deterministic IDs for the synthetic update batch.
    """

    return [f"{config['new_item_prefix']}{index:04d}" for index in range(1, config["insert_count"] + 1)]

def apply_inserts(fragments_df, new_item_ids, target_fragments, config):
    """
    Adds all synthetic items to one fragment of each fragmentation scheme.
    """

    fragment_id_column = config["fragment_id"]
    item_ids_column = config["item_ids"]
    fragment_size_column = config["fragment_size"]

    # works on a copy to preserve baseline fragment
    update_df = fragments_df.copy()

    target_fragments = validate_target_fragments(fragments_df, target_fragments, config)

    # Restricts insertion batch to selected target_fragments
    mask = update_df[fragment_id_column].isin(target_fragments)

    def append_new_items(value):
        """
        Adds synthetic items without creating duplicate memberships
        """

        curr_item_ids = parse_item_ids(value)
        existing_item_ids = set(curr_item_ids)

        curr_item_ids.extend(item_id for item_id in new_item_ids if item_id not in existing_item_ids)

        return ",".join(curr_item_ids)

    # adds new item to selected fragments
    # Applies the function to selected rows and writes updated values back
    update_df.loc[mask, item_ids_column] = (update_df.loc[mask, item_ids_column].apply(append_new_items))

    # Recalculates the fragment sizes after an insert
    update_df.loc[mask, fragment_size_column] = (update_df.loc[mask, item_ids_column]
                                                 .apply(lambda value: len(set(parse_item_ids(value)))))

    return update_df

def create_update_items(items_df, new_item_ids, target_fragments, config):
    """
    Adds synthetic item rows used by node creation and recovery.
    """

    item_id_column = config["item_id"]
    item_name_column = config["item_name"]
    existing_item_ids = set(items_df[item_id_column].astype(str))

    # prevents item ids from rewriting already existing items
    collisions = existing_item_ids & set(new_item_ids)

    if collisions:
        raise ValueError(f"Item ids already exist: {sorted(collisions)[:10]}")

    rows = []

    for index, item_id in enumerate(new_item_ids, start=1):
        item_name = f"{config['new_item_name_prefix']} {index}"

        metadata_json = json.dumps({"synthetic_reoptimization_item": True,
                                    "target_fragments": target_fragments},
                                    ensure_ascii=False,
                                    separators=(",", ":"))

        item_size_bytes = len((item_id + item_name + metadata_json).encode("utf-8"))

        rows.append({item_id_column: item_id,
                     item_name_column: item_name,
                     "metadata_json": metadata_json,
                     "item_size_bytes": item_size_bytes})

    return pd.concat([items_df, pd.DataFrame(rows)], ignore_index=True, sort=False)


def save(updated_df, path):
    """
    Saves updated dataframe to a CSV
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    updated_df.to_csv(path, index=False)

    print(f"Updated fragments saved to: {path}")


def load_assignment(path):
    """
    Loads baseline or updated fragment assignment
    """

    if not path.exists():
        raise FileNotFoundError(f"Assignment file not found: {path}")

    assignment_df = pd.read_csv(path)

    required_columns = {"fragment_id", "node_id"}

    missing_columns = (required_columns - set(assignment_df.columns))

    if missing_columns:
        raise ValueError(f"Following columns are missing from the assignment file {path}: "
                         f"{sorted(missing_columns)}")

    assignment_df["fragment_id"] = (assignment_df["fragment_id"].astype("string"))

    # checks that every fragment must have exactly one assignment
    duplicate_fragments = assignment_df["fragment_id"].duplicated(keep=False)

    if duplicate_fragments.any():

        raise ValueError(f"There are duplicate fragments in assignment file {path}.")

    return assignment_df

def compare(placement_type, baseline, updated):
    """
    Compares pairwise relationships before and after reoptimization for a placement type.
    """

    # Maps every fragment to its baseline node
    baseline_nodes = dict(zip(baseline["fragment_id"], baseline["node_id"]))

    # Maps every fragment to its updated node
    updated_nodes = dict(zip(updated["fragment_id"], updated["node_id"]))

    baseline_fragments_ids = set(baseline_nodes)

    updated_fragments_ids = set(updated_nodes)

    # insertion changes fragment contents, but does not create or remove fragments.
    # Therefore fragment ids need to be the same:
    if baseline_fragments_ids != updated_fragments_ids:
        missing_updated = sorted(baseline_fragments_ids - updated_fragments_ids)
        missing_baseline = sorted(updated_fragments_ids - baseline_fragments_ids)

        raise ValueError(f"Baseline and updated assignments contain different fragments:"
                         f"Placement Type: {placement_type} "
                         f"Missing in updated assignment: {missing_updated} "
                         f"Missing in baseline assignment: {missing_baseline}")

    fragment_ids = sorted(baseline_fragments_ids)

    comparison_rows = []

    # compares colocation relationships instead of the node ids because equivalent solver situations
    # may use different node labels
    for fragment_i, fragment_j in combinations(fragment_ids, 2):
        same_before_node = baseline_nodes[fragment_i] == baseline_nodes[fragment_j]
        same_after_node = updated_nodes[fragment_i] == updated_nodes[fragment_j]

        # calculates relationship changes
        locality_relation_changed = (same_before_node != same_after_node)

        comparison_rows.append({
            "placement_type": placement_type,
            "fragment_i": fragment_i,
            "fragment_j": fragment_j,
            "same_before_node": same_before_node,
            "same_after_node": same_after_node,
            "locality_relation_changed": locality_relation_changed,
            })

    comparison_df = pd.DataFrame(comparison_rows)
    return comparison_df


def compare_assignments(config):
    """
    Compares baseline and updated assignments for all configured placement types.
    """

    comparison = []

    for placement_type, paths in config["assignment_paths"].items():
        missing_paths = [path for path in paths.values() if not path.exists()]

        if missing_paths:
            print(f"Skipping {placement_type} assignment comparison as following files are missing: "
                  f"{missing_paths}")
            continue

        baseline_df = load_assignment(paths["baseline"])

        updated_df = load_assignment(paths["updates"])

        comparison_df = compare(placement_type, baseline_df, updated_df)

        comparison.append(comparison_df)

    if not comparison:
        raise FileNotFoundError("No placement has both baseline and updated assignments.")
    
    # Combines results from all placement types
    complete_comparison = pd.concat(comparison, ignore_index=True)

    # Selects pairs whose relationship changed
    changed = complete_comparison[complete_comparison["locality_relation_changed"]].copy()

    total_pairs = complete_comparison.groupby("placement_type").size()

    amount_changed = changed.groupby("placement_type").size().reindex(total_pairs.index, fill_value=0)

    change_ratio = amount_changed/total_pairs

    summary = pd.DataFrame({
        "total_pairs": total_pairs,
        "changed_pairs": amount_changed,
        "unchanged_pairs": total_pairs - amount_changed,
        "change_ratio": change_ratio
    }).reset_index()

    return complete_comparison, changed, summary

def load_node_loads(path):
    """
    Loads node-load result file
    """
    if not path.exists():
        raise FileNotFoundError(f"File for node load not found: {path}")

    node_load = pd.read_csv(path)

    required_columns = {"used", "node_load", "remaining_capacity"}

    missing_columns = required_columns - set(node_load.columns)

    if missing_columns:
        raise ValueError(f"Following columns are missing from the node-load file {path}: "
                         f"{sorted(missing_columns)}")

    return node_load

def summarize_node_loads(placement_type, mode, node_loads):
    """
    Calculates node-load metrics for a placement type and an optimization mode
    """

    if mode not in {"baseline", "updates"}:
        raise ValueError(f"Unknown mode: {mode}")

    # statistics are calculated only for nodes that are used by the placement
    # excludes unused nodes from the load distribution statistics
    used_nodes = node_loads[node_loads["used"] == 1].copy()

    if used_nodes.empty:
        raise ValueError(f"There are no nodes used for {placement_type} in mode {mode}.")

    return {
        "placement_type": placement_type,
        "mode": mode,
        "used_nodes_amount": len(used_nodes),
        "total_load": used_nodes["node_load"].sum(),
        "min_node_load": used_nodes["node_load"].min(),
        "max_node_load": used_nodes["node_load"].max(),
        "avg_node_load": used_nodes["node_load"].mean(),
        "node_load_standard_deviation": used_nodes["node_load"].std(ddof=0),
        "total_remaining_capacity": used_nodes["remaining_capacity"].sum()
    }

def compare_loads(config):
    """
    Creates summaries for all placement types and optimization modes
    """

    summary_rows = []

    for placement_type, paths in config["node_load_paths"].items():
        missing_paths = [path for path in paths.values() if not path.exists()]

        if missing_paths:
            print(f"Skipping {placement_type} node-load comparison because there are missing files: "
                  f"{missing_paths}")
            continue


        for mode, path in paths.items():
            node_loads_df = load_node_loads(path)
            summary_row = summarize_node_loads(placement_type, mode, node_loads_df)
            summary_rows.append(summary_row)

    if not summary_rows:
        raise FileNotFoundError(f"There is no placement that has node-load files for both modes.")
    
    node_load_summary = pd.DataFrame(summary_rows)

    return node_load_summary

def save_assignment_comparison(complete_comparison, changed, summary, config):
    """
    Saves complete, changed, and summarized assignment comparison results.
    """
    comparison_path = config["comparison_path"]
    changed_path = config["changed_comparison_path"]
    summary_path = config["summary_comparison_path"]

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    complete_comparison.to_csv(comparison_path, index=False)
    changed.to_csv(changed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Complete assignment comparison saved to: {comparison_path}")
    print(f"Changed colocation relationships saved to: {changed_path}")
    print(f"Reoptimization summary saved to: {summary_path}")

def save_node_load_summary(node_load_summary, config):
    """
    Saves the combined node-load summary.
    """

    node_load_path = config["node_load_summary_path"]

    node_load_path.parent.mkdir(parents=True, exist_ok=True,)

    node_load_summary.to_csv(node_load_path, index=False,)

    print(f"Node-load summary saved to: {node_load_path}")

    return node_load_path


def process_prepare_reoptimization(config):
    """
    Creates and saves updated fragment state.
    """
    fragments_df = load_fragments(path=config["fragments_path"], config=config)

    items_df = pd.read_csv(config["items_path"])

    target_fragments = choose_target_fragments(fragments_df, config)

    new_item_ids = create_new_item_ids(config)

    update_df = apply_inserts(fragments_df=fragments_df, new_item_ids=new_item_ids,
                              target_fragments=target_fragments, config=config)

    updated_items_df = create_update_items(items_df, new_item_ids, target_fragments, config)

    save(update_df, config["updated_fragments_path"])

    save(updated_items_df, config["updated_items_path"])

    return update_df, updated_items_df, target_fragments, new_item_ids

def process_evaluate_reoptimization(config):
    """
    Evaluates assignment changes and node loads after the reoptimization
    """
    complete_comparison, changed, summary = compare_assignments(config)

    save_assignment_comparison(complete_comparison, changed, summary, config)

    node_load_summary = compare_loads(config)

    save_node_load_summary(node_load_summary, config)

    return complete_comparison, changed, summary, node_load_summary


def main():
    if DATASET not in CONFIGS:
        raise ValueError(f"Unknown dataset: {DATASET}")

    config = CONFIGS[DATASET]

    if MODE == "prepare":
        update_df, updated_items_df, target_fragments, new_item_ids = process_prepare_reoptimization(config)

        columns = [config["fragment_id"], config["fragment_size"], config["item_ids"]]

        target_rows = update_df[update_df[config["fragment_id"]].isin(target_fragments)]

        print(f"Inserted items: {len(new_item_ids)}")
        print(f"Target fragments: {target_fragments}")

        print(target_rows[columns])

    elif MODE == "evaluate":
        complete_comparison, changed, summary, node_load_summary = process_evaluate_reoptimization(config)

        print("\nReoptimization comparison:")
        print(summary)

        print("\nNode-load comparison")
        print(node_load_summary)

        print("\nNumber of changed colocation relationships:")
        print(len(changed))

    else:
        raise ValueError(f"Unknown mode: {MODE}")

if __name__ == "__main__":
    main()