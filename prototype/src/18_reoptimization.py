from pathlib import Path
import pandas as pd
from itertools import combinations

CONFIGS = {
    "mesh": {
        "fragments_path": Path("prototype/output/processed/mesh_fragments_sample.csv"),
        "updated_fragments_path": Path("prototype/output/reoptimization/mesh_fragments_sample_updates.csv"),

        "fragment_id": "fragment_id",
        "item_ids": "descriptor_ids",
        "fragment_size": "fragment_size",

        "new_item_id": "D_RE_001",
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
                "baseline": Path("prototype/output/processed/mesh_fragment_assignment_tuple_ilp.csv"),
                "updates": Path("prototype/output/reoptimization/mesh_fragment_assignment_tuple_ilp_updated.csv")
            },

            "conflict_locality_ilp":{
                "baseline": Path("prototype/output/processed/mesh_fragment_assignment_conflict_locality_ilp.csv"),
                "updates": Path("prototype/output/reoptimization/mesh_fragment_assignment_conflict_locality_ilp_updated.csv")
            }
        },

        "comparison_path": Path("prototype/output/reoptimization/mesh_reoptimization_compared.csv"),
        "changed_comparison_path": Path("prototype/output/reoptimization/mesh_reoptimization_changed.csv"),
        "summary_comparison_path": Path("prototype/output/reoptimization/mesh_reoptimization_summary.csv"),
        "node_load_summary_path": Path("prototype/output/reoptimization/mesh_reoptimization_node_load_summary.csv"),

        "node_load_paths": {
            "tuple_ilp":{
                "baseline": Path("prototype/output/results/mesh/tuple_ilp/node_loads.csv"),
                "updates": Path("prototype/output/reoptimization/mesh/tuple_ilp/node_loads_updated.csv")
            },
            "conflict_locality_ilp": {
                "baseline": Path("prototype/output/results/mesh/conflict_locality_ilp/node_loads.csv"),
                "updates": Path("prototype/output/reoptimization/mesh/conflict_locality_ilp/node_loads_updated.csv")
            }
        }
    },

    "imdb": {
        "fragments_path": Path("prototype/output/processed/imdb_fragments.csv"),
        "updated_fragments_path": Path("prototype/output/reoptimization/imdb_fragments_updates.csv"),

        "fragment_id": "fragment_id",
        "item_ids": "title_ids",
        "fragment_size": "fragment_size",

        "new_item_id": "tt_RE_001",
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
                "baseline": Path("prototype/output/processed/imdb_fragment_assignment_tuple_ilp.csv"),
                "updates": Path("prototype/output/reoptimization/imdb_fragment_assignment_tuple_ilp_updated.csv")
            },

            "conflict_locality_ilp":{
                "baseline": Path("prototype/output/processed/imdb_fragment_assignment_conflict_locality_ilp.csv"),
                "updates": Path("prototype/output/reoptimization/imdb_fragment_assignment_conflict_locality_ilp_updated.csv")
            }
        },

        "comparison_path": Path("prototype/output/reoptimization/imdb_reoptimization_compared.csv"),
        "changed_comparison_path": Path("prototype/output/reoptimization/imdb_reoptimization_changed.csv"),
        "summary_comparison_path": Path("prototype/output/reoptimization/imdb_reoptimization_summary.csv"),
        "node_load_summary_path": Path("prototype/output/reoptimization/imdb_reoptimization_node_load_summary.csv"),

        "node_load_paths": {
            "tuple_ilp":{
                "baseline": Path("prototype/output/results/imdb/tuple_ilp/node_loads.csv"),
                "updates": Path("prototype/output/reoptimization/imdb/tuple_ilp/node_loads_updated.csv")
            },
            "conflict_locality_ilp": {
                "baseline": Path("prototype/output/results/imdb/conflict_locality_ilp/node_loads.csv"),
                "updates": Path("prototype/output/reoptimization/imdb/conflict_locality_ilp/node_loads_updated.csv")
            }
        }
    }
}

DATASET = "imdb"
MODE = "prepare" # prepare oder evaluate

def load_fragments(path):
    """
    Lädt die Fragmente.
    """

    if not path.exists():
        raise FileNotFoundError(f"Datei {path} wurde nicht gefunden.")

    fragments_df = pd.read_csv(path)

    return fragments_df


def apply_inserts(
    fragments_df,
    new_item_id,
    target_fragments,
    config
):
    """
    Fügt ein neues Item genau einem Fragment pro Schema hinzu.
    """

    fragment_id_column = config["fragment_id"]
    item_ids_column = config["item_ids"]
    fragment_size_column = config["fragment_size"]
    expected_schemes = set(config["expected_schemes"])

    update_df = fragments_df.copy()

    target_fragments = set(target_fragments)
    existing_fragments = set(update_df[fragment_id_column])

    missing_fragments = target_fragments - existing_fragments

    if missing_fragments:
        raise ValueError(
            f"Folgende Fragmente wurden nicht gefunden: "
            f"{sorted(missing_fragments)}"
        )

    target_rows = update_df[
        update_df[fragment_id_column].isin(target_fragments)
    ]

    actual_schemes = set(target_rows["scheme"])

    if (
        actual_schemes != expected_schemes
        or len(target_fragments) != len(expected_schemes)
    ):
        raise ValueError(
            "Das neue Item muss genau einem Fragment jedes Schemas "
            f"zugeordnet werden. Erwartet: {sorted(expected_schemes)}, "
            f"gefunden: {sorted(actual_schemes)}."
        )

    mask = update_df[fragment_id_column].isin(target_fragments)

    def append_new_item(value):
        if pd.isna(value) or value == "":
            current_item_ids = []
        else:
            current_item_ids = [
                item_id.strip()
                for item_id in str(value).split(",")
                if item_id.strip()
            ]

        if new_item_id not in current_item_ids:
            current_item_ids.append(new_item_id)

        return ",".join(current_item_ids)

    update_df.loc[mask, item_ids_column] = (
        update_df.loc[mask, item_ids_column]
        .apply(append_new_item)
    )

    update_df.loc[mask, fragment_size_column] = (
        update_df.loc[mask, item_ids_column]
        .apply(
            lambda value: len([
                item_id
                for item_id in str(value).split(",")
                if item_id.strip()
            ])
        )
    )

    return update_df


def save(updated_df, path):
    """
    Speichert update Fragmentzustand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    updated_df.to_csv(path, index=False)

    print(f"Aktualisierte Fragmente sind hier gespeichert {path}")


def load_assignment(path):
    """
    Lädt ein Fragment Assignment
    """

    if not path.exists():
        raise FileNotFoundError(f"Die Datei im Pfad {path} wurde nicht gefunden.")

    assignment_df = pd.read_csv(path)

    return assignment_df

def compare(placement_type, baseline, updated):
    """
    Zum Vergleichen der Werte zwischen der Baseline und dem reoptimierten Assignments eines Placement-Typs.
    """
    
    baseline_nodes = dict(zip(baseline["fragment_id"], baseline["node_id"]))
    updated_nodes = dict(zip(updated["fragment_id"], updated["node_id"]))

    fragment_ids = sorted(set(baseline_nodes.keys()) & set(updated_nodes.keys()))

    comparison_rows = []

    for fragment_i, fragment_j in combinations(fragment_ids, 2):
        same_before_node = baseline_nodes[fragment_i] == baseline_nodes[fragment_j]
        same_after_node = updated_nodes[fragment_i] == updated_nodes[fragment_j]

        locality_relation_changed = same_before_node != same_after_node

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
    Vergleicht die Placements der Fragmenten vor und nach einer Neuoptimierung
    """

    comparison = []

    for placement_type, paths in config["assignment_paths"].items():
        baseline_df = load_assignment(paths["baseline"])

        updated_df = load_assignment(paths["updates"])

        comparison_df = compare(placement_type, baseline_df, updated_df)

        comparison.append(comparison_df)


    complete_comparison = pd.concat(comparison, ignore_index=True)
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
    Lädt die Node-Auslastungsdateien
    """
    if not path.exists():
        raise FileNotFoundError(f"Der Pfad {path} wurde nicht gefunden.")

    node_load = pd.read_csv(path)

    return node_load

def summarize_node_loads(placement_type, mode, node_loads):
    """
    Berechnet die Kennzahlen der Node Auslastungen
    """

    if mode not in {"baseline", "updates"}:
        raise ValueError(f"Falscher Modus: {mode}")

    used_nodes = node_loads[node_loads["used"] == 1].copy()

    if used_nodes.empty:
        raise ValueError(f"Es gibt keine verwendeten Nodes für {placement_type} im Modus {mode}.")

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
    Erstellt die Zusammenfassung für alle Node-Load Dateien auf ihren Placements und den Modes
    """

    summary_rows = []

    for placement_type, paths in config["node_load_paths"].items():
        for mode, path in paths.items():
            node_loads_df = load_node_loads(path)
            summary_row = summarize_node_loads(placement_type, mode, node_loads_df)
            summary_rows.append(summary_row)

    node_load_summary = pd.DataFrame(summary_rows)

    return node_load_summary

def save_assignment_comparison(complete_comparison, changed, summary, config):
    comparison_path = config["comparison_path"]
    changed_path = config["changed_comparison_path"]
    summary_path = config["summary_comparison_path"]

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    complete_comparison.to_csv(comparison_path, index=False)
    changed.to_csv(changed_path, index=False)
    summary.to_csv(summary_path, index=False)


def process_prepare_reoptimization(config):
    fragments_df = load_fragments(config["fragments_path"])

    update_df = apply_inserts(fragments_df=fragments_df,
                              new_item_id=config["new_item_id"],
                              target_fragments=config["target_fragments"],
                              config=config)

    save(update_df, config["updated_fragments_path"])

    return update_df

def process_evaluate_reoptimization(config):
    complete_comparison, changed, summary = compare_assignments(config)

    save_assignment_comparison(complete_comparison, changed, summary, config)

    node_load_summary = compare_loads(config)

    node_load_path = config["node_load_summary_path"]
    node_load_path.parent.mkdir(parents=True, exist_ok=True)

    node_load_summary.to_csv(node_load_path, index=False)

    return complete_comparison, changed, summary, node_load_summary


def main():
    if DATASET not in CONFIGS:
        raise ValueError(f"Unbekannter Datensatz {DATASET}")

    config = CONFIGS[DATASET]

    if MODE == "prepare":
        update_df = process_prepare_reoptimization(config)

        columns = [config["fragment_id"], config["fragment_size"], config["item_ids"]]

        target_rows = update_df[update_df[config["fragment_id"]].isin(config["target_fragments"])]

        print(target_rows[columns])

    elif MODE == "evaluate":
        complete_comparison, changed, summary, node_load_summary = process_evaluate_reoptimization(config)

        print("\nVergleich nach der Reoptimierung")
        print(summary)

        print("\nVergleich der Node Auslastungen")
        print(node_load_summary)

        print("\nAnzahl geänderter Paar Beziehungen:")
        print(len(changed))

    else:
        raise ValueError(f"Unbekannter Modus: {MODE}")


    
if __name__ == "__main__":
    main()