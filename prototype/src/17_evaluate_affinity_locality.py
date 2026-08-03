from pathlib import Path
import pandas as pd

AFFINITY_LOCALITY_CONFIGS = {
    "mesh": {
        "affinity_path": Path("prototype/output/workload_affinities/mesh_workload_affinities.csv"),
        "output_directory": Path("prototype/output/affinity_evaluation"),
        "assignment_paths": {
            "round_robin" : Path("prototype/output/processed/mesh_fragment_assignment_round_robin.csv"),
            "tuple_ilp": Path("prototype/output/processed/mesh_fragment_assignment_tuple_ilp.csv"),
            "conflict_locality_ilp": Path("prototype/output/processed/mesh_fragment_assignment_conflict_locality_ilp.csv"),
            "conflict_locality_ilp_updated": Path("prototype/output/reoptimization/" \
            "mesh_fragment_assignment_conflict_locality_ilp_updated.csv")
        }
    },

    "imdb": {
        "affinity_path": Path("prototype/output/workload_affinities/imdb_workload_affinities.csv"),
        "output_directory": Path("prototype/output/affinity_evaluation"),
        "assignment_paths": {
            "round_robin" : Path("prototype/output/processed/imdb_fragment_assignment_round_robin.csv"),
            "tuple_ilp": Path("prototype/output/processed/imdb_fragment_assignment_tuple_ilp.csv"),
            "conflict_locality_ilp": Path("prototype/output/processed/imdb_fragment_assignment_conflict_locality_ilp.csv"),
            "conflict_locality_ilp_updated": Path("prototype/output/reoptimization/" \
            "imdb_fragment_assignment_conflict_locality_ilp_updated.csv")
        }
    }
}

DATASET = "imdb"

def load_affinities(path):
    """
    Lädt die berechneten Fragmentaffinitäten
    """

    if not path.exists():
        raise FileNotFoundError(f"Datei {path} wurde nicht gefunden.")

    affinity_df = pd.read_csv(path)

    return affinity_df


def load_assignment(assignment_path):
    """
    Lädt die Assignments eines Placement-Typs
    """

    if not assignment_path.exists():
        raise FileNotFoundError(f"Datei {assignment_path} wurde nicht gefunden.")

    assignment_df = pd.read_csv(assignment_path)
    return assignment_df


def evaluate(placement_type, assignment_df, affinity_df):
    """
    Prüft welche Fragmentpaare, die affin sind, bei einem Placement-Typ auf derselben Node assigned wurde.
    """

    fragment_node = dict(zip(assignment_df["fragment_id"], assignment_df["node_id"]))

    details = []

    for row in affinity_df.itertuples(index=False):
        fragment_i = row.fragment_i
        fragment_j = row.fragment_j
        affinity = row.affinity

        if fragment_i not in fragment_node:
            raise ValueError(f"{fragment_i} ist nicht assigned worden.")

        if fragment_j not in fragment_node:
            raise ValueError(f"{fragment_j} ist nicht assigned worden.")

        node_i = fragment_node[fragment_i]
        node_j = fragment_node[fragment_j]

        if node_i == node_j:
            same_node = True
        else:
            same_node = False

        details.append({
            "placement_type": placement_type,
            "fragment_i": fragment_i,
            "fragment_j": fragment_j,
            "affinity": affinity,
            "node_i": node_i,
            "node_j": node_j,
            "same_node": same_node
        })

    details_df = pd.DataFrame(details)

    # Gesamte Affinität aller Paare
    total_affinity = details_df["affinity"].sum()

    # Affinität der Zeilen, wo same_node = True
    colocated_affinity = details_df.loc[details_df["same_node"]==True, "affinity"].sum()

    # Getrennte Affinitäten
    separated_affinity = details_df.loc[details_df["same_node"]==False, "affinity"].sum()

    # Locality Ratio (Anteil gemeinsam platzierter Affinität)
    if total_affinity == 0:
        locality_ratio = 0
    else:
        locality_ratio = colocated_affinity / total_affinity

    summary = {
        "placement_type": placement_type,
        "number_affinity_pairs": len(details_df),
        "same_node_pairs": int(details_df["same_node"].sum()),
        "total_affinity": total_affinity,
        "colocated_affinity": colocated_affinity,
        "separated_affinity": separated_affinity,
        "locality_ratio": locality_ratio
    }

    return summary, details_df

def save(summary, details, dataset, config):
    """
    Speichert die jeweilige Summary und Detail Ergebnisse.
    """

    output_directory = config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    summary_path = output_directory/f"{dataset}_affinity_locality_summary.csv"
    details_path = output_directory/f"{dataset}_affinity_locality_details.csv"

    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)

    print(f"Die Affinity Summary ist gespeichert unter: {summary_path}")
    print(f"Die Affinity Details sind gespeichert unter: {details_path}")

    return summary_path, details_path

def process_evaluate_affinity_locality(dataset, config):
    affinity = load_affinities(config["affinity_path"])

    summary_array = []
    detail_array = []

    for placement_type, path in config["assignment_paths"].items():
        assignment_df = load_assignment(path)

        summary, detail = evaluate(placement_type=placement_type, assignment_df=assignment_df, affinity_df=affinity)

        summary_array.append(summary)
        detail_array.append(detail)

    summary_df = pd.DataFrame(summary_array)
    detail_df = pd.concat(detail_array)

    save(summary=summary_df, details=detail_df, dataset=dataset, config=config)

    return summary_df, detail_df


def main():

    if DATASET not in AFFINITY_LOCALITY_CONFIGS:
        raise ValueError(f"Unbekannter Datensatz {DATASET}")

    config = AFFINITY_LOCALITY_CONFIGS[DATASET]

    summary_df, detail_df = process_evaluate_affinity_locality(dataset=DATASET, config=config)

    print("\nSummary:")
    print(summary_df)

    print("\nAnzahl ausgewerteter Details:", len(detail_df))

if __name__ == "__main__":
    main()