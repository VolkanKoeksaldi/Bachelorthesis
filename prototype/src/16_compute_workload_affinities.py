from pathlib import Path
from itertools import combinations
import json
import pandas as pd

AFFINITY_CONFIGS = {
    "mesh": {
        "workload_path": Path("prototype/output/workloads/mesh_workload.json"),
        "output_directory": Path("prototype/output/workload_affinities"),
        "overlap_path": Path("prototype/output/processed/mesh_overlaps_sample.csv"),

        "overlap_fragment_1": "fragment_1",
        "overlap_fragment_2": "fragment_2"
    },

    "imdb": {
        "workload_path": Path("prototype/output/workloads/imdb_workload.json"),
        "output_directory": Path("prototype/output/workload_affinities"),
        "overlap_path": Path("prototype/output/processed/imdb_overlaps.csv"),

        "overlap_fragment_1": "fragment_1",
        "overlap_fragment_2": "fragment_2"
    }
}

DATASET = "imdb"

def load_workload(path):
    """
    Zum Laden der Workload Ergebnisse.
    """

    if not path.exists():
        raise FileNotFoundError(f"Die erwartete Datei wurde nicht gefunden {path}.")

    with path.open("r", encoding="utf-8") as file:
        workload = json.load(file)

    return workload

def normalize_pair(fragment_i, fragment_j):
    return tuple(sorted((fragment_i, fragment_j)))

def compute_affinities(workload):
    """
    Berechnet die affinities für jedes Fragmentpaar anhand der FRAGMENT_SELECT-Operationen.
    """

    affinities = {}

    for operation in workload:
        # Hierbei werden nur FRAGMENT_SELECT-Operationen berücksichtigt
        if operation["operation"] != "FRAGMENT_SELECT":
            continue

        fragment_ids = operation["fragment_ids"]

        # Um doppelte IDs innerhalb einer Query zu entfernen
        fragment_ids = sorted(set(fragment_ids))

        for fragment_i, fragment_j in combinations(fragment_ids, 2):
            couple = normalize_pair(fragment_i, fragment_j)
            if couple not in affinities:
                affinities[couple] = 0

            affinities[couple] += 1

    return affinities

def create_affinity_df(affinities):
    """
    Zum erstellen einer Tabelle mit den berechneten Affinitäten
    """

    rows = []
    for couple, affinity in affinities.items():
        fragment_i = couple[0]
        fragment_j = couple[1]

        rows.append({"fragment_i": fragment_i, "fragment_j": fragment_j, "affinity": affinity})

    affinity_df = pd.DataFrame(rows)

    if affinity_df.empty:
        raise ValueError(f"Die Affinity Tabelle ist leer.")
    else:
        affinity_df = affinity_df.sort_values(by="affinity", ascending=False)

    return affinity_df


def save(affinity, dataset, config):
    """
    Zum Speichern der Affinitäten
    """

    output_directory = config["output_directory"]

    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / f"{dataset}_workload_affinities.csv"

    affinity.to_csv(output_path, index=False)

    print(f"Die Affinitäten wurden gespeichert unter {output_path}")

    return output_path

def compare_affinity_conflict(affinity, config):
    """
    Vergleicht Affinität mit Overlap-Konfliktpaaren
    """

    overlap_path = config["overlap_path"]
    fragment_1 = config["overlap_fragment_1"]
    fragment_2 = config["overlap_fragment_2"]

    if not overlap_path.exists():
        raise FileNotFoundError(f"Overlap Datei wurde nicht gefunden {overlap_path}")

    affinity_pairs = {
        normalize_pair(row["fragment_i"], row["fragment_j"])
        for _, row in affinity.iterrows()
    }

    overlap_df = pd.read_csv(overlap_path)
    conflict_pairs = {
        normalize_pair(row[fragment_1], row[fragment_2])
        for _, row in overlap_df.iterrows()
    }

    affinity_conflicts = affinity_pairs & conflict_pairs
    non_conflict_affinities = affinity_pairs - conflict_pairs

    comparison = {
        "amount_affinity_pairs": len(affinity_pairs),
        "amount_conflict_affinities": len(affinity_conflicts),
        "amount_non_conflict_affinities": len(non_conflict_affinities)
    }

    return comparison, non_conflict_affinities


def process_compute_workload_affinities(dataset, config):

    workload = load_workload(config["workload_path"])

    affinities = compute_affinities(workload)

    affinity_df = create_affinity_df(affinities)

    save(affinity_df, dataset, config)

    comparison, non_conflict_affinities = compare_affinity_conflict(affinity=affinity_df, config=config)

    return affinity_df, comparison, non_conflict_affinities


def main():
    if DATASET not in AFFINITY_CONFIGS:
        raise ValueError(f"Unbekannter Datensatz in der Affinitäts Konfiguration {DATASET}")

    config = AFFINITY_CONFIGS[DATASET]

    affinity_df, comparison, non_conflict_affinities = process_compute_workload_affinities(dataset=DATASET, config=config)

    print("Die Anzahl unterschiedlicher Fragmentpaare beträgt: ", len(affinity_df))

    print("\nDie Top 5 Fragmentpaare mit der höchsten Affinität sind hierbei: ")
    print(affinity_df.head(5))

    print("\nAffinitäts-Paare insgesamt:", comparison["amount_affinity_pairs"])

    print("Konfliktpaare:", comparison["amount_conflict_affinities"])

    print("Nicht-Konfliktpaare:", comparison["amount_non_conflict_affinities"])
    
    for pair in sorted(non_conflict_affinities):
        print("\n Paare:", pair)


if __name__ == "__main__":
    main()        
