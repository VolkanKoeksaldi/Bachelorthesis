import pulp
import time
import pandas as pd
from pathlib import Path
import math
import highspy
from itertools import combinations
from placement_capacity import calculate_node_capacity

DATASETS = {
    "mesh": {
        "item_ids_column": "descriptor_ids",
        "affinity_path": Path("prototype/output/workload_affinities/mesh_workload_affinities.csv"),
        "num_nodes": 10,
        "node_capacity": None, # bedeutet automatische berechnung mit einem Puffer von "capacity_buffer"
        "capacity_buffer": 0.50,
        "replication_factor": 3,
        "capacity_reference_path": Path("prototype/output/processed/mesh_fragments_sample.csv"),


        "modes": {
            "baseline": {
                "fragments_path": Path("prototype/output/processed/mesh_fragments_sample.csv"),
                "overlaps_path": Path("prototype/output/processed/mesh_overlaps_sample.csv"),
                "assignment_output_path": Path("prototype/output/processed/mesh_fragment_assignment_conflict_locality_ilp.csv"),
                "loads_output_path": Path("prototype/output/results/mesh/conflict_locality_ilp/node_loads.csv"),
                "solver_result_path": Path("prototype/output/processed/mesh/mesh_baseline_solver_result_conflict_locality_ilp.csv")

            },

            "updates": {
                "fragments_path": Path("prototype/output/reoptimization/mesh_fragments_sample_updates.csv"),
                "overlaps_path": Path("prototype/output/reoptimization/mesh_overlaps_sample_updates.csv"),
                "assignment_output_path": Path("prototype/output/reoptimization/mesh_fragment_assignment_conflict_locality_ilp_updated.csv"),
                "loads_output_path": Path("prototype/output/reoptimization/mesh/conflict_locality_ilp/node_loads_updated.csv"),
                "solver_result_path": Path("prototype/output/processed/mesh/mesh_updates_solver_result_conflict_locality_ilp.csv")
           
            }
        }
    },

    "imdb": {
        "item_ids_column": "title_ids",
        "affinity_path": Path("prototype/output/workload_affinities/imdb_workload_affinities.csv"),
        "num_nodes": 10,
        "node_capacity": None, # bedeutet automatische berechnung mit einem Puffer von "capacity_buffer"
        "capacity_buffer": 0.50,
        "replication_factor": 3,
        "capacity_reference_path": Path("prototype/output/processed/imdb_fragments.csv"),


        "modes": {
            "baseline": {
                "fragments_path": Path("prototype/output/processed/imdb_fragments.csv"),
                "overlaps_path": Path("prototype/output/processed/imdb_overlaps.csv"),
                "assignment_output_path": Path("prototype/output/processed/imdb_fragment_assignment_conflict_locality_ilp.csv"),
                "loads_output_path": Path("prototype/output/results/imdb/conflict_locality_ilp/node_loads.csv"),
                "solver_result_path": Path("prototype/output/processed/imdb/imdb_baseline_solver_result_conflict_locality_ilp.csv")
            },

            "updates": {
                "fragments_path": Path("prototype/output/reoptimization/imdb_fragments_updates.csv"),
                "overlaps_path": Path("prototype/output/reoptimization/imdb_overlaps_updates.csv"),
                "assignment_output_path": Path("prototype/output/reoptimization/imdb_fragment_assignment_conflict_locality_ilp_updated.csv"),
                "loads_output_path": Path("prototype/output/reoptimization/imdb/conflict_locality_ilp/node_loads_updated.csv"),
                "solver_result_path": Path("prototype/output/processed/imdb/imdb_updates_solver_result_conflict_locality_ilp.csv")           
            }
        }    }
}

MODE = "baseline"
DATASET = "mesh"

def parse_item_ids(item_ids_string):
    """
    Wandelt Descriptor-IDs aus der CSV wieder in ein Set um.

    Beispiel:
    "D000001,D000002,D000003"
    -> {"D000001", "D000002", "D000003"}
    """

    if pd.isna(item_ids_string) or item_ids_string == "":
        return set()

    return set(item_ids_string.split(","))

def load_fragments(fragments_path, item_ids_column):
    """
    Wird genutzt um die Fragmente aus der CSV-Datei zu laden.
    return:
        - fragments_df:
            das ist der ursprüngliche DataFrame mit einer zusätzlichen Set-Spalte
        - fragment_descriptor_ids:
            Dictionary von fragment_id auf das Set von Descriptor-IDs die darin enthalten sind
        - fragment_weights:
            Dictionary von fragment_id auf die Anzahl enthaltener Descriptoren
    """

    if not fragments_path.exists():
        raise FileNotFoundError(f"Fragmentdatei nicht gefunden in folgendem Pfad: {fragments_path}")
    
    # Fragment CSV wird eingelesen
    fragments_df = pd.read_csv(fragments_path)

    # in fragments_df wird eine Spalte erzeugt, die den Set an Descriptor_ids enthält
    fragments_df["item_id_set"] = (fragments_df[item_ids_column].apply(parse_item_ids))

    # Leere Fragmente werden hier entfernt.
    # Es wird also fragments_df überschrieben mit einer Copy dessen, wo leere Fragmente 
    # rausgefiltert werden.
    fragments_df = fragments_df[fragments_df["item_id_set"].map(len) > 0].copy()

    # Fragment Descriptor Menge:
    # für jede Zeile/Tuple wird das Dictionary gebaut von fragment_id: Descriptor_IDs Set
    fragment_item_ids = {row.fragment_id: row.item_id_set
                                for row in fragments_df.itertuples()}
    
    # Fragmentgewicht:
    # für jedes fragment_id wird jetzt die Gewichtung gemessen anhand der Anzahl der descriptor_ids 
    # die sie enthalten:
    fragment_weights = {fragment_id: len(item_ids)
                        for fragment_id, item_ids in fragment_item_ids.items()}

    if not fragments_df["fragment_id"].is_unique:
        raise ValueError("Die Fragmentdatei enthält doppelte fragment_id-Werte.")
    
    return fragments_df, fragment_item_ids, fragment_weights

def validate_conflicts(fragment_item_ids, conflict_pairs, num_nodes, replication_factor):
    item_fragments = {}


    
    for fragment_id, item_ids in fragment_item_ids.items():
        for item_id in item_ids:
            item_fragments.setdefault(item_id, set()).add(fragment_id)

    max_memberships = max(
        (len(fragments) for fragments in item_fragments.values()),
        default=0
    )

    invalid_memberships = {
        item_id: sorted(fragments)
        for item_id, fragments in item_fragments.items()
        if len(fragments) != replication_factor
    }

    if invalid_memberships:
        raise ValueError(
            f"{len(invalid_memberships)} Items besitzen nicht genau "
            f"{replication_factor} Fragmentmitgliedschaften. "
            f"Erste Verletzungen: "
            f"{list(invalid_memberships.items())[:10]}"
        )

    if max_memberships > num_nodes:
        raise ValueError(
            f"Conflict model infeasible: one item belongs to "
            f"{max_memberships} fragments, but only {num_nodes} nodes exist."
        )

    expected_pairs = set()

    for fragments in item_fragments.values():
        expected_pairs.update(combinations(sorted(fragments), 2))

    loaded_pairs = {
        tuple(sorted((fragment_i, fragment_j)))
        for fragment_i, fragment_j in conflict_pairs
        if fragment_i != fragment_j
    }

    missing_pairs = expected_pairs - loaded_pairs

    if missing_pairs:
        raise ValueError(
            f"Overlap file is incomplete: {len(missing_pairs)} "
            "conflict pairs are missing."
        )

def assignments(fragment_ids, fragment_weights, node_ids, x):
    """
    Auswertung welches Fragment auf welchem Node liegt.
    """
    assignment_rows = []

    for fragment_id in fragment_ids:
        assigned_nodes = [node_id
                          for node_id in node_ids
                          if pulp.value(x[fragment_id][node_id]) > 0.5
                          ]
        if len(assigned_nodes) != 1:
            raise RuntimeError(f"Fragment {fragment_id} wurde {len(assigned_nodes)} Nodes zugewiesen")
        
        assignment_rows.append({"fragment_id": fragment_id,
                                "node_id": assigned_nodes[0],
                                "fragment_weight": fragment_weights[fragment_id]})
        
    return pd.DataFrame(assignment_rows)

def compute_loads(fragment_ids, fragment_weights, node_ids, x, y, node_capacity):
    """
    
    """
    
    load_rows = []

    for node_id in node_ids:
        assigned_fragments = [fragment_id
                              for fragment_id in fragment_ids
                              if pulp.value(x[fragment_id][node_id]) > 0.5]
        
        node_load = sum(fragment_weights[fragment_id] for fragment_id in assigned_fragments)
        load_rows.append({
            "node_id": node_id,
            "used": int(pulp.value(y[node_id]) > 0.5),
            "number_of_fragments": len(assigned_fragments),
            "node_load": node_load,
            "node_capacity": node_capacity,
            "remaining_capacity": node_capacity - node_load
        })

    return pd.DataFrame(load_rows)

def load_overlap_pairs(overlaps_path):
    """
    Lädt alle Fragmentpaare mit Overlaps aus der CSV-Datei.
    """

    if not overlaps_path.exists():
        raise FileNotFoundError(f"Overlap-Datei nicht gefunden: {overlaps_path}")

    overlap_file = pd.read_csv(overlaps_path)

    overlap_file = overlap_file[overlap_file["overlap_size"] > 0].copy()

    overlap_pair = list(overlap_file[["fragment_1", "fragment_2"]].itertuples(index=False, name=None))
    
    return overlap_file, overlap_pair

def load_affinity(affinity_path):
    """
    Zum Laden der Affinity Workload Datei.
    """

    if not affinity_path.exists():
        raise FileNotFoundError(f"Die Affinity Datei wurde nicht gefunden auf dem Pfad {affinity_path}")
    
    affinity_df = pd.read_csv(affinity_path)

    affinities = {}

    for _, row in affinity_df.iterrows():
        fragment_i = row["fragment_i"]
        fragment_j = row["fragment_j"]
        affinity = float(row["affinity"])

        if fragment_i == fragment_j:
            continue

        pair = tuple(sorted((fragment_i, fragment_j)))

        affinities[pair] = (affinities.get(pair, 0) + affinity)

    return affinities

def process_conflict_locality_ilp(config, mode):
    if mode not in config["modes"]:
        raise ValueError(f"Unbekannter Modus: {mode}")

    mode_config = config["modes"][mode]

    fragments_path = mode_config["fragments_path"]
    overlaps_path = mode_config["overlaps_path"]
    output_assignment_path = mode_config["assignment_output_path"]
    output_loads_path = mode_config["loads_output_path"]

    for stale_output_path in (
        output_assignment_path,
        output_loads_path
    ):
        if stale_output_path.exists():
            stale_output_path.unlink()

    item_ids_column = config["item_ids_column"]
    affinity_path = config["affinity_path"]
    num_nodes = config["num_nodes"]
    node_capacity_config = config.get("node_capacity")
    capacity_buffer = config.get("capacity_buffer", 0.10)

    node_ids = [f"node_{i}" for i in range(1, num_nodes + 1)]

    fragments_df, fragment_item_ids, fragment_weights = load_fragments(fragments_path, item_ids_column)

    fragment_ids = list(fragment_item_ids.keys())

    overlaps_df, conflict_pairs = load_overlap_pairs(overlaps_path)

    validate_conflicts(fragment_item_ids, conflict_pairs, num_nodes, config["replication_factor"])

    total_fragment_weight = sum(fragment_weights.values())

    affinities = load_affinity(affinity_path)

    affinity_pairs = list(affinities.keys())

    gamma_sum = sum(affinities.values())
    
    gamma = 1 + 2*gamma_sum

    print("Conflict-Locality-based ILP: Eingabedaten")
    print("--------------------------------")
    print(f"Fragmente: {len(fragment_ids)}")
    print(f"Gesamtes Fragmentgewicht: {total_fragment_weight}")

    max_fragment_weight = max(fragment_weights.values())
    average_node_weight = total_fragment_weight / num_nodes

    min_capacity = max(
        max_fragment_weight,
        math.ceil(average_node_weight)
    )

    if node_capacity_config is None:
        node_capacity = calculate_node_capacity(
            reference_fragments_path=config["capacity_reference_path"],
            item_ids_column=item_ids_column,
            num_nodes=num_nodes,
            capacity_buffer=capacity_buffer
        )
    else:
        node_capacity = node_capacity_config

    if node_capacity < min_capacity:
        raise ValueError(f"Die konfigurierte Node Kapazität {node_capacity} ist kleiner als die minimale Node Kapazität {min_capacity}.")


    print(f"Größtes Fragmentgewicht: {max_fragment_weight}")
    print(f"Durchschnittliche Node-Gewichtung: {average_node_weight:.2f}")
    print(f"Minimale theoretische Node-Kapazität: {min_capacity}")
    print(f"Maximale festgelegte Node-Kapazität: {node_capacity}")
    print(f"Konfliktpaare: {len(conflict_pairs)}")
    print(f"Affinitätspaare: {len(affinity_pairs)}")
    print(f"Summe der Affinitäten: {gamma_sum}")

    print("\nGeladene Affinitäten:")

    for pair, affinity in affinities.items():
        print(f"{pair}: {affinity}")

    # ILP Modell mit Minimierungsziel
    model = pulp.LpProblem("Conflict_Locality_Based", pulp.LpMinimize)

    # Constraint (16)
    # Hier wird y[k] = 1 gesetzt, wenn Node k verwendet wird
    y = pulp.LpVariable.dicts("y", node_ids, cat="Binary")

    # Constraint (17)
    # Hier wird x[i][k] = 1 gesetzt, wenn Fragment i zu Node k zugewiesen wird.
    x = pulp.LpVariable.dicts("x", (fragment_ids, node_ids), cat="Binary")

    # Constraint (18)
    a = pulp.LpVariable.dicts("a", (affinity_pairs, node_ids), cat="Binary")

    # Constraint (19)
    b = pulp.LpVariable.dicts("b", (affinity_pairs, node_ids), cat="Binary")
    
    # Zielfunktion definieren
    model += (gamma * pulp.lpSum(y[node_id] for node_id in node_ids) + 
              pulp.lpSum(affinities[(fragment_i, fragment_j)] * 
                         pulp.lpSum(a[(fragment_i, fragment_j)][node_id] + 
                                    b[(fragment_i, fragment_j)][node_id] 
                                    for node_id in node_ids)
                         for fragment_i, fragment_j in affinity_pairs if fragment_i != fragment_j),
                "Minimize_nodes_separated_affinities"
            )
    
    # Constraint (11)
    for fragment_id in fragment_ids:
        model += (pulp.lpSum(x[fragment_id][node_id] for node_id in node_ids) == 1,
                  f"Assign_fragment_{fragment_id}_exactly_one")
    
    # Constraint (12)
    for node_id in node_ids:
        model += (pulp.lpSum(fragment_weights[fragment_id] * x[fragment_id][node_id]
                             for fragment_id in fragment_ids)
                             <= node_capacity * y[node_id],
                             f"Capacity_{node_id}"
                             )
    
    # Constraint (13)
    for node_id in node_ids:
        for fragment_i, fragment_j in conflict_pairs:
            model += (x[fragment_i][node_id] + x[fragment_j][node_id] <= y[node_id],
                      f"Conflict_{fragment_i}_{fragment_j}_{node_id}")
    
    # Constraint (14)
    for node_id in node_ids:
        for fragment_i, fragment_j in affinity_pairs:
            model += ((x[fragment_i][node_id] - x[fragment_j][node_id]) 
                      <= a[(fragment_i, fragment_j)][node_id],
                      f"Affinity_a_{fragment_i}_{fragment_j}_{node_id}")
    
    # Constraint (15)
    for node_id in node_ids:
        for fragment_i, fragment_j in affinity_pairs:
            model += ((x[fragment_j][node_id] - x[fragment_i][node_id]) 
                      <= b[(fragment_i, fragment_j)][node_id],
                      f"Affinity_b_{fragment_i}_{fragment_j}_{node_id}")

    print("\nConflict-Locality-based ILP wird gelöst ---")
    start_time = time.perf_counter()

    # Hier wird der solver deklariert mit CBC
    solver = pulp.HiGHS(msg=True, threads=8, timeLimit=1800, parallel="on", mip_heuristic_effort=0.5, mip_heuristic_run_shifting=True, mip_heuristic_run_zi_round=True)
    # Dem Model wird der Solver hinzugefügt und alle Gleichungen im Model werden gelöst
    model.solve(solver)

    highs_model = model.solverModel
    highs_status = highs_model.getModelStatus()
    highs_status_text = highs_model.modelStatusToString(highs_status)
    highs_info = highs_model.getInfo()

    has_variable_values = all(variable.varValue is not None for variable in model.variables())

    solver_result_path = mode_config["solver_result_path"]

    is_feasible_sol = (has_variable_values and model.valid(1e-6))

    if is_feasible_sol:
        used_nodes = sum(pulp.value(y[node_id]) > 0.5 for node_id in node_ids)
        objective_value = pulp.value(model.objective)
    else:
        used_nodes = None
        objective_value = None

    solver_result_path.parent.mkdir(parents=True, exist_ok= True)

    solver_runtime = time.perf_counter() - start_time
    solver_status = pulp.LpStatus[model.status]

    is_optimal = (highs_status == highspy.HighsModelStatus.kOptimal and highs_info.mip_gap <= 1e-9)

    solver_result = pd.DataFrame([{
        "dataset": DATASET,
        "mode": mode,
        "solver_status": highs_status_text,
        "feasible": is_feasible_sol,
        "optimal": is_optimal,
        "runtime_seconds": solver_runtime,
        "time_limit_seconds": 1800,
        "objective_value": objective_value,
        "dual_bound": highs_info.mip_dual_bound,
        "mip_gap": highs_info.mip_gap,
        "used_nodes": used_nodes,
        "threads": 8,
        "num_variables": len(model.variables()),
        "num_constraints": len(model.constraints)
    }])

    solver_result.to_csv(solver_result_path, index=False)

    print("\n Ergebnisse")
    print("-----------------------------")
    print(f"Status des Solvers: {highs_status_text}")
    print(f"Runtime: {solver_runtime} Sekunden")
    print(f"Dual Bound: {highs_info.mip_dual_bound}")
    print(f"MIP Gap: {100 * highs_info.mip_gap}%")

    if highs_status == highspy.HighsModelStatus.kOptimal:
        if highs_info.mip_gap <= 1e-9:
            print("Optimalität wurde beweisen")
        else:
            print("Lösung erfüllt die Gap-Toleranz, aber eine exakte Optimale Lösung wurde nicht gefunden.")

    elif is_feasible_sol:
        print("Eine zulässige Lösung wurde gefunden, Optimalität konnte nicht bewiesen werden.")

    else: raise RuntimeError(f"Keine Lösung gefunden. Status: {highs_status_text}")

    used_nodes = sum(pulp.value(y[node_id]) > 0.5 for node_id in node_ids)

    print(f"Verwendete nodes: {used_nodes}")

    output_assignment_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_loads_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    assignment_df = assignments(fragment_ids, fragment_weights, node_ids, x)

    loads_df = compute_loads(fragment_ids, fragment_weights, node_ids, x, y, node_capacity)

    assignment_df.to_csv(output_assignment_path, index=False)
    loads_df.to_csv(output_loads_path, index=False)

    print(f"Modus: {mode}")
    print(f"Fragmentzuweisung gespeichert unter: {output_assignment_path}")
    print(f"Node-Lasten gespeichert unter: {output_loads_path}")

    return assignment_df, loads_df

def main():
    if DATASET not in DATASETS:
        raise ValueError(f"Unbekannter Datensatz {DATASET}")

    config = DATASETS[DATASET]
    process_conflict_locality_ilp(config, MODE)

if __name__ == "__main__":
    main()
