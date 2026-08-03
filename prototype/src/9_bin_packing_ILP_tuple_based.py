import pulp
import time
import pandas as pd
from pathlib import Path
import math
import highspy
from placement_capacity import calculate_node_capacity

DATASETS = {
    "mesh": {
        "item_ids_column": "descriptor_ids",
        "num_nodes": 10,
        "node_capacity": None, # bedeutet automatische berechnung mit einem Puffer von "capacity_buffer"
        "capacity_buffer": 0.50,
        "capacity_reference_path": Path("prototype/output/processed/mesh_fragments_sample.csv"),
        "replication_factor": 3,

        "modes": {
                "baseline": {
                    "fragments_path": Path("prototype/output/processed/mesh_fragments_sample.csv"),
                    "assignment_output_path": Path("prototype/output/processed/mesh_fragment_assignment_tuple_ilp.csv"),
                    "loads_output_path": Path("prototype/output/results/mesh/tuple_ilp/node_loads.csv"),
                    "solver_result_path": Path("prototype/output/processed/mesh/mesh_baseline_solver_result_tuple_ilp.csv")

                },
                "updates": {
                    "fragments_path": Path("prototype/output/reoptimization/mesh_fragments_sample_updates.csv"),
                    "assignment_output_path": Path("prototype/output/reoptimization/mesh_fragment_assignment_tuple_ilp_updated.csv"),
                    "loads_output_path": Path("prototype/output/reoptimization/mesh/tuple_ilp/node_loads_updated.csv"),
                    "solver_result_path": Path("prototype/output/processed/mesh/mesh_updates_solver_result_tuple_ilp.csv")
                },
            
            },

        },

    "imdb": {
        "item_ids_column": "title_ids",
        "num_nodes": 10,
        "node_capacity": None, # bedeutet automatische berechnung mit einem Puffer von "capacity_buffer"
        "capacity_buffer": 0.50,
        "capacity_reference_path": Path("prototype/output/processed/imdb_fragments.csv"),
        "replication_factor": 3,

        "modes": {
            "baseline": {
                "fragments_path": Path("prototype/output/processed/imdb_fragments.csv"),
                "assignment_output_path": Path("prototype/output/processed/imdb_fragment_assignment_tuple_ilp.csv"),
                "loads_output_path": Path("prototype/output/results/imdb/tuple_ilp/node_loads.csv"),
                "solver_result_path": Path("prototype/output/processed/imdb/imdb_baseline_solver_result_tuple_ilp.csv")
            },
            "updates": {
                "fragments_path": Path("prototype/output/reoptimization/imdb_fragments_updates.csv"),
                "assignment_output_path": Path("prototype/output/reoptimization/imdb_fragment_assignment_tuple_ilp_updated.csv"),
                "loads_output_path": Path("prototype/output/reoptimization/imdb/tuple_ilp/node_loads_updated.csv"),
                "solver_result_path": Path("prototype/output/processed/imdb/imdb_updates_solver_result_tuple_ilp.csv")

            },
        },
    }
}

MODE = "baseline"
DATASET = "imdb"

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
        - fragment_item_ids:
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
    
    return fragments_df, fragment_item_ids, fragment_weights


def item_to_fragments(fragment_item_ids):
    """
    Erstellen einer Dictionary für Constraint 5, die descriptor_id auf eine Liste der Fragmente abbildet, 
    die diesen Descriptor enthalten.
    """

    item_fragments = {}

    for fragment_id, item_ids in fragment_item_ids.items():
        for item_id in item_ids:
            # Falls es noch keinen descriptor_id Eintrag in dem Set gibt, 
            # dann wird da eine leere Liste eingesetzt
            # danach wird fragment_id an die Liste angehängt
            if item_id not in item_fragments:
                item_fragments[item_id] = []
            
            item_fragments[item_id].append(fragment_id)
    
    return item_fragments

def check_replication_feasibility(item_fragments, replication_factor):
    """
    Prüfung ob jedes Tupel in mindestens zwei Fragmenten vorhanden ist, 
    da ansonsten Replication mit zum Beispiel m = 2 nicht möglich ist.
    """

    insufficient_items = {item_id: fragment_ids
                                for item_id, fragment_ids in item_fragments.items()
                                if len(fragment_ids) != replication_factor}
    
    if insufficient_items:
        raise ValueError(f"Replikationsfaktor m = {replication_factor} ist für "
                         f"{len(insufficient_items)} Items nicht erreichbar")
    
    print(f"Alle Items kommen in mindestens "
          f"{replication_factor} Fragmenten vor")

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

def compute_loads(fragment_ids, pattern_ids, pattern_weights, node_ids, x, y, z, node_capacity):
    """
    
    """
    
    load_rows = []

    for node_id in node_ids:
        assigned_fragments = [fragment_id
                              for fragment_id in fragment_ids
                              if pulp.value(x[fragment_id][node_id]) > 0.5]
        
        node_load = sum(pattern_weights[pattern_id] for pattern_id in pattern_ids if pulp.value(z[pattern_id][node_id]) > 0.5)

        load_rows.append({
            "node_id": node_id,
            "used": int(pulp.value(y[node_id]) > 0.5),
            "number_of_fragments": len(assigned_fragments),
            "node_load": node_load,
            "node_capacity": node_capacity,
            "remaining_capacity": node_capacity - node_load
        })

    return pd.DataFrame(load_rows)

def process_tuple_ilp(config, mode):
    modes = config["modes"]

    if mode not in modes:
        raise ValueError(f"Unbekannter Modus: {mode}")

    mode_config = modes[mode]

    fragments_path = mode_config["fragments_path"]
    assignment_output_path = mode_config["assignment_output_path"]
    loads_output_path = mode_config["loads_output_path"]

    for stale_output_path in (
        assignment_output_path,
        loads_output_path
    ):
        if stale_output_path.exists():
            stale_output_path.unlink()

    item_ids_column = config["item_ids_column"]
    num_nodes = config["num_nodes"]
    node_capacity_config = config.get("node_capacity")
    capacity_buffer = config.get("capacity_buffer", 0.10)
    replication_factor = config["replication_factor"]

    node_ids = [f"node_{i}" for i in range(1, num_nodes + 1)]

    fragments_df, fragment_item_ids, fragment_weights = load_fragments(fragments_path, item_ids_column)

    item_fragments = item_to_fragments(fragment_item_ids)

    fragment_ids = list(fragment_item_ids.keys())
    item_ids = list(item_fragments.keys())

    total_fragment_weight = sum(fragment_weights.values())

    fragments_per_item = [len(fragment_ids_for_items)
                                    for fragment_ids_for_items
                                    in item_fragments.values()]

    print("Tuple-based ILP: Eingabedaten")
    print("--------------------------------")
    print(f"Fragmente: {len(fragment_ids)}")
    print(f"Eindeutige Items: {len(item_ids)}")
    print(f"Gesamtes Fragmentgewicht: {total_fragment_weight}")
    
    if fragments_per_item:
        print(
            "Minimale Fragmentanzahl pro Item: "
            f"{min(fragments_per_item)}"
        )
        print(
            "Maximale Fragmentanzahl pro Item: "
            f"{max(fragments_per_item)}"
        )

    check_replication_feasibility(
        item_fragments,
        replication_factor
    )

    max_fragment_weight = max(fragment_weights.values())

    total_unique_item_weight = len(item_ids)
    min_replicated_weight = (
        replication_factor * total_unique_item_weight
    )

    # Bei genau einem Fragment pro Schema müssen beide Werte identisch sein.
    if total_fragment_weight != min_replicated_weight:
        raise ValueError(
            "Das gesamte Fragmentgewicht entspricht nicht der erwarteten "
            "Anzahl an Tuple-Kopien. Prüfe die Fragmentierung."
        )

    average_required_tuple_load = (
        min_replicated_weight / num_nodes
    )

    capacity_lower_bound = max(
        max_fragment_weight,
        math.ceil(total_fragment_weight / num_nodes)
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
    
    if node_capacity < capacity_lower_bound:
        raise ValueError(f"Die konfigurierte Node Kapazität {node_capacity} ist kleiner als die minimale Node Kapazität {capacity_lower_bound}.")


    print(f"Größtes Fragmentgewicht: {max_fragment_weight}")
    print(f"Mindestens benötigte Tuple Kopien auf Nodes: {min_replicated_weight}")
    print(f"Durchschnittliche Node Gewichtung: {average_required_tuple_load:.2f}")
    print(f"Minimale theoretische Node-Kapazität: {capacity_lower_bound}")
    print(f"Maximale festgelegte Node-Kapazität: {node_capacity}")

    pattern_for_items = {}

    for item_id, containing_fragments in item_fragments.items():
        pattern = frozenset(containing_fragments)

        pattern_for_items.setdefault(pattern, []).append(item_id)

    patterns = list(pattern_for_items.keys())

    pattern_ids = list(range(len(patterns)))

    pattern_fragments = {pattern_id: patterns[pattern_id] for pattern_id in pattern_ids}

    pattern_weights = {pattern_id: len(pattern_for_items[patterns[pattern_id]]) for pattern_id in pattern_ids}

    # ILP Modell mit Minimierungsziel
    model = pulp.LpProblem("Tuple_Based", pulp.LpMinimize)

    # Constraint (8)
    # Hier wird x[i][k] = 1 gesetzt, wenn Fragment i zu Node k zugewiesen wird.
    x = pulp.LpVariable.dicts("x", (fragment_ids, node_ids), cat="Binary")
    
    # Constraint (7)
    # Hier wird y[k] = 1 gesetzt, wenn Node k verwendet wird
    y = pulp.LpVariable.dicts("y", node_ids, cat="Binary")

    # Constraint (9)
    # z[j][k] = 1, wenn Descriptor bzw. Tupel j auf Node k verfügbar ist
    z = pulp.LpVariable.dicts("z", (pattern_ids, node_ids), cat="Binary")

    node_load_expression = {node_id: pulp.lpSum(pattern_weights[pattern_id] * z[pattern_id][node_id]
                                                for pattern_id in pattern_ids)
                            for node_id in node_ids}

    # Zielfunktion definieren
    model += (pulp.lpSum(y[node_id] for node_id in node_ids), "Minimize_number_of_used_nodes")

    # Constraint (2) 
    # Jedes Fragment wird genau einer Node zugewiesen
    for fragment_id in fragment_ids:
        model += (pulp.lpSum(x[fragment_id][node_id] for node_id in node_ids
                             ) == 1, f"Assign_fragment_{fragment_id}_exactly_once")
    
    # Constraint (3)
    # summiertes Gewicht der Fragmente auf einer Node dürfen nicht größer sein als
    # die Gewichtsschranke von einem Node k
    for node_id in node_ids:
        model += (node_load_expression[node_id] <= node_capacity * y[node_id], f"Capacity_{node_id}")

    min_used_nodes = max(replication_factor, math.ceil(min_replicated_weight / node_capacity))

    model += (pulp.lpSum(y[node_id] for node_id in node_ids) >= min_used_nodes, "Minimum_used_nodes")

    print(f"Mindestanzahl verwendeter Nodes: {min_used_nodes}")

    # Symmetrie zwischen Nodes reduzieren
    for curr_node, next_node in zip(node_ids, node_ids[1:]):
        # Verwendete Nodes stehen vorne:
        model += (y[curr_node] >= y[next_node], f"Used_node_order_{curr_node}_{next_node}")

        # Node Lasten sortieren
        model += (node_load_expression[curr_node] >= node_load_expression[next_node], f"Load_Order_{curr_node}_{next_node}")
        
    

    # Berechnen der Tuple Availability für jedes Fragment f_i die Tuple t_j enthalten
    for pattern_id, containing_fragments in pattern_fragments.items():
        pattern_size = len(containing_fragments)

        for node_id in node_ids:
            fragment_sum = pulp.lpSum(x[fragment_id][node_id] for fragment_id in containing_fragments)

            # Constraint (4) wenn Fragmente dieses Patterns auf Node liegen, so muss z positiv werden
            model += (fragment_sum <= pattern_size * z[pattern_id][node_id],
                        f"Availability_lower_{pattern_id}_{node_id}")

            # Constraint (5)
            # Hierbei ist z = 1 nur erlaubt, wenn mindestens eine der containing fragments auf der Node liegt.
            # Wenn es kein zugehöriges Fragment gibt, so ist z = 0
            model += (z[pattern_id][node_id] <= fragment_sum,
                        f"Availability_upper_"
                        f"{pattern_id}_{node_id}")


    # Constraint (6)
    # Tuple-Level Replication mit m wird sichergestellt
    for pattern_id in pattern_ids:
        model += (pulp.lpSum(z[pattern_id][node_id] 
                            for node_id in node_ids) >= replication_factor,
                            f"Replication_{pattern_id}_{replication_factor}"
                        )

    print("\nILP tuple-based wird gelöst ---")
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

    assignment_output_path.parent.mkdir(parents=True, exist_ok=True)

    loads_output_path.parent.mkdir(parents=True, exist_ok=True)

    assignment_df = assignments(fragment_ids, fragment_weights, node_ids, x)

    loads_df = compute_loads(fragment_ids, pattern_ids, pattern_weights, node_ids, x, y, z, node_capacity)

    assignment_df.to_csv(assignment_output_path, index=False)
    loads_df.to_csv(loads_output_path, index=False)

    print(f"Modus: {mode}")
    print(f"Fragmentzuweisung gespeichert unter: {assignment_output_path}")
    print(f"Node-Lasten gespeichert unter: {loads_output_path}")

    return assignment_df, loads_df



def main():
    if DATASET not in DATASETS:
        raise ValueError(f"Unbekannter Datensatz: {DATASET}")

    config = DATASETS[DATASET]
    process_tuple_ilp(config, MODE)


if __name__ == "__main__":
    main()