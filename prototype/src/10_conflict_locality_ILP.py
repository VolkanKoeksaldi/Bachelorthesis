import pulp
import time
import pandas as pd
from pathlib import Path
from experiment_config import CAPACITY_BUFFER, NUM_NODES, experiment_path
import math
import highspy
from itertools import combinations
from placement_capacity import calculate_node_capacity


DATASETS = {
    "mesh": {
        "item_ids_column": "tuple_ids",
        "affinity_path": experiment_path("workload_affinities/mesh_workload_affinities.csv"),
        "num_nodes": NUM_NODES,
        "node_capacity": None, # If value is None then the capacity is 
                               # calculated automatically using capacity buffer
        "capacity_buffer": CAPACITY_BUFFER,
        "capacity_reference_path": experiment_path("processed/mesh_fragments.csv"),


        "modes": {
            "baseline": {
                "fragments_path": experiment_path("processed/mesh_fragments.csv"),
                "overlaps_path": experiment_path("processed/mesh_overlaps.csv"),
                "assignment_output_path": experiment_path("processed/mesh_fragment_assignment_conflict_locality_ilp.csv"),
                "loads_output_path": experiment_path("results/mesh/conflict_locality_ilp/node_loads.csv"),
                "solver_result_path": experiment_path("processed/mesh/mesh_baseline_solver_result_conflict_locality_ilp.csv")

            },

            "updates": {
                "fragments_path": experiment_path("reoptimization/mesh_fragments_updates.csv"),
                "overlaps_path": experiment_path("reoptimization/mesh_overlaps_updates.csv"),
                "assignment_output_path": experiment_path("reoptimization/mesh_fragment_assignment_conflict_locality_ilp_updated.csv"),
                "loads_output_path": experiment_path("reoptimization/mesh/conflict_locality_ilp/node_loads_updated.csv"),
                "solver_result_path": experiment_path("processed/mesh/mesh_updates_solver_result_conflict_locality_ilp.csv")
           
            }
        }
    },

    "imdb": {
        "item_ids_column": "title_ids",
        "affinity_path": experiment_path("workload_affinities/imdb_workload_affinities.csv"),
        "num_nodes": NUM_NODES,
        "node_capacity": None, # If value is None then the capacity is
                               # calculated automatically using capacity buffer
        "capacity_buffer": CAPACITY_BUFFER,
        "capacity_reference_path": experiment_path("processed/imdb_fragments.csv"),


        "modes": {
            "baseline": {
                "fragments_path": experiment_path("processed/imdb_fragments.csv"),
                "overlaps_path": experiment_path("processed/imdb_overlaps.csv"),
                "assignment_output_path": experiment_path("processed/imdb_fragment_assignment_conflict_locality_ilp.csv"),
                "loads_output_path": experiment_path("results/imdb/conflict_locality_ilp/node_loads.csv"),
                "solver_result_path": experiment_path("processed/imdb/imdb_baseline_solver_result_conflict_locality_ilp.csv")
            },

            "updates": {
                "fragments_path": experiment_path("reoptimization/imdb_fragments_updates.csv"),
                "overlaps_path": experiment_path("reoptimization/imdb_overlaps_updates.csv"),
                "assignment_output_path": experiment_path("reoptimization/imdb_fragment_assignment_conflict_locality_ilp_updated.csv"),
                "loads_output_path": experiment_path("reoptimization/imdb/conflict_locality_ilp/node_loads_updated.csv"),
                "solver_result_path": experiment_path("processed/imdb/imdb_updates_solver_result_conflict_locality_ilp.csv")           
            }
        }
    }
}

MODE = "updates" # baseline or updates
DATASET = "imdb"

def parse_item_ids(item_ids_string):
    """
    Converts item ids string into a set.

    Example:
    "D000001,D000002,D000003"
    -> {"D000001", "D000002", "D000003"}
    """

    # Empty or missing ids list are returned as an empty set
    if pd.isna(item_ids_string) or item_ids_string == "":
        return set()

    return set(item_ids_string.split(","))

def load_fragments(fragments_path, item_ids_column):
    """
    Loads fragment csv file and calculates the weight of each fragment.

    return:
        - fragments_df as a DataFrame with the additional item_id_set column
        - fragment_tuple_ids as a dictionary mapping every fragment id
          to the set of items that contains it
        - fragment_weights as a dictionary mapping every fragment id to its weight.
          Weight equals the number of contained items in fragment.
    """

    if not fragments_path.exists():
        raise FileNotFoundError(f"Fragment data not found : {fragments_path}")
    
    fragments_df = pd.read_csv(fragments_path)

    # Verifies scheme by checking whether every required column for calculation is existent
    required_columns = {"fragment_id", item_ids_column}
    missing_columns = required_columns - set(fragments_df.columns)

    if missing_columns:
        raise ValueError(f"Fragment stored on {fragments_path} is missing following columns: {missing_columns}")

    # Converts item ids of each fragment into a set of item ids.
    fragments_df["item_id_set"] = (fragments_df[item_ids_column].apply(parse_item_ids))

    # Removes empty fragments and then creates a DataFrame copy
    fragments_df = fragments_df[fragments_df["item_id_set"].map(len) > 0].copy()

    # Maps every fragment it to set of items that are contained in the fragment
    fragment_item_ids = {row.fragment_id: row.item_id_set
                                for row in fragments_df.itertuples()}
    
    # Calculates the weight of a fragment, which equals number of items in a fragment
    fragment_weights = {fragment_id: len(item_ids)
                        for fragment_id, item_ids in fragment_item_ids.items()}

    # Checks whether every fragment_id in fragment DataFrame is unique
    if not fragments_df["fragment_id"].is_unique:
        raise ValueError("Fragment file contains duplicate fragment_id values.")
    
    return fragments_df, fragment_item_ids, fragment_weights

def validate_conflicts(fragment_item_ids, conflict_pairs, num_nodes):
    """
    Validates the loaded conflict pairs. These need to cover every item overlap.
    First maps items to all fragments that contain that item.  The number of
    memberships is the item's implicit replication count.
    The function then verifies that every implied conflict pair occurs in
    the overlap file.
    """

    item_fragments = {}

    # maps item ids to set of fragments that contain that item id
    for fragment_id, item_ids in fragment_item_ids.items():
        for item_id in item_ids:
            item_fragments.setdefault(item_id, set()).add(fragment_id)

    min_memberships = min((len(fragments) for fragments in item_fragments.values()), default=0)

    # Determines the largest implicit item replication count.
    max_memberships = max((len(fragments) for fragments in item_fragments.values()), default=0)

    # Checks that if fragment memberships is greater than number of nodes, then
    # there have to be overlaps in a single node between fragments that are stored in that node
    if max_memberships > num_nodes:
        raise ValueError(f"Conflict model infeasible: one item belongs to {max_memberships} fragments, but only {num_nodes} nodes exist.")

    # Generates fragment pairs that should be a conflict according to item memberships in fragment file
    expected_pairs = set()

    for fragments in item_fragments.values():
        expected_pairs.update(combinations(sorted(fragments), 2))

    # normalizes order of loaded pair ids.
    # this means that pair: (i, l) and (l, i) are treated as same conflict pair.
    loaded_pairs = {tuple(sorted((fragment_i, fragment_j))) for fragment_i, fragment_j in conflict_pairs
                    if fragment_i != fragment_j}

    # Conflict pairs only reference fragments when they are part of the model
    known_fragments = set(fragment_item_ids)
    unknown_pairs = {pair for pair in loaded_pairs if pair[0] not in known_fragments or pair[1] not in known_fragments}

    if unknown_pairs:
        raise ValueError(f"Overlap file references {len(unknown_pairs)} conflict pairs "
                         "that contain unknown or empty fragments.")

    # Overlaps implied by fragment file must also occur in overlap file
    missing_pairs = expected_pairs - loaded_pairs

    if missing_pairs:
        raise ValueError(f"Overlap file is incomplete: {len(missing_pairs)} conflict pairs are missing.")

    unexpected_pairs = loaded_pairs - expected_pairs

    if unexpected_pairs:
        raise ValueError(f"Overlap file contains {len(unexpected_pairs)} conflict pairs that actually should not overlap.")

    return min_memberships, max_memberships


def assignments(fragment_ids, fragment_weights, node_ids, x):
    """
    Extracts fragment -> node assignments from solution.
    Identifies for every fragment the node for which x[i][k] = 1.
    Then returns a DataFrame that contains fragment id, assigned node id, and fragment weight for fragments.
    """
    assignment_rows = []

    # Finds all nodes where fragment was assigned
    # Using Constraint 11 should have made this list contain exactly one node for each fragment
    for fragment_id in fragment_ids:
        assigned_nodes = [node_id for node_id in node_ids if pulp.value(x[fragment_id][node_id]) > 0.5]

        # Checks whether fragment is stored on multiple nodes or not at all
        if len(assigned_nodes) != 1:
            raise RuntimeError(f"Fragment {fragment_id} was assigned to {len(assigned_nodes)} nodes.")
        
        assignment_rows.append({"fragment_id": fragment_id,
                                "node_id": assigned_nodes[0],
                                "fragment_weight": fragment_weights[fragment_id]})
        
    return pd.DataFrame(assignment_rows)


def compute_loads(fragment_ids, fragment_weights, node_ids, x, y, node_capacity):
    """
    Calculates load and remaining capacity of nodes.
    Load = sum of the weights of all fragments on a node.
    Returns a DataFrame that contains usage state, number of assigned fragments,
    fragment load, capacity, and remaining capacity of every node.
    """
    
    load_rows = []

    for node_id in node_ids:
        # Determines which fragments were assigned to node_id.
        assigned_fragments = [fragment_id for fragment_id in fragment_ids
                              if pulp.value(x[fragment_id][node_id]) > 0.5]

        # Adds weight of every assigned node as node_load
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
    Loads overlapping fragment pairs from csv.
    Only rows with positive overlap size are used because only these pairs
    share items and therefore require the conflict constraint.
    Then returns a DataFrame with the filtered overlap data
    and a list of fragment id pairs with positive overlap.
    """

    if not overlaps_path.exists():
        raise FileNotFoundError(f"Overlap file not found: {overlaps_path}")

    overlap_file = pd.read_csv(overlaps_path)

    # Verifies expected overlap-file scheme:
    required_columns = {"fragment_1", "fragment_2", "overlap_size"}
    missing_columns = required_columns - set(overlap_file.columns)

    if missing_columns:
        raise ValueError(f"Overlap file {overlaps_path} has some missing columns: {missing_columns}")

    # Filters the overlap size, so that empty overlaps are removed
    overlap_file = overlap_file[overlap_file["overlap_size"] > 0].copy()

    # Normalizes pairs by removing self-pairs (i, i) and eliminates duplicates.
    conflict_pairs = sorted({tuple(sorted((row.fragment_1, row.fragment_2)))
                            for row in overlap_file.itertuples()
                            if row.fragment_1 != row.fragment_2})
    
    # overlap_pair = list(overlap_file[["fragment_1", "fragment_2"]].itertuples(index=False, name=None))
    
    return overlap_file, conflict_pairs

def load_affinity(affinity_path, valid_fragment_ids):
    """
    Loads and aggregates fragment affinities.
    If affinity file contains several entries for the same pair, then their affinity
    values are added together. When pairs reference fragments that are not a part of 
    the current model, they are ignored. Necessary for update mode.

    Then returns a tuple of dictionary with fragment pair -> affinity value and number of
    skipped rows that reference unavailable fragments.
    """

    if not affinity_path.exists():
        raise FileNotFoundError(f"Affinity file not found: {affinity_path}")
    
    affinity_df = pd.read_csv(affinity_path)

    # Verifies the expected affinity file scheme
    required_columns = {"fragment_i", "fragment_j", "affinity"}
    missing_columns = required_columns - set(affinity_df.columns)

    if missing_columns:
        raise ValueError(f"{affinity_path} is missing columns: {missing_columns}")

    affinities = {}
    valid_fragment_ids = set(valid_fragment_ids)
    unavailable_pairs = 0

    # iterates through affinity DataFrame and extracts information
    for _, row in affinity_df.iterrows():
        fragment_i = row["fragment_i"]
        fragment_j = row["fragment_j"]
        affinity = float(row["affinity"])

        if not math.isfinite(affinity) or affinity < 0:
            raise ValueError("Affinity value must be non negative and a finite value. "
                             f"There are invalid values for ({fragment_i}, {fragment_j})"
                             f" with value {affinity}")

        # An affinity of zero does not affec tthe objective function and therefore is not required.
        if affinity == 0:
            continue

        # if same fragment then the self pair is ignored as it does not affect the objective function.
        if fragment_i == fragment_j:
            continue

        # Ignores pairs that are no longer in the current model.
        if (fragment_i not in valid_fragment_ids
            or fragment_j not in valid_fragment_ids):
            unavailable_pairs += 1
            continue

        # normalizes pair order
        pair = tuple(sorted((fragment_i, fragment_j)))

        # adds repeated observation of the same pair
        affinities[pair] = affinities.get(pair, 0) + affinity

    return affinities, unavailable_pairs

def process_conflict_locality_ilp(config, mode):
    """
    Loads selected data and then constructs and solves conflict-locality ILP.
    Then stores the resulting assignments, node loads and statistics.
    Returns a tuple of the two DataFrames that contain assignments and node loads.
    """

    if mode not in config["modes"]:
        raise ValueError(f"Unknown mode: {mode}")

    mode_config = config["modes"][mode]

    fragments_path = mode_config["fragments_path"]
    overlaps_path = mode_config["overlaps_path"]
    output_assignment_path = mode_config["assignment_output_path"]
    output_loads_path = mode_config["loads_output_path"]

    # Deletes old assignment and load files
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
    # Uses 0.10 as capacity buffer if there is no specified value in configuration
    capacity_buffer = config.get("capacity_buffer", 0.10)

    node_ids = [f"node_{i}" for i in range(1, num_nodes + 1)]

    fragments_df, fragment_item_ids, fragment_weights = load_fragments(fragments_path, item_ids_column)

    # stores ids of all non-empty fragments
    fragment_ids = list(fragment_item_ids.keys())

    overlaps_df, conflict_pairs = load_overlap_pairs(overlaps_path)

    # Validates memberships and verifies overlap file
    min_memberships, max_memberships = validate_conflicts(fragment_item_ids, conflict_pairs, num_nodes)

    total_fragment_weight = sum(fragment_weights.values())

    # loads and aggregates affinity values
    affinities, unavailable_pairs = load_affinity(affinity_path, fragment_ids)

    affinity_pairs = list(affinities.keys())

    # Calculates sum of all affinity values
    gamma_sum = sum(affinities.values())

    # Affinity pair contributes 2*affinity to objective.
    # Gamma is larger than max possible total separation penalty for affinities.
    # Gives node minimization priority over affinity locality.
    gamma = 1 + 2*gamma_sum

    print("Conflict-Locality-based ILP: input data")
    print("--------------------------------")
    print(f"Number of fragments: {len(fragment_ids)}")
    print(f"Total fragment weight: {total_fragment_weight}")
    print("Implicit item replication range: "f"{min_memberships} to {max_memberships} fragment memberships")

    # Every fragment must fit completely on one node, so calculating max_fragment_weight
    # ensures that node_capacity cannot be smaller than largest fragment weight
    max_fragment_weight = max(fragment_weights.values())

    # Calculates average node weight as a lower bound when all configures
    # nodes are available for distributing total fragment weight
    average_node_weight = total_fragment_weight / num_nodes

    # Combines max_fragment_weight and average_node_weight as lower bounds
    min_capacity = max(max_fragment_weight, math.ceil(average_node_weight))

    # Calculates node capacity according to configuration.
    # Baseline fragment file is here used as common capacity reference for baseline and updates mode.
    if node_capacity_config is None:
        node_capacity = calculate_node_capacity(
            reference_fragments_path=config["capacity_reference_path"],
            item_ids_column=item_ids_column,
            num_nodes=num_nodes,
            capacity_buffer=capacity_buffer
        )
    else:
        node_capacity = node_capacity_config

    # checks whether capacity is smaller than lower bound
    if node_capacity < min_capacity:
        raise ValueError(f"Configured node capacity {node_capacity} is smaller than minimum node capacity {min_capacity}.")

    # checks whether enough nodes for both total fragment weight and largest
    # pairwise conflicting fragments are in feasible solution.
    # Represents a lower bound
    min_used_nodes = max(math.ceil(total_fragment_weight / node_capacity), max_memberships)


    print(f"Largest fragment weight: {max_fragment_weight}")
    print(f"Average node weight: {average_node_weight:.2f}")
    print(f"Theoretical minimum node capacity: {min_capacity}")
    print(f"Maximum configured node capacity: {node_capacity}")
    print(f"Number of conflict pairs: {len(conflict_pairs)}")
    print(f"Number of affinity pairs: {len(affinity_pairs)}")
    print(f"Sum of affinity values: {gamma_sum}")

    if unavailable_pairs:
        print(f"Affinity rows ignored because a fragment was unavailable: {unavailable_pairs}")

    print("\nLoaded affinities:")

    # Displays short preview of 10 affinity pairs
    affinity_preview = 10

    for pair, affinity in list(affinities.items())[:affinity_preview]:
        print(f"{pair}: {affinity}")

    if len(affinities) > affinity_preview:
        print(f"... There are {len(affinities) - affinity_preview} additional affinity pairs.")

    # Create ILP model with minimization objective
    model = pulp.LpProblem("Conflict_Locality_Based", pulp.LpMinimize)

    # Constraint (16)
    # y[k] = 1 if node k is used
    # y[k] = 0 otherwise
    y = pulp.LpVariable.dicts("y", node_ids, cat="Binary")

    # Constraint (17)
    # x[i][k] = 1 if fragment i is assigned to node k
    # x[i][k] = 0 otherwise
    x = pulp.LpVariable.dicts("x", (fragment_ids, node_ids), cat="Binary")

    # Constraint (18)
    # a[(i, l)][k] represents positive assignment difference
    # x[i][k] - x[l][k] is being calculated. This means, it becomes one on node containing
    # fragment i when affine pair is separated
    a = pulp.LpVariable.dicts("a", (affinity_pairs, node_ids), cat="Binary")

    # Constraint (19)
    # b[(i, l)][k] represents reverse assignment difference x[l][k] - x[i][k].
    # This means, it becomes one on node containing fragment l when affine pair is separated
    b = pulp.LpVariable.dicts("b", (affinity_pairs, node_ids), cat="Binary")
    
    # Is the Objective function.
    # First term minimizes number of used nodes. By multiplying it with gamma it gives this term
    # priority over complete locality term.
    # Second term penalizes the separation of affine fragment pairs.
    # If fragments i and l are assigned to different nodes, variable a and variable b become one.
    # That pair therefore contributes 2*affinities[(i, l)] to objective value.
    model += (gamma * pulp.lpSum(y[node_id] for node_id in node_ids) + 
              pulp.lpSum(affinities[(fragment_i, fragment_j)] * 
                         pulp.lpSum(a[(fragment_i, fragment_j)][node_id] + 
                                    b[(fragment_i, fragment_j)][node_id] 
                                    for node_id in node_ids)
                         for fragment_i, fragment_j in affinity_pairs if fragment_i != fragment_j),
                "Minimize_nodes_separated_affinities"
            )
    
    # Constraint (11)
    # Every fragment is here assigned to exactly one node
    for fragment_id in fragment_ids:
        model += (pulp.lpSum(x[fragment_id][node_id] for node_id in node_ids) == 1,
                  f"Assign_fragment_{fragment_id}_exactly_one")
    
    # Constraint (12)
    # Checks whether the sum of weights of all fragments assigned to node k
    # exceed its nodes max capacity or not.
    # By multiplying with y[node_id] the node load is linked to node usage.
    # y[k] = 0: right-hand side is zero -> no fragment can use the node
    # y[1] = 1: node may contain data with size <= node capacity
    for node_id in node_ids:
        model += (pulp.lpSum(fragment_weights[fragment_id] * x[fragment_id][node_id]
                             for fragment_id in fragment_ids)
                             <= node_capacity * y[node_id],
                             f"Capacity_{node_id}"
                             )

    # Adds lower bound calculated from capacity and conflicts to the model.
    # Gives the solver useful information before branch-and-bound search.
    model += (pulp.lpSum(y[node_id] for node_id in node_ids) >= min_used_nodes,
              "Minimum_used_nodes")

    # Used nodes need to appear before unused nodes, thus removing equivalent pair permutations
    # without changing the placement itself.
    for curr_node, next_node in zip(node_ids, node_ids[1:]):
        model += (y[curr_node] >= y[next_node],
                    f"Used_node_order_{curr_node}_{next_node}")
    
    # Constraint (13)
    # Two fragments that have an overlap cannot be assigned to the same node.
    # That means if node k is used, then the right-hand side also equals one,
    # therefore at most one of the two assignment variables can be equal one.
    for node_id in node_ids:
        for fragment_i, fragment_j in conflict_pairs:
            model += (x[fragment_i][node_id] + x[fragment_j][node_id] <= y[node_id],
                      f"Conflict_{fragment_i}_{fragment_j}_{node_id}")
    
    # Constraint (14)
    # If fragment i is assigned to node k and fragment l is not, then
    # x[i][k] - x[l][k] = 1 and forces a[(i, l)][k] = 1
    for node_id in node_ids:
        for fragment_i, fragment_j in affinity_pairs:
            model += ((x[fragment_i][node_id] - x[fragment_j][node_id]) 
                      <= a[(fragment_i, fragment_j)][node_id],
                      f"Affinity_a_{fragment_i}_{fragment_j}_{node_id}")
    
    # Constraint (15)
    # If fragment l is assigned to node k and fragment i is not, then
    # x[l][k] - x[i][k] = 1 and forces b[(i, l)][k] = 1.
    # Together with Constraint 14, they model whether an affine fragment
    # pair is assigned to different nodes
    for node_id in node_ids:
        for fragment_i, fragment_j in affinity_pairs:
            model += ((x[fragment_j][node_id] - x[fragment_i][node_id]) 
                      <= b[(fragment_i, fragment_j)][node_id],
                      f"Affinity_b_{fragment_i}_{fragment_j}_{node_id}")

    print("\nSolving conflict-locality-based ILP---")

    # Starts measuring complete runtime
    start_time = time.perf_counter()

    # Configures HiGHS solver and MIP stands for Mixed-Integer Programming
    # msg=True: Display solver log
    # threads=8: allows solver to use up to 8 CPU threads
    # timeLimit=1800: stops solver after 1800 seconds
    # parallel="on": Enables parallel solver execution
    # mip_heuristic_effort=0.5: Increases effort spent on heuristics 
    #                           for finding feasible solutions (finds a feasible solution quicker)
    # mip_heuristic_run_shifting=True: Enables shifting heuristics for quicker finding feasible solutions
    # mip_heuristic_run_zi_round=True: Enable ZI round heuristics, which attempts to obtain feasible 
    #                                  solutions by rounding a LP solution
    solver = pulp.HiGHS(msg=True, threads=8, timeLimit=1800, parallel="on", mip_heuristic_effort=0.5, mip_heuristic_run_shifting=True, mip_heuristic_run_zi_round=True)

    model.solve(solver)

    # Accesses model to retrieve status and MIP information
    highs_model = model.solverModel

    # retrieves status
    highs_status = highs_model.getModelStatus()

    # Converts internal status into string
    highs_status_text = highs_model.modelStatusToString(highs_status)

    # retrieves additional solver information like dual bound and remaining MIP gap
    highs_info = highs_model.getInfo()

    # checks whether variables received a value from solver
    has_variable_values = all(variable.varValue is not None for variable in model.variables())

    solver_result_path = mode_config["solver_result_path"]

    # checks whether solution is feasible if:
    #   1. all variables have values
    #   2. and all constraints are satisfied within tolerance of 1e-6 
    is_feasible_sol = (has_variable_values and model.valid(1e-6))

    # only reads solution values if there is a feasible solution
    if is_feasible_sol:
        used_nodes = sum(pulp.value(y[node_id]) > 0.5 for node_id in node_ids)
        objective_value = pulp.value(model.objective)
    else:
        used_nodes = None
        objective_value = None

    solver_result_path.parent.mkdir(parents=True, exist_ok= True)

    # stops runtime measurement
    solver_runtime = time.perf_counter() - start_time

    # retrieves PuLP status. Output however only uses HiGHS status text.
    solver_status = pulp.LpStatus[model.status]

    # Checks whether solver has proven optimality if it reports optimal status with
    # remaining MIP gap <= 1e-9
    is_optimal = (is_feasible_sol and highs_status == highspy.HighsModelStatus.kOptimal)

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

    print("\nResults")
    print("-----------------------------")
    print(f"Solver status: {highs_status_text}")
    print(f"Runtime: {solver_runtime} seconds")
    print(f"Dual Bound: {highs_info.mip_dual_bound}")
    print(f"MIP Gap: {100 * highs_info.mip_gap}%")

    # Distinguishes between optimal solution, feasible but not proven optimal solution,
    # and execution without feasible solution
    if is_optimal:
        print("Optimality of solution was proven.")
    elif is_feasible_sol:
        print("Feasible solution was found, but optimality could not be proven.")

    else: raise RuntimeError(f"No feasible solution found. Solver status: {highs_status_text}")

    # Calculates number of used nodes from solution
    used_nodes = sum(pulp.value(y[node_id]) > 0.5 for node_id in node_ids)

    print(f"Used nodes: {used_nodes}")

    output_assignment_path.parent.mkdir(parents=True, exist_ok=True)

    output_loads_path.parent.mkdir(parents=True, exist_ok=True)

    # Extracts fragment assignments from solution
    assignment_df = assignments(fragment_ids, fragment_weights, node_ids, x)

    # Calculates load and remaining capacity of every node
    loads_df = compute_loads(fragment_ids, fragment_weights, node_ids, x, y, node_capacity)

    assignment_df.to_csv(output_assignment_path, index=False)
    loads_df.to_csv(output_loads_path, index=False)

    print(f"Mode: {mode}")
    print(f"Fragment assignment saved: {output_assignment_path}")
    print(f"Node loads saved: {output_loads_path}")

    return assignment_df, loads_df

def main():
    if DATASET not in DATASETS:
        raise ValueError(f"Unknown dataset: {DATASET}")

    config = DATASETS[DATASET]
    process_conflict_locality_ilp(config, MODE)

if __name__ == "__main__":
    main()