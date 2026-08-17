import pulp
import time
import pandas as pd
import math
import highspy
from placement_capacity import calculate_node_capacity
from experiment_config import CAPACITY_BUFFER, NUM_NODES, REPLICATION_FACTOR, experiment_path

MODE = "baseline" # baseline or updates
DATASET = "mesh" # imdb or mesh



DATASETS = {
    "mesh": {
        "item_ids_column": "tuple_ids",
        "num_nodes": NUM_NODES,
        "node_capacity": None, # If value is None then the capacity 
                                # is calculated automatically using capacity buffer
        "capacity_buffer": CAPACITY_BUFFER,
        "capacity_reference_path": experiment_path("processed/mesh_fragments.csv"),
        "replication_factor": REPLICATION_FACTOR,

        "modes": {
                "baseline": {
                    "fragments_path": experiment_path("processed/mesh_fragments.csv"),
                    "assignment_output_path": experiment_path(
                        "processed/mesh_fragment_assignment_tuple_ilp.csv"),
                    "loads_output_path": experiment_path(
                        "results/mesh/tuple_ilp/node_loads.csv"),
                    "solver_result_path": experiment_path(
                        "processed/mesh/mesh_baseline_solver_result_tuple_ilp.csv")

                },
                "updates": {
                    "fragments_path": experiment_path(
                        "reoptimization/mesh_fragments_updates.csv"),
                    "assignment_output_path": experiment_path(
                        "reoptimization/mesh_fragment_assignment_tuple_ilp_updated.csv"),
                    "loads_output_path": experiment_path(
                        "reoptimization/mesh/tuple_ilp/node_loads_updated.csv"),
                    "solver_result_path": experiment_path(
                        "processed/mesh/mesh_updates_solver_result_tuple_ilp.csv")
                },
            
            },

        },

    "imdb": {
        "item_ids_column": "title_ids",
        "num_nodes": NUM_NODES,
        "node_capacity": None, # If value is None then the capacity
                                # is calculated automatically using capacity buffer
        "capacity_buffer": CAPACITY_BUFFER,
        "capacity_reference_path": experiment_path("processed/imdb_fragments.csv"),
        "replication_factor": REPLICATION_FACTOR,

        "modes": {
            "baseline": {
                "fragments_path": experiment_path("processed/imdb_fragments.csv"),
                "assignment_output_path": experiment_path(
                    "processed/imdb_fragment_assignment_tuple_ilp.csv"),
                "loads_output_path": experiment_path(
                    "results/imdb/tuple_ilp/node_loads.csv"),
                "solver_result_path": experiment_path(
                    "processed/imdb/imdb_baseline_solver_result_tuple_ilp.csv")
            },
            "updates": {
                "fragments_path": experiment_path(
                    "reoptimization/imdb_fragments_updates.csv"),
                "assignment_output_path": experiment_path(
                    "reoptimization/imdb_fragment_assignment_tuple_ilp_updated.csv"),
                "loads_output_path": experiment_path(
                    "reoptimization/imdb/tuple_ilp/node_loads_updated.csv"),
                "solver_result_path": experiment_path(
                    "processed/imdb/imdb_updates_solver_result_tuple_ilp.csv")

            }
        }
    }
}



def parse_item_ids(item_ids_string):
    """
    Converts comma-separated item ids stored in a CSV file into a Python set.

    Parameters:
        item_ids_string: item ids field from CSV
    
    Returns:
        Set of item id Strings
    """

    if pd.isna(item_ids_string) or item_ids_string == "":
        return set()

    return set(item_ids_string.split(","))


def load_fragments(fragments_path, item_ids_column):
    """
    Loads fragments from CSV file and prepares the information for the ILP.

    Parameters:
        fragments_path: Path to fragments CSV file
        item_ids_column: Defines on which column the item ids are stored

    Returns:
        fragments_df:
            fragment DataFrame with additional column "item_id_set"
        fragment_item_ids:
            Dictionary mapping fragment id to the set of items contained in fragment
        fragment_weights:
            Dictionary mapping every fragment id to its weight. 
            Weight = number of unique items contained in fragment.
    """

    if not fragments_path.exists():
        raise FileNotFoundError(f"Fragment file not found under path: {fragments_path}")

    fragments_df = pd.read_csv(fragments_path)

    # Converts item ids of every fragment into a set and stores the result in a new column
    fragments_df["item_id_set"] = fragments_df[item_ids_column].apply(parse_item_ids)

    # Removes empty fragments
    fragments_df = fragments_df[fragments_df["item_id_set"].map(len) > 0].copy()

    # Constructs a dictionary that maps fragment id to the set of items contained in the fragment
    fragment_item_ids = {row.fragment_id: row.item_id_set for row in fragments_df.itertuples()}
    
    # Calculates the weight of every fragment
    fragment_weights = {fragment_id: len(item_ids) for fragment_id, 
                        item_ids in fragment_item_ids.items()}
    
    return fragments_df, fragment_item_ids, fragment_weights


def item_to_fragments(fragment_item_ids):
    """
    Maps every item to the list of fragments that contain that item.
    
    Example:
        Input:
            fragment_1 -> {item_a, item_b}
            fragment_2 -> {item_a, item_c}
        
        Result:
            item_a -> [fragment_1, fragment_2]
            item_b -> [fragment_1]
            item_c -> [fragment_2]
    
    Parameters:
        fragment_item_ids: Dictionary that maps fragment id to its set of items

    Returns:
        item_fragments: Dictionary that maps items to fragments they belong to
    """

    item_fragments = {}

    # Processes every item contained in every fragment from fragment_item_ids dictionary
    for fragment_id, item_ids in fragment_item_ids.items():
        for item_id in item_ids:
            # if item has not been seen encountered, an empty list is created for its
            # fragment memberships
            if item_id not in item_fragments:
                item_fragments[item_id] = []
            
            item_fragments[item_id].append(fragment_id)
    
    return item_fragments

def check_replication_feasibility(item_fragments, replication_factor):
    """
    Checks whether every item is in atleast "replication_factor" amount of fragments.
    If not, then replication is not possible as every fragment is assigned to one node. 

    Parameters:
        item_fragments: A map from every item to fragments they belong to
        replication_factor: The required replication factor
    """

    # Identifies all items that occur in fewer fragments than required
    insufficient_items = {item_id: fragment_ids
                                for item_id, fragment_ids in item_fragments.items()
                                if len(fragment_ids) < replication_factor}

    if insufficient_items:
        raise ValueError(f"Replication factor m = {replication_factor} cannot be achieved for "
                         f"{len(insufficient_items)} items.")
    
    print(f"All items occur in at least {replication_factor} fragments.")

def assignments(fragment_ids, fragment_weights, node_ids, x):
    """
    Extracts fragment to node assignments from the solution.
    For every fragment, the node is determined for which x[i][k] = 1.
    Returns a DataFrame that contains the fragment id, assigned node id, 
    and fragment weight for every fragment.

    Parameters:
        fragment_ids: The ids of every fragment
        fragment_weights: The weights corresponding to every fragment_id
        node_ids: The node_ids where fragments are stored on
        x: Describes whether fragment i is assigned to node k

    Returns:
        A DataFrame of the assignment rows for every fragment_id to their node_id and weights
    """

    assignment_rows = []

    # Finds all nodes to which current fragment is assigned on.
    # The resulting assigned_nodes list should only contain one node, 
    # because Constraint 2 assigns every fragment exactly once.
    for fragment_id in fragment_ids:
        assigned_nodes = [node_id for node_id in node_ids 
                          if pulp.value(x[fragment_id][node_id]) > 0.5]

        # Detects invalid solver results.
        if len(assigned_nodes) != 1:
            raise RuntimeError(f"Fragment {fragment_id} "
                               f"was assigned to {len(assigned_nodes)} nodes.")
        
        assignment_rows.append({"fragment_id": fragment_id,
                                "node_id": assigned_nodes[0],
                                "fragment_weight": fragment_weights[fragment_id]})
        
    return pd.DataFrame(assignment_rows)

def compute_loads(fragment_ids, pattern_ids, pattern_weights, node_ids, x, y, z, node_capacity):
    """
    Calculates load and remaining capacity of every node.
    Node load is calculated by using variable z.
    Then returns a DataFrame containing usage state, number of assigned fragments,
    tuple load, capacity of nodes, and remaining capacity of nodes.

    Parameters:
        fragment_ids: IDs of every fragment
        pattern_ids: Every membership pattern id
        pattern_weights: Map from each pattern id to their number of represented items
        x: Fragment-assignment variable
        y: Node-usage variable
        z: Membership-pattern availability variable
        node_capacity: Describes the maximum capacity of each node

    Returns:
        A DataFrame that contains the usage state, number of assigned fragments,
        node load, node capacity, and remaining capacities.
    """
    
    load_rows = []

    for node_id in node_ids:
        assigned_fragments = [fragment_id for fragment_id in fragment_ids 
                              if pulp.value(x[fragment_id][node_id]) > 0.5]

        # Calculates node load by summing weights of all membership patterns on node_id
        # z[pattern_id][node_id] = 1 when items represented by pattern are available on node
        node_load = sum(pattern_weights[pattern_id] for pattern_id in pattern_ids 
                        if pulp.value(z[pattern_id][node_id]) > 0.5)

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
    """
    Builds tuple-based ILP and solves it for the selected configurations.
    Assignments, node loads and statistics are then stored as CSV files.

    Parameters:
        config: Configuration with dataset-specific paths and parameters
        mode: Execution mode with "baseline" or "updates"

    Returns:
        assignment_df: DataFrame with the fragment assignment
        loads_df: DataFrame with the node-loads
    """

    modes = config["modes"]

    if mode not in modes:
        raise ValueError(f"Unknown mode: {mode}")

    mode_config = modes[mode]

    fragments_path = mode_config["fragments_path"]
    assignment_output_path = mode_config["assignment_output_path"]
    loads_output_path = mode_config["loads_output_path"]

    # Deletes assignment and load files from earlier executions.
    for stale_output_path in (assignment_output_path, loads_output_path):
        if stale_output_path.exists():
            stale_output_path.unlink()

    item_ids_column = config["item_ids_column"]
    num_nodes = config["num_nodes"]
    node_capacity_config = config.get("node_capacity")
    # Sets a default buffer if no capacity buffer is specified.
    capacity_buffer = config.get("capacity_buffer", 0.10)
    replication_factor = config["replication_factor"]

    # Generates node ids.
    node_ids = [f"node_{i}" for i in range(1, num_nodes + 1)]

    # Loads fragment data and calculates fragment weights.
    fragments_df, fragment_item_ids, fragment_weights = load_fragments(fragments_path, 
                                                                       item_ids_column)

    item_fragments = item_to_fragments(fragment_item_ids)

    fragment_ids = list(fragment_item_ids.keys())
    item_ids = list(item_fragments.keys())

    total_fragment_weight = sum(fragment_weights.values())

    fragments_per_item = [len(fragment_ids_for_items)
                                    for fragment_ids_for_items
                                    in item_fragments.values()]

    print("Tuple-based ILP: Input")
    print("--------------------------------")
    print(f"Number of fragments: {len(fragment_ids)}")
    print(f"Number of unique items: {len(item_ids)}")
    print(f"Total fragment weight: {total_fragment_weight}")
    
    if fragments_per_item:
        print(f"Minimum number of fragments per item: {min(fragments_per_item)}")
        print(f"Maximum number of fragments per item: {max(fragments_per_item)}")

    check_replication_feasibility(item_fragments, replication_factor)

    max_fragment_weight = max(fragment_weights.values())

    total_num_unique_items = len(item_ids)

    # Constraint 6 requires that every item is available on atleast m nodes.
    # Shows the minimum total number of item copies that must be stored across all nodes.
    min_replicated_weight = (replication_factor * total_num_unique_items)

    average_required_tuple_load = (min_replicated_weight / num_nodes)

    # Calculates lower bound for node capacity.
    capacity_lower_bound = max(max_fragment_weight, math.ceil(min_replicated_weight / num_nodes))

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
        raise ValueError(f"Node capacity of {node_capacity} "
                         f"is smaller than capacity lower bound of {capacity_lower_bound}.")


    print(f"Largest fragment weight: {max_fragment_weight}")
    print(f"Minimum number of tuple copies needed on nodes: {min_replicated_weight}")
    print(f"Average node load: {average_required_tuple_load:.2f}")
    print(f"Capacity lower bound: {capacity_lower_bound}")
    print(f"Calculated node capacity: {node_capacity}")

    # Compresses membership patterns.
    # Items that occur in the same fragments always have the same node availability.
    # Example:
    #   item_a -> {fragment_1, fragment_3}
    #   item_b -> {fragment_1, fragment_3}
    # Both items thus always move together.
    # They can be represented by one membership pattern instead of separate variables
    pattern_for_items = {}

    for item_id, containing_fragments in item_fragments.items():
        # Frozenset in order to create an immutable set that can be used as dictionary key
        pattern = frozenset(containing_fragments)

        pattern_for_items.setdefault(pattern, []).append(item_id)

    patterns = list(pattern_for_items.keys())

    # Assigns id to each membership pattern
    pattern_ids = list(range(len(patterns)))

    # Maps pattern id to fragments belonging to pattern
    pattern_fragments = {pattern_id: patterns[pattern_id] for pattern_id in pattern_ids}

    # Pattern weights are calculated by the amount of items represented by pattern
    pattern_weights = {pattern_id: len(pattern_for_items[patterns[pattern_id]]) 
                       for pattern_id in pattern_ids}

    model = pulp.LpProblem("Tuple_Based", pulp.LpMinimize)

    # Constraint (8)
    # x[i][k] = 1 if fragment i is assigned to node k
    # x[i][k] = 0 otherwise
    x = pulp.LpVariable.dicts("x", (fragment_ids, node_ids), cat="Binary")
    
    # Constraint (7)
    # y[k] = 1 if node k is used
    # y[k] = 0 otherwise
    y = pulp.LpVariable.dicts("y", node_ids, cat="Binary")

    # Constraint (9)
    # z[p][k] = 1 if items represented by pattern are available on node k
    z = pulp.LpVariable.dicts("z", (pattern_ids, node_ids), cat="Binary")

    # Calculates load expression of every node
    # pattern_weights[pattern_id] specifies how many items are on pattern.
    # z[p][k] specifies whether items are available on node k
    node_load_expression = {node_id: pulp.lpSum(
        pattern_weights[pattern_id] * z[pattern_id][node_id]
        for pattern_id in pattern_ids)
        for node_id in node_ids}

    model += (pulp.lpSum(y[node_id] for node_id in node_ids), "Minimize_number_of_used_nodes")

    # Constraint (2) 
    # Every fragment must be assigned to exactly one node
    for fragment_id in fragment_ids:
        model += (pulp.lpSum(x[fragment_id][node_id] for node_id in node_ids) == 1,
                  f"Assign_fragment_{fragment_id}_exactly_once")
    
    # Constraint (3)
    # Tuple load of node must not exceed max capacity.
    # y[k] = 0:
    #   means right hand side is zero, node cannot contain data
    # y[k] = 1:
    #   nodes have items up to node capacity
    for node_id in node_ids:
        model += (node_load_expression[node_id] <= node_capacity * y[node_id], 
                  f"Capacity_{node_id}")

    # Calculates lower bound of number of used nodes.
    min_used_nodes = max(replication_factor, math.ceil(min_replicated_weight / node_capacity))

    # adds lower bound to model
    model += (pulp.lpSum(y[node_id] for node_id in node_ids) >= min_used_nodes, 
              "Minimum_used_nodes")

    print(f"Minimum number of used nodes: {min_used_nodes}")

    # Breaks symmetry:
    # Nodes are interchangable. These constraints reduce equivalent solutions
    # and help the solver search more efficiently.
    for curr_node, next_node in zip(node_ids, node_ids[1:]):
        # Used nodes appear before unused nodes.
        # Example: y[node_2] = 1 also requires y[node_1] = 1
        model += (y[curr_node] >= y[next_node], f"Used_node_order_{curr_node}_{next_node}")

        # Sorts node loads in descending order.
        # This means earlier nodes must have >= node load as later nodes.
        model += (node_load_expression[curr_node] >= node_load_expression[next_node], 
                  f"Load_Order_{curr_node}_{next_node}")
        
    

    # Constraint 4 and 5 ensure tuple availability
    # z[p][k] = 1 if at least one fragment containing pattern p is assigned to node k
    for pattern_id, containing_fragments in pattern_fragments.items():
        # Number of fragments that contain the items
        pattern_size = len(containing_fragments)

        for node_id in node_ids:
            # Counts how many fragments belonging to pattern are assigned to node_id
            fragment_sum = pulp.lpSum(x[fragment_id][node_id] 
                                      for fragment_id in containing_fragments)

            # Constraint (4)
            # If at least one containing fragment is assigned to node, fragment_sum is positive.
            # Because z is binary, because of inequality z[p][k] = 1 is forced.
            model += (fragment_sum <= pattern_size * z[pattern_id][node_id],
                        f"Availability_lower_{pattern_id}_{node_id}")

            # Constraint (5)
            # z[p][k] = 1 only if at least one containing fragment is assigned to node.
            # If fragment_sum = 0, then z[p][k] = 0 is being forced.
            model += (z[pattern_id][node_id] <= fragment_sum,
                        f"Availability_upper_{pattern_id}_{node_id}")


    # Constraint (6)
    # Tuple-Level Replication with replication factor m is ensured.
    for pattern_id in pattern_ids:
        model += (pulp.lpSum(z[pattern_id][node_id] 
                            for node_id in node_ids) >= replication_factor,
                            f"Replication_{pattern_id}_{replication_factor}")

    print("\nSolving tuple-based ILP ---")
    print("-----------------------------")

    start_time = time.perf_counter()

    # Configures HiGHS solver
    # msg=True: Display solver log
    # threads=8: allows solver to use up to 8 CPU threads
    # timeLimit=1800: stops solver after 1800 seconds
    # parallel="on": Enables parallel solver execution
    # mip_heuristic_effort=0.5: Increases effort spent on heuristics 
    #                           for finding feasible solutions 
    #                           (finds a feasible solution quicker)
    # mip_heuristic_run_shifting=True: Enables shifting heuristics for quicker 
    #                                  finding feasible solutions
    # mip_heuristic_run_zi_round=True: Enable ZI round heuristics, which attempts 
    #                                  to obtain feasible solutions by rounding a solution
    solver = pulp.HiGHS(msg=True, threads=8, timeLimit=1800, parallel="on", 
                        mip_heuristic_effort=0.5, 
                        mip_heuristic_run_shifting=True, mip_heuristic_run_zi_round=True)

    # Solves the model
    model.solve(solver)

    # Accesses model to retrieve status and MIP information
    highs_model = model.solverModel
    highs_status = highs_model.getModelStatus()
    highs_status_text = highs_model.modelStatusToString(highs_status)
    highs_info = highs_model.getInfo()

    # checks whether variables received a value from solver
    has_values = all(variable.varValue is not None for variable in model.variables())

    solver_result_path = mode_config["solver_result_path"]

    # checks whether solution is feasible if:
    #   1. all variables have values
    #   2. and all constraints are satisfied within tolerance of 1e-6 
    is_feasible_sol = (has_values and model.valid(1e-6))

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

    # Checks whether HiGHS reports an optimal solution for the feasible solution
    is_optimal = is_feasible_sol and highs_status == highspy.HighsModelStatus.kOptimal

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

    # Distinguishes between optimal solution, feasible solution and no feasible solution
    if is_optimal:
        print("Optimality of solution was proven.")

    elif is_feasible_sol:
        print("Feasible solution was found, but optimality could not be proven.")

    else: raise RuntimeError(f"No feasible solution found. Solver status: {highs_status_text}")

    print(f"Used nodes: {used_nodes}")

    assignment_output_path.parent.mkdir(parents=True, exist_ok=True)

    loads_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extracts fragment assignments from solution
    assignment_df = assignments(fragment_ids, fragment_weights, node_ids, x)

    # Calculates load and remaining capacity of every node
    loads_df = compute_loads(fragment_ids, pattern_ids, pattern_weights, 
                             node_ids, x, y, z, node_capacity)

    assignment_df.to_csv(assignment_output_path, index=False)
    loads_df.to_csv(loads_output_path, index=False)

    print(f"Mode: {mode}")
    print(f"Fragment assignment saved: {assignment_output_path}")
    print(f"Node loads saved: {loads_output_path}")

    return assignment_df, loads_df


def main():
    if DATASET not in DATASETS:
        raise ValueError(f"Unknown dataset: {DATASET}")

    config = DATASETS[DATASET]
    process_tuple_ilp(config, MODE)


if __name__ == "__main__":
    main()