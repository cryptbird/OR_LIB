import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from scipy.optimize import linprog

class TFT:
    def __init__(self, c1, c2, constraints):
        self.c1 = c1
        self.c2 = c2
        self.constraints = constraints

    def graphical_solve(self):
        # Create a figure for plotting
        plt.figure()
        
        # Create a grid for plotting the feasible region
        x1 = np.linspace(0, 20, 400)
        x2 = np.linspace(0, 20, 400)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Plot the objective function: c1*x1 + c2*x2
        Z = self.c1 * X1 + self.c2 * X2
        plt.contour(X1, X2, Z, levels=20, colors='gray')
        plt.xlabel('x1')
        plt.ylabel('x2')
        
        # Plot the feasible region
        for A, B, op in self.constraints:
            if op == '<=':
                plt.fill_between(x1, 0, (B - A[0] * x1) / A[1], where=((B - A[0] * x1) / A[1] >= 0), alpha=0.5)
            elif op == '>=':
                plt.fill_between(x1, 0, (B - A[0] * x1) / A[1], where=((B - A[0] * x1) / A[1] <= 0), alpha=0.5)
            elif op == '==':
                plt.plot(x1, (B - A[0] * x1) / A[1], label=f'{A[0]}x1 + {A[1]}x2 = {B}', linestyle='--')
        
        plt.legend(loc='upper right')
        
        # Find and plot the optimal solution
        obj_values = self.c1 * x1 + self.c2 * x2
        optimal_idx = np.argmax(obj_values)
        optimal_x1 = x1[optimal_idx]
        optimal_x2 = x2[optimal_idx]
        plt.plot(optimal_x1, optimal_x2, 'ro', label=f'Optimal: ({optimal_x1:.2f}, {optimal_x2:.2f})')
        plt.legend(loc='upper right')
        
        plt.grid(True)
        plt.title('Graphical Approach for Linear Programming')
        plt.show()
        
        return optimal_x1, optimal_x2
      
    def simplex_method(self, A, b):
        c = [-1] * len(A[0])
        result = linprog(c, A_ub=A, b_ub=b, method='highs')
        return result.fun, result.x

    def minimax_strategy(self, payoff_matrix):
        m, n = payoff_matrix.shape
        c = [-1] * n
        A_ub = np.transpose(-payoff_matrix)
        b_ub = [-1] * m

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
        return result.fun, result.x
    
    def gomory_cutting_plane(self, A, b, integer_indices):
        integer_solution = None

        while integer_solution is None:
            # Solve the LP relaxation
            c = [-1] * len(A[0])
            result = linprog(c, A_ub=A, b_ub=b, method='highs')
            fractional_values = result.x

            # Check for integer solution
            is_integer_solution = all(np.isclose(fractional_values[integer_indices], np.round(fractional_values[integer_indices])))
            
            if is_integer_solution:
                integer_solution = fractional_values
            else:
                # Add Gomory cut
                cut_coefficients = np.round(fractional_values[integer_indices]) - fractional_values[integer_indices]
                A = np.vstack([A, cut_coefficients])
                b = np.append(b, 0)
                integer_indices = np.append(integer_indices, len(cut_coefficients) - 1)

        return result.fun, integer_solution

    def branch_and_bound(self, A, b, integer_indices):
        def solve_ilp_subproblem(A_subproblem, b_subproblem, integer_indices_subproblem, incumbent):
            # Solve ILP subproblem with the current set of constraints
            c = [-1] * len(A_subproblem[0])
            result = linprog(c, A_ub=A_subproblem, b_ub=b_subproblem, method='highs')
            objective_value = result.fun
            integer_solution = result.x

            # Check for integer solution
            is_integer_solution = all(np.isclose(integer_solution[integer_indices_subproblem], np.round(integer_solution[integer_indices_subproblem]))

            if is_integer_solution:
                if objective_value < incumbent[0]:
                    incumbent[0] = objective_value
                    incumbent[1] = integer_solution
                return

            # Find the fractional variable with the largest deviation
            fractional_indices = np.where(~np.isclose(integer_solution[integer_indices_subproblem], np.round(integer_solution[integer_indices_subproblem])))[0]
            max_deviation_index = np.argmax(np.abs(integer_solution[integer_indices_subproblem] - np.round(integer_solution[integer_indices_subproblem])))
            chosen_variable = fractional_indices[max_deviation_index]

            # Branch into two subproblems with additional constraints
            subproblem1 = A_subproblem.copy()
            subproblem2 = A_subproblem.copy()
            b_subproblem1 = b_subproblem.copy()
            b_subproblem2 = b_subproblem.copy()

            subproblem1 = np.vstack([subproblem1, [0] * len(subproblem1[0])])
            subproblem2 = np.vstack([subproblem2, [0] * len(subproblem2[0])])

            subproblem1[-1, chosen_variable] = 1
            subproblem2[-1, chosen_variable] = -1

            b_subproblem1 = np.append(b_subproblem1, 0)
            b_subproblem2 = np.append(b_subproblem2, 0)

            solve_ilp_subproblem(subproblem1, b_subproblem1, integer_indices_subproblem, incumbent)
            solve_ilp_subproblem(subproblem2, b_subproblem2, integer_indices_subproblem, incumbent)

        incumbent = [np.inf, None]
        solve_ilp_subproblem(A, b, integer_indices, incumbent)

        return incumbent[0], incumbent[1]
                                      
    def big_m_method(self, c, A, b):
        m = len(b)
        n = len(c)
        M = 1e6  # A large positive number

        # Convert the maximization problem into a minimization problem
        c = [-x for x in c]

        # Introduce surplus variables for inequality constraints
        A_surplus = np.column_stack((A, np.eye(m)))
        c_surplus = c + [0] * m

        # Add artificial variables for less than or equal constraints
        A_artificial = np.column_stack((A_surplus, -np.eye(m)))
        c_artificial = c_surplus + [M] * m

        # Initialize the tableau
        tableau = np.vstack((A_artificial, c_artificial))
        b_extended = np.hstack((b, [0] * m))

        while np.any(tableau[-1, :-1] > 0):
            entering_column = np.argmax(tableau[-1, :-1])

            if all(tableau[:-1, entering_column] <= 0):
                raise Exception("Unbounded solution")

            # Find the departing variable with the minimum ratio
            departing_variable = np.argmin(b_extended / tableau[:-1, entering_column])

            # Pivot the tableau
            pivot_element = tableau[departing_variable, entering_column]
            tableau[departing_variable, :] /= pivot_element

            for i in range(len(tableau)):
                if i != departing_variable:
                    pivot_ratio = tableau[i, entering_column]
                    tableau[i, :] -= pivot_ratio * tableau[departing_variable, :]

        # Get the optimal solution and value
        optimal_solution = tableau[-1, -1]
        optimal_solution_values = tableau[-1, -1-m:-1]

        return -optimal_solution, optimal_solution_values[:n]

    def transportation_LCM(self, cost_matrix, supply, demand):
        allocation = np.zeros(cost_matrix.shape)
        i, j = 0, 0

        while i < len(supply) and j < len(demand):
            min_supply = min(supply[i], demand[j])
            allocation[i][j] = min_supply
            supply[i] -= min_supply
            demand[j] -= min_supply

            if supply[i] == 0:
                i += 1
            else:
                j += 1

        return allocation
                              
    def transportation_NWCR(self, cost_matrix, supply, demand):
        m, n = cost_matrix.shape
        allocation = np.zeros((m, n))
        i, j = 0, 0

        while i < m and j < n:
            quantity = min(supply[i], demand[j])
            allocation[i, j] = quantity
            supply[i] -= quantity
            demand[j] -= quantity

            if supply[i] == 0:
                i += 1
            else:
                j += 1

        return allocation

    def transportation_VAM(self, cost_matrix, supply, demand):
        m, n = cost_matrix.shape
        allocation = np.zeros((m, n))
        supply_copy = supply.copy()
        demand_copy = demand.copy()

        while np.sum(supply_copy) > 0 and np.sum(demand_copy) > 0:
            u = np.zeros(m)  # Row penalties
            v = np.zeros(n)  # Column penalties

            # Step 1: Calculate row and column penalties
            for i in range(m):
                if np.sum(allocation[i, :]) == 0:
                    min1 = np.partition(cost_matrix[i, :], 2)[:2]
                    u[i] = min1[1] - min1[0]

            for j in range(n):
                if np.sum(allocation[:, j]) == 0:
                    min2 = np.partition(cost_matrix[:, j], 2)[:2]
                    v[j] = min2[1] - min2[0]

            # Step 2: Find the cell with the maximum penalty
            max_penalty = -1
            max_i, max_j = -1, -1

            for i in range(m):
                for j in range(n):
                    if allocation[i, j] == 0:
                        penalty = u[i] + v[j] - cost_matrix[i, j]
                        if penalty > max_penalty:
                            max_penalty = penalty
                            max_i, max_j = i, j

            # Step 3: Allocate as much as possible
            quantity = min(supply_copy[max_i], demand_copy[max_j])
            allocation[max_i, max_j] = quantity
            supply_copy[max_i] -= quantity
            demand_copy[max_j] -= quantity

        return allocation
           
    def transportation_MODI(self, cost_matrix, allocation):
        m, n = cost_matrix.shape

        # Compute initial u and v values using Northwest Corner Rule
        u = np.zeros(m)
        v = np.zeros(n)

        # Step 1: Calculate v values
        for j in range(n):
            indices = np.where(allocation[:, j] > 0)[0]
            if len(indices) == 1:
                v[j] = cost_matrix[indices[0], j]

        while True:
            delta = np.zeros(cost_matrix.shape)
            for i in range(m):
                for j in range(n):
                    if allocation[i, j] == 0:
                        delta[i, j] = u[i] + v[j] - cost_matrix[i, j]

            row_indices, col_indices = linear_sum_assignment(delta)
            min_delta = delta[row_indices, col_indices].min()

            if min_delta >= 0:
                break

            for i in range(m):
                for j in range(n):
                    if (i, j) == (row_indices[i], col_indices[i]):
                        allocation[i, j] = 1
                    else:
                        allocation[i, j] = 0

            v.fill(0)
            u.fill(0)

            for j in range(n):
                indices = np.where(allocation[:, j] > 0)[0]
                if len(indices) == 1:
                    v[j] = cost_matrix[indices[0], j]

            for i in range(m):
                indices = np.where(allocation[i, :] > 0)[0]
                if len(indices) == 1:
                    u[i] = cost_matrix[i, indices[0]]

        return allocation

    def hungarian_method(self, cost_matrix):
        m, n = cost_matrix.shape

        # Step 1: Subtract the minimum value of each row from the row
        for i in range(m):
            min_row_val = np.min(cost_matrix[i, :])
            cost_matrix[i, :] -= min_row_val

        # Step 2: Subtract the minimum value of each column from the column
        for j in range(n):
            min_col_val = np.min(cost_matrix[:, j])
            cost_matrix[:, j] -= min_col_val

        # Step 3: Find the minimum number of lines to cover all the zeros
        row_covered = np.zeros(m, dtype=bool)
        col_covered = np.zeros(n, dtype=bool)

        while True:
            # Find the minimum uncovered value
            min_uncovered_val = np.inf

            for i in range(m):
                if not row_covered[i]:
                    for j in range(n):
                        if not col_covered[j]:
                            if cost_matrix[i, j] < min_uncovered_val:
                                min_uncovered_val = cost_matrix[i, j]

            if min_uncovered_val == np.inf:
                break

            # Subtract the minimum value from all uncovered elements
            for i in range(m):
                for j in range(n):
                    if not row_covered[i] and not col_covered[j]:
                        cost_matrix[i, j] -= min_uncovered_val

            # Cover rows and columns with all zeros
            for i in range(m):
                if not row_covered[i]:
                    if np.count_nonzero(cost_matrix[i, :] == 0) > 0:
                        j = np.where(cost_matrix[i, :] == 0)[0][0]
                        col_covered[j] = True
                        row_covered[i] = True

        # Step 4: Find the minimum number of lines required to cover all zeros
        num_lines = np.sum(row_covered) + np.sum(col_covered)

        if num_lines < m:
            # Step 5: Find the smallest uncovered element
            min_uncovered_val = np.inf

            for i in range(m):
                if not row_covered[i]:
                    for j in range(n):
                        if not col_covered[j]:
                            if cost_matrix[i, j] < min_uncovered_val:
                                min_uncovered_val = cost_matrix[i, j]

            # Subtract the minimum uncovered value from all uncovered elements
            for i in range(m):
                for j in range(n):
                    if not row_covered[i] and not col_covered[j]:
                        cost_matrix[i, j] -= min_uncovered_val

            # Add the minimum value to the intersections of the lines
            for i in range(m):
                for j in range(n):
                    if row_covered[i] and col_covered[j]:
                        cost_matrix[i, j] += min_uncovered_val

            return self.hungarian_method(cost_matrix)
        else:
            # Construct the assignment from the covered cells
            assignment = np.zeros((m, n), dtype=int)
            for i in range(m):
                if np.sum(assignment[i, :]) == 0:
                    j = np.where(cost_matrix[i, :] == 0)[0][0]
                    assignment[i, j] = 1

            return assignment
    def pert_cpm(self, activities):
        G = nx.DiGraph()
        for activity in activities:
            G.add_node(activity["name"], duration=activity["duration"])

        for activity in activities:
            for successor in activity["successors"]:
                G.add_edge(activity["name"], successor)

        # Calculate earliest start times
        earliest_start_times = {}
        for node in nx.topological_sort(G):
            if len(G.pred[node]) == 0:
                earliest_start_times[node] = 0
            else:
                earliest_start_times[node] = max([earliest_start_times[predecessor] + G[predecessor][node]["duration"] for predecessor in G.pred[node]])

        # Calculate latest start times
        latest_start_times = {}
        for node in nx.topological_sort(G):
            if len(G.succ[node]) == 0:
                latest_start_times[node] = earliest_start_times[node]
            else:
                latest_start_times[node] = min([latest_start_times[successor] - G[node][successor]["duration"] for successor in G.succ[node]])

        # Calculate slack times
        slack_times = {}
        for node in G.nodes:
            slack_times[node] = latest_start_times[node] - earliest_start_times[node]

        # Calculate critical path
        critical_path = [node for node, slack in slack_times.items() if slack == 0]

        return {
            "earliest_start_times": earliest_start_times,
            "latest_start_times": latest_start_times,
            "slack_times": slack_times,
            "critical_path": critical_path
        }                                
# Example Graphical usage:
c1 = 3  # Coefficient of x1 in the objective function
c2 = 2  # Coefficient of x2 in the objective function
constraints = [([1, 2], 10, '<='), ([2, 1], 8, '<='), ([1, 1], 5, '<=')]

lpp_solver = TFT(c1, c2, constraints)
optimal_x1, optimal_x2 = lpp_solver.graphical_solve()
print(f'Optimal solution: x1 = {optimal_x1:.2f}, x2 = {optimal_x2:.2f}')

# Example usage of the class for Simplex and Game Theory
A_simplex = np.array([[2, 1], [1, 2]])
b_simplex = np.array([4, 3])
c_simplex = np.array([3, 5])

payoff_matrix_game = np.array([[3, 2, 4], [1, 4, 2]])
constraints_simplex = [([1, 2], 10, '<='), ([2, 1], 8, '<='), ([1, 1], 5, '<=')]

lpp_solver = TFT(c_simplex, constraints_simplex)
optimal_val_simplex, solution_simplex = lpp_solver.simplex_method(A_simplex, b_simplex)
minimax_value, minimax_strategy = lpp_solver.minimax_strategy(payoff_matrix_game)

print("Simplex Method - Optimal Value:", optimal_val_simplex)
print("Simplex Method - Optimal Solution:", solution_simplex)
print("Minimax Value:", minimax_value)
print("Minimax Strategy:", minimax_strategy)

gomory_optimal_val, gomory_solution = lpp_solver.gomory_cutting_plane(A_simplex, b_simplex, integer_indices)
bb_optimal_val, bb_solution = lpp_solver.branch_and_bound(A_simplex, b_simplex, integer_indices)
print("Gomory's Cutting Plane Method - Optimal Value:", gomory_optimal_val)
print("Gomory's Cutting Plane Method - Optimal Solution:", gomory_solution)
print("Branch and Bound - Optimal Value:", bb_optimal_val)
print("Branch and Bound - Optimal Solution:", bb_solution)

cost_matrix = np.array([[3, 2, 4], [1, 4, 2]])
supply = np.array([10, 20])
demand = np.array([15, 15, 30])

lpp_solver = TFT(c1, c2, constraints)
nwcr_allocation = lpp_solver.transportation_NWCR(cost_matrix, supply, demand)
vam_allocation = lpp_solver.transportation_VAM(cost_matrix, supply, demand)

print("NWCR Allocation:")
print(nwcr_allocation)
print("VAM Allocation:")
print(vam_allocation)
cost_matrix_hungarian = np.array([[3, 2, 4], [1, 4, 2], [2, 2, 1]])
cost_matrix_transportation = np.array([[3, 2, 4], [1, 4, 2]])
supply_transportation = np.array([10, 20])
demand_transportation = np.array([15, 15, 30])

lpp_solver = TFT(c1, c2, constraints)
modi_allocation = lpp_solver.transportation_MODI(cost_matrix_transportation, nwcr_allocation.copy())
hungarian_allocation = lpp_solver.hungarian_method(cost_matrix_hungarian)

print("MODI Allocation:")
print(modi_allocation)
print("Hungarian Method Allocation:")
print(hungarian_allocation)

activities_pert_cpm = [
    {"name": "A", "duration": 4, "successors": ["B", "C"]},
    {"name": "B", "duration": 2, "successors": ["D"]},
    {"name": "C", "duration": 3, "successors": ["D"]},
    {"name": "D", "duration": 5, "successors": []}
]

lpp_solver = TFT(c1, c2, constraints)
pert_cpm_result = lpp_solver.pert_cpm(activities_pert_cpm)

print("PERT & CPM Result:")
print("Earliest Start Times:", pert_cpm_result["earliest_start_times"])
print("Latest Start Times:", pert_cpm_result["latest_start_times"])
print("Slack Times:", pert_cpm_result["slack_times"])
print("Critical Path:", pert_cpm_result["critical_path"])
