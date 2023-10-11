Certainly, here's the documentation in a README format with text:

```markdown
# Linear Programming and Optimization Toolkit

📋 **Table of Contents**
1. [Introduction](#introduction)
2. [Graphical Approach](#graphical-approach)
3. [Simplex Method](#simplex-method)
4. [Minimax Strategy](#minimax-strategy)
5. [Gomory Cutting Plane](#gomory-cutting-plane)
6. [Branch and Bound](#branch-and-bound)
7. [Big M Method](#big-m-method)
8. [Transportation Problem](#transportation-problem)
9. [PERT & CPM](#pert--cpm)

## Introduction
This toolkit provides functions to solve various optimization and linear programming problems. Below are the functions available and examples of how to use them.

## Graphical Approach
- `graphical_solve`: Visualizes the solution of a Linear Programming Problem (LPP) using the graphical method.

Example:
```python
c1 = 3  # Coefficient of x1 in the objective function
c2 = 2  # Coefficient of x2 in the objective function
constraints = [([1, 2], 10, '<='), ([2, 1], 8, '<='), ([1, 1], 5, '<=')]

lpp_solver = TFT(c1, c2, constraints)
optimal_x1, optimal_x2 = lpp_solver.graphical_solve()
print(f'Optimal solution: x1 = {optimal_x1:.2f}, x2 = {optimal_x2:.2f}')
```

## Simplex Method
- `simplex_method`: Solves an LPP using the Simplex method.

Example:
```python
A = np.array([[2, 1], [1, 2]])  # Coefficients of constraints
b = np.array([4, 3])          # Right-hand side values
c = np.array([3, 5])          # Coefficients of the objective function

lpp_solver = TFT(c, constraints)
optimal_val_simplex, solution_simplex = lpp_solver.simplex_method(A, b)
print("Simplex Method - Optimal Value:", optimal_val_simplex)
print("Simplex Method - Optimal Solution:", solution_simplex)
```

## Minimax Strategy
- `minimax_strategy`: Calculates the minimax value and strategy for a two-player zero-sum game using Linear Programming.

Example:
```python
payoff_matrix_game = np.array([[3, 2, 4], [1, 4, 2]]),  # Payoff matrix
constraints_simplex = [([1, 2], 10, '<='), ([2, 1], 8, '<='), ([1, 1], 5, '<=')]

lpp_solver = TFT(c_simplex, constraints_simplex)
minimax_value, minimax_strategy = lpp_solver.minimax_strategy(payoff_matrix_game)

print("Minimax Value:", minimax_value)
print("Minimax Strategy:", minimax_strategy)
```

## Gomory Cutting Plane
- `gomory_cutting_plane`: Applies the Gomory Cutting Plane method to solve an integer linear programming problem.

Example:
```python
A_simplex = np.array([[2, 1], [1, 2]])  # Coefficients of constraints
b_simplex = np.array([4, 3])           # Right-hand side values
c_simplex = np.array([3, 5])           # Coefficients of the objective function

integer_indices = np.array([0, 1])     # Indices of integer variables

lpp_solver = TFT(c_simplex, constraints_simplex)
gomory_optimal_val, gomory_solution = lpp_solver.gomory_cutting_plane(A_simplex, b_simplex, integer_indices)
print("Gomory's Cutting Plane Method - Optimal Value:", gomory_optimal_val)
print("Gomory's Cutting Plane Method - Optimal Solution:", gomory_solution)
```

## Branch and Bound
- `branch_and_bound`: Applies the Branch and Bound method to solve an integer linear programming problem.

Example:
```python
A_simplex = np.array([[2, 1], [1, 2]])  # Coefficients of constraints
b_simplex = np.array([4, 3])           # Right-hand side values
c_simplex = np.array([3, 5])           # Coefficients of the objective function

integer_indices = np.array([0, 1])     # Indices of integer variables

lpp_solver = TFT(c_simplex, constraints_simplex)
bb_optimal_val, bb_solution = lpp_solver.branch_and_bound(A_simplex, b_simplex, integer_indices)
print("Branch and Bound - Optimal Value:", bb_optimal_val)
print("Branch and Bound - Optimal Solution:", bb_solution)
```

## Big M Method
- `big_m_method`: Applies the Big M method to solve a linear programming problem.

Example:
```python
c = np.array([3, 2])  # Coefficients of the objective function
A = np.array([[1

, 2], [2, 1], [1, 1]])  # Coefficients of constraints
b = np.array([10, 8, 5])  # Right-hand side values

lpp_solver = TFT(c, constraints)
optimal_value, optimal_solution = lpp_solver.big_m_method(c, A, b)
print("Big M Method - Optimal Value:", optimal_value)
print("Big M Method - Optimal Solution:", optimal_solution)
```

## Transportation Problem
- `transportation_LCM`: Solves the Transportation Problem using the Least Cost Method (LCM).

Example:
```python
cost_matrix = np.array([[3, 2, 4], [1, 4, 2]])  # Cost matrix
supply = np.array([10, 20])  # Supply at each source
demand = np.array([15, 15, 30])  # Demand at each destination

lpp_solver = TFT(c1, c2, constraints)
allocation = lpp_solver.transportation_LCM(cost_matrix, supply, demand)
print("LCM Allocation:")
print(allocation)
```

- `transportation_NWCR`: Solves the Transportation Problem using the Northwest Corner Rule (NWCR).

Example:
```python
cost_matrix = np.array([[3, 2, 4], [1, 4, 2]])  # Cost matrix
supply = np.array([10, 20])  # Supply at each source
demand = np.array([15, 15, 30])  # Demand at each destination

lpp_solver = TFT(c1, c2, constraints)
allocation = lpp_solver.transportation_NWCR(cost_matrix, supply, demand)
print("NWCR Allocation:")
print(allocation)
```

- `transportation_VAM`: Solves the Transportation Problem using the Vogel's Approximation Method (VAM).

Example:
```python
cost_matrix = np.array([[3, 2, 4], [1, 4, 2]])  # Cost matrix
supply = np.array([10, 20])  # Supply at each source
demand = np.array([15, 15, 30])  # Demand at each destination

lpp_solver = TFT(c1, c2, constraints)
allocation = lpp_solver.transportation_VAM(cost_matrix, supply, demand)
print("VAM Allocation:")
print(allocation)
```

- `transportation_MODI`: Solves the Transportation Problem using the Modified Distribution Method (MODI).

Example:
```python
cost_matrix = np.array([[3, 2, 4], [1, 4, 2]])  # Cost matrix
allocation = np.array([[10, 5, 0], [0, 10, 15]])  # Initial feasible allocation

lpp_solver = TFT(c1, c2, constraints)
modified_allocation = lpp_solver.transportation_MODI(cost_matrix, allocation)
print("MODI Allocation:")
print(modified_allocation)
```

## Hungarian Method
- `hungarian_method`: Solves the Assignment Problem using the Hungarian Algorithm.

Example:
```python
cost_matrix = np.array([[3, 2, 4], [1, 4, 2], [2, 2, 1]])  # Cost matrix

lpp_solver = TFT(c1, c2, constraints)
assignment = lpp_solver.hungarian_method(cost_matrix)
print("Hungarian Method Assignment:")
print(assignment)
```

## PERT & CPM
- `pert_cpm`: Performs Program Evaluation and Review Technique (PERT) and Critical Path Method (CPM) analysis on a set of activities in a project.

Example:
```python
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
```

These functions provide a comprehensive set of tools for solving different types of optimization and project management problems using Linear Programming, Game Theory, and related techniques.
```

Feel free to copy and paste this markdown text into a README.md file in your project.
