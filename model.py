import pandas as pd
from ortools.linear_solver import pywraplp

def solve_aggregate_planning():
    months = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun', 'July', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.']
    demand_values = [300.0, 300.0, 350.0, 400.0, 450.0, 500.0, 650.0, 600.0, 475.0, 475.0, 450.0, 450.0]
    T = len(months)
    
    solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')
    if not solver:
        return

    X = [solver.NumVar(0.0, solver.infinity(), f'X_{t}') for t in range(T)]
    I = [solver.NumVar(0.0, solver.infinity(), f'I_{t}') for t in range(T)]
    W = [solver.IntVar(0.0, solver.infinity(), f'W_{t}') for t in range(T)]
    H = [solver.IntVar(0.0, solver.infinity(), f'H_{t}') for t in range(T)]
    F = [solver.IntVar(0.0, solver.infinity(), f'F_{t}') for t in range(T)]

    I_prev = 50.0
    W_prev = 40.0

    for t in range(T):
        d_t = demand_values[t]
        
        solver.Add(X[t] - 10 * W[t] == 0)
        
        if t == 0:
            solver.Add(I[t] - I_prev - X[t] + d_t == 0)
            solver.Add(W[t] - W_prev - H[t] + F[t] == 0)
        else:
            solver.Add(I[t] - I[t-1] - X[t] + d_t == 0)
            solver.Add(W[t] - W[t-1] - H[t] + F[t] == 0)

    objective = solver.Objective()
    for t in range(T):
        objective.SetCoefficient(W[t], 2.0)
        objective.SetCoefficient(H[t], 1.0)
        objective.SetCoefficient(F[t], 2.0)
        objective.SetCoefficient(I[t], 0.06)
    objective.SetMinimization()

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print(f"Total Cost: {objective.Value():.2f}")
        results = []
        for t, month in enumerate(months):
            results.append({
                'Month': month,
                'Demand': demand_values[t],
                'Xt': X[t].solution_value(),
                'It': I[t].solution_value(),
                'Wt': W[t].solution_value(),
                'Ht': H[t].solution_value(),
                'Ft': F[t].solution_value(),
            })
        print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    solve_aggregate_planning()
