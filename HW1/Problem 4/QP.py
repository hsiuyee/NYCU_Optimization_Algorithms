import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve(u):
    model = gp.Model('Markowitz Portfolio Optimization problem with u=' + str(u))

    n = 4 # dimension
    lb = -GRB.INFINITY
    ub = GRB.INFINITY
    x = model.addMVar(shape=n, lb=lb, ub=ub, name="x")

    p = np.array([0.12, 0.1, 0.07, 0.03])
    all_1_matrix = np.ones(4)
    sigma = np.array([[0.2, -0.03, 0, 0], 
                      [-0.03, 0.1, -0.02, 0], 
                      [0, -0.02, 0.05, 0], 
                      [0, 0, 0, 0.01]])

    # Define the objective: minimize -p.T @ x + u * x.T @ sigma @ x
    linear_term = -p @ x
    quadratic_term = u * (x @ sigma @ x)
    obj = linear_term + quadratic_term
    model.setObjective(obj, GRB.MINIMIZE)

    # Add constraints
    model.addConstr(all_1_matrix @ x == 1, "constraint_1")
    model.addConstr(x >= 0, "constraint_2")

    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"u = {u}")
        print("Optimal solution (x):", x.X)
        print("Optimal objective value:", model.objVal)
    else:
        print(f"u = {u}")
        print("No optimal solution found.")

if __name__ == "__main__":
    u_matrix = [0, 0.1, 1.0, 2.0, 5.0, 10.0]
    for u in u_matrix:
        solve(u)
