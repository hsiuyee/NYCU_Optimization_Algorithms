import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import time

# Load the dataset
data = sio.loadmat('473500_wk.mat')

# Check for the correct key in the loaded data
data_keys = list(data.keys())
print(f"Available keys in the .mat file: {data_keys}")

# 用戶指定需要的鍵名
key_to_use = 'W'

if key_to_use in data_keys:
    a = data[key_to_use]
else:
    raise KeyError(f"The specified key '{key_to_use}' is not found in the .mat file.")

# Convert 'a' to a Torch tensor
a = torch.tensor(a, dtype=torch.float64)  # Use double precision for better accuracy

# Frank-Wolfe Parameters
max_iter = 4500
n, p = a.shape

# Initial Solution
x = torch.ones(p, dtype=torch.float64) / p
x.requires_grad = True

# Step 1: Compute the optimal solution x* using Frank-Wolfe without Line Search
f_values = []
runtime = []
f_star = float('inf')  # 初始設置為無窮大
start_time = time.time()

for iteration in range(max_iter):
    # Compute the function value and gradient
    f_x = -torch.sum(torch.log(torch.matmul(a, x) + 1e-10))  # Add small value to avoid log(0)
    f_x.backward()
    grad = x.grad.data.numpy()

    # 更新最小的函數值
    if f_x.item() < f_star:
        f_star = f_x.item()
        x_star = x.clone().detach()  # 保存當前的最佳解

    # Gurobi Optimization for Linear Minimization Step
    model = gp.Model()
    model.Params.OutputFlag = 0  # Suppress Gurobi output
    model.Params.NumericFocus = 3  # Increase numerical stability
    model.Params.Method = 1  # Use primal simplex
    x_var = model.addMVar(shape=p, lb=0, ub=1, name='x')
    model.setObjective(grad @ x_var, GRB.MINIMIZE)
    model.addConstr(x_var.sum() == 1, name='c')
    model.optimize()

    # Update Step with Fixed Step Size or Away Step
    with torch.no_grad():
        x_new = torch.tensor(x_var.x, dtype=torch.float64)

        # Diminishing step size or fixed step size
        gamma = 2 / (iteration + 2)
        
        x = (1 - gamma) * x + gamma * x_new

    # Record runtime and function value at selected iterations to reduce the number of points
    if iteration % 300 == 0 or iteration == max_iter - 1:  # Plot every 170 iterations for fewer points # 300, 450
        runtime.append(time.time() - start_time)
        f_values.append(f_x.item())

    # Reset gradient for next iteration
    x.requires_grad = True
    if x.grad is not None:
        x.grad.zero_()

# Print the optimal value f(x*)
print(f"Optimal solution f(x*): {f_star}")

# Compute f_values - f_star
relative_errors = [f_value - f_star for f_value in f_values]

# Store runtime and relative_errors as lists of points
points_x = np.array(runtime)
points_y = np.array(relative_errors)

# Remove zero or negative values from points_y for plotting
points_y = np.maximum(points_y, 3 * 1e-6)

# Plot the points and connect them with lines more smoothly
plt.plot(points_x, points_y, 'g-d')  # Plot points with markers
plt.plot(points_x, points_y, 'g-', alpha=0.7)   # Connect the points with smooth lines
plt.xlabel('Time (s)')
plt.ylabel(r'$f(x) - f^*$')
plt.yscale('log')
plt.ylim(1e-6, 1e0)  # Extend y-axis range for better visualization
plt.title(r'real1: $n=473, p=500$')
plt.show()
