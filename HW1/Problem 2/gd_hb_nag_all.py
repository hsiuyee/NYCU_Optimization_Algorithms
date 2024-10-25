import tensorflow as tf
import matplotlib.pyplot as plt
import math


def f(x):
    if x < 1:
        return 25 * x * x
    elif 1 <= x and x <= 2:
        return x * x + 48 * x - 24
    else:
        return 25 * x * x - 48 * x + 72
    # return x * x


def gradient_f(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = f(x)
    return tape.gradient(y, x)


class GD:
    def __init__(self, x_0, epochs, eta):
        self.x_0 = x_0
        self.epochs = epochs
        self.eta = eta

        self.sub_optimality_gap = []
        self.optimal_gap = []
        self.theoretical_gap = []
        self.kappa = 50 / 2

    def optimize(self):
        x_t = self.x_0

        for t in range(0, self.epochs):
            grad = gradient_f(tf.convert_to_tensor(x_t, dtype=tf.float32))
            x_new = x_t - self.eta * grad
            x_t = x_new
            self.sub_optimality_gap.append(f(x_t) - f(0))
            self.optimal_gap.append(abs(x_t - 0))
            self.theoretical_gap.append(((self.kappa - 1) / (self.kappa + 1)) ** t * (abs(self.x_0 - 0) ** 2))
            print(self.x_0)

        return self.sub_optimality_gap, self.optimal_gap, self.theoretical_gap


def gradient_f(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = f(x)
    return tape.gradient(y, x)

class HB:
    def __init__(self, x_0, epochs, eta, theta):
        self.x_0 = x_0
        self.epochs = epochs
        self.eta = eta
        self.theta = theta

        self.sub_optimality_gap = []
        self.optimal_gap = []

    def optimize(self):
        x_t_minus_1 = self.x_0
        x_t = self.x_0

        for t in range(1, self.epochs + 1):
            grad = gradient_f(tf.convert_to_tensor(x_t, dtype=tf.float32))
            x_new = x_t - self.eta * grad + self.theta * (x_t - x_t_minus_1)
            x_t_minus_1 = x_t
            x_t = x_new
            self.sub_optimality_gap.append(f(x_t) - f(0))
            self.optimal_gap.append(abs(x_t - 0))

        return self.sub_optimality_gap, self.optimal_gap

class NAG:
    def __init__(self, x_0, epochs, eta, theta):
        self.x_0 = x_0
        self.epochs = epochs
        self.eta = eta
        self.theta = theta

        self.sub_optimality_gap = []
        self.optimal_gap = []

    def optimize(self):
        x_t_minus_1 = self.x_0
        x_t = self.x_0

        for t in range(1, self.epochs + 1):
            new_theta = (1 + math.sqrt(1 + 4 * theta ** 2)) / 2
            y_t = x_t + (self.theta - 1) / new_theta * (x_t - x_t_minus_1)
            grad = gradient_f(tf.convert_to_tensor(y_t, dtype=tf.float32))
            x_new = x_t - self.eta * grad
            x_t_minus_1 = x_t
            x_t = x_new
            self.theta = new_theta
            self.sub_optimality_gap.append(f(x_t) - f(0))
            self.optimal_gap.append(abs(x_t - 0))

        return self.sub_optimality_gap, self.optimal_gap
    

def plot(GD_sub_optimality_gap, GD_optimal_gap, GD_theoretical_gap, HB_sub_optimality_gap=[], HB_optimal_gap=[], NAG_sub_optimality_gap=[], NAG_optimal_gap=[]):
    plt.figure(figsize=(10, 6))

    plt.plot(GD_sub_optimality_gap, label='GD Sub-optimality Gap', color='red', marker='o')
    plt.plot(GD_optimal_gap, label='GD Optimal Gap', color='orange', marker='s')

    # plt.plot(HB_sub_optimality_gap, label='HB Sub-optimality Gap', color='darkgoldenrod', marker='o')
    # plt.plot(HB_optimal_gap, label='HB Optimal Gap', color='green', marker='s')

    # plt.plot(NAG_sub_optimality_gap, label='NAG Sub-optimality Gap', color='blue', marker='o')
    # plt.plot(NAG_optimal_gap, label='NAG Optimal Gap', color='violet', marker='s')

    plt.title('Sub-optimality Gap and Optimal Gap over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Gap Value')

    plt.legend()
    plt.grid(True)
    plt.show()

def plot2(name, sub_optimality_gap, optimal_gap, theoretical_gap):
    plt.figure(figsize=(10, 6))

    plt.plot(sub_optimality_gap, label=name+' Sub-optimality Gap', color='red', marker='o')
    plt.plot(optimal_gap, label=name+' Optimal Gap', color='orange', marker='s')
    plt.plot(theoretical_gap, label=name+' Theoretical Bound', color='blue', marker='^')

    plt.title(name+' Sub-optimality Gap and Optimal Gap over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Gap Value')

    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    # GD
    x_0 = 3
    epochs = 50
    # eta = 1/50
    eta = 2 / (2 + 50)
    GD_optimizer = GD(x_0, epochs, eta)
    GD_sub_optimality_gap, GD_optimal_gap, GD_theoretical_gap = GD_optimizer.optimize()

    # HB
    x_0 = 3
    epochs = 50
    eta = 1/18
    theta = 4/9
    HB_optimizer = HB(x_0, epochs, eta, theta)
    HB_sub_optimality_gap, HB_optimal_gap = HB_optimizer.optimize()

    # x_0 = 3
    epochs = 50
    eta = 1/50
    theta = 2/3
    NAG_optimizer = NAG(x_0, epochs, eta, theta)
    NAG_sub_optimality_gap, NAG_optimal_gap = NAG_optimizer.optimize()

    # plot(GD_sub_optimality_gap, GD_optimal_gap, HB_sub_optimality_gap, HB_optimal_gap, NAG_sub_optimality_gap, NAG_optimal_gap)
    plot2('GD', GD_sub_optimality_gap, GD_optimal_gap, GD_theoretical_gap)