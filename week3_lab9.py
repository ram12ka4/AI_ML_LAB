# Optional Lab - Regularized Cost and Gradient

# Goals

# In this lab, you will:

#     extend the previous linear and logistic cost functions with a regularization term.
#     rerun the previous example of over-fitting with a regularization term added.


import numpy as np
import matplotlib.pyplot as plt

plt.style.use("./deeplearning.mplstyle")
from plt_overfit import overfit_example, output
from lab_utils_common import sigmoid

# Compute Regularization Cost for Linear Regression


def compute_cost_linear_reg(X, y, w, b, lambda_="1"):
    m, n = X.shape
    cost = 0

    for i in range(m):
        f_wb_i = np.dot(w, X[i]) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)

    reg_cost = 0

    for j in range(n):
        reg_cost = reg_cost + (w[j] ** 2)
    reg_cost = reg_cost * (lambda_ / (2 * m))

    total_cost = cost + reg_cost

    return total_cost


# Calling compute_cost_linear_reg

np.random.seed(1)
X_tmp = np.random.rand(5, 6)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = (
    np.random.rand(X_tmp.shape[1]).reshape(
        -1,
    )
    - 0.5
)
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print(f"Regularised Cost@=={cost_tmp}")

# Compute Regularized Cost for Logistic Regression


def compute_cost_logistic_reg(X, y, w, b, lambda_="1"):

    m, n = X.shape
    cost = 0

    for i in range(m):
        z_i = np.dot(w, X[i]) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost = cost / m

    reg_cost = 0

    for j in range(n):
        reg_cost = reg_cost + (w[j] ** 2)
    reg_cost = reg_cost * (lambda_ / (2 * m))

    total_cost = cost + reg_cost

    return total_cost


# Calling Compute Regularized Cost for Logistic Regression

np.random.seed(1)
X_tmp = np.random.rand(5, 6)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = (
    np.random.rand(X_tmp.shape[1]).reshape(
        -1,
    )
    - 0.5
)
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print(f"Regularised Cost for logistic regression@=={cost_tmp}")

# Gradient function for regularized linear regression


def compute_gradient_linear_reg(X, y, w, b, lambda_="1"):

    m, n = X.shape

    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        err = (np.dot(w, X[i]) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]

    return dj_dw, dj_db


# Calling

np.random.seed(1)
X_tmp = np.random.rand(5, 3)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_dw_tmp, dj_db_tmp = compute_gradient_linear_reg(
    X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp
)

print(
    f"dj_db: {dj_db_tmp}",
)
print(
    f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}",
)


def compute_gradient_logistic_reg(X, y, w, b, lambda_="1"):

    m, n = X.shape

    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(w, X[i]) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i, j]
        dj_db += err_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]

    return dj_dw, dj_db


# Calling

np.random.seed(1)
X_tmp = np.random.rand(5, 3)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_dw_tmp, dj_db_tmp = compute_gradient_logistic_reg(
    X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp
)

print(
    f"dj_db: {dj_db_tmp}",
)
print(
    f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}",
)
