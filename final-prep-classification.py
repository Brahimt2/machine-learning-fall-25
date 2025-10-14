import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ======================================================
# LOAD DATA
# ======================================================
data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
n, p = data.shape

# ======================================================
# SPLIT DATA
# ======================================================
num_train = int(0.01 * n)  # 1% training
num_test = int(0.25 * n)   # 25% testing

# S = first 1%
X_train = data[0:num_train, :-1]
y_train = data[0:num_train, -1]

# T = last 25%
X_test = data[n - num_test:, :-1]
y_test = data[n - num_test:, -1]

# Middle for pre-trained β*
X_middle = data[num_train:n - num_test, :-1]
y_middle = data[num_train:n - num_test, -1]

# ======================================================
# SIGMOID FUNCTION
# ======================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ======================================================
# GRADIENT DESCENT FUNCTION
# ======================================================
def gradient_descent_logistic(X, y, beta_init, lr=0.1, n_iter=5000, lam=0, beta_star=None):
    beta = beta_init.copy()
    for _ in range(n_iter):
        y_pred = sigmoid(X @ beta)
        grad = X.T @ (y_pred - y)
        if beta_star is None:
            grad += lam * beta        # Ridge logistic
        else:
            grad += lam * (beta - beta_star)  # Transfer logistic
        beta -= lr * grad
    return beta

# ======================================================
# TRAIN PRE-TRAINED β* ON MIDDLE DATASET
# ======================================================
beta_star_init = np.zeros(X_middle.shape[1])
beta_star = gradient_descent_logistic(X_middle, y_middle, beta_star_init, lr=0.1, n_iter=5000)

# ======================================================
# HYPERPARAMETERS
# ======================================================
lambda_list = [0.01, 0.1, 1, 5, 10]
learning_rate = 0.1
n_iter = 5000

# Store errors (1 - accuracy)
ridge_train_err = []
ridge_test_err = []
transfer_train_err = []
transfer_test_err = []

# ======================================================
# LEAST SQUARES-LIKE BASELINE LOGISTIC (no reg)
# ======================================================
beta_ls_init = np.zeros(X_train.shape[1])
beta_ls = gradient_descent_logistic(X_train, y_train, beta_ls_init, lr=learning_rate, n_iter=n_iter, lam=0)

ls_train_err = 1 - accuracy_score(y_train, sigmoid(X_train @ beta_ls) >= 0.5)
ls_test_err = 1 - accuracy_score(y_test, sigmoid(X_test @ beta_ls) >= 0.5)

# ======================================================
# LOOP OVER LAMBDA VALUES FOR RIDGE AND TRANSFER
# ======================================================
for lam in lambda_list:
    # Ridge Logistic Regression
    beta_ridge_init = np.zeros(X_train.shape[1])
    beta_ridge = gradient_descent_logistic(X_train, y_train, beta_ridge_init,
                                           lr=learning_rate, n_iter=n_iter, lam=lam)
    ridge_train_err.append(1 - accuracy_score(y_train, sigmoid(X_train @ beta_ridge) >= 0.5))
    ridge_test_err.append(1 - accuracy_score(y_test, sigmoid(X_test @ beta_ridge) >= 0.5))

    # Transfer Logistic Regression
    beta_transfer_init = np.zeros(X_train.shape[1])
    beta_transfer = gradient_descent_logistic(X_train, y_train, beta_transfer_init,
                                              lr=learning_rate, n_iter=n_iter, lam=lam, beta_star=beta_star)
    transfer_train_err.append(1 - accuracy_score(y_train, sigmoid(X_train @ beta_transfer) >= 0.5))
    transfer_test_err.append(1 - accuracy_score(y_test, sigmoid(X_test @ beta_transfer) >= 0.5))

# ======================================================
# PLOT TESTING ERROR VS LAMBDA
# ======================================================
plt.figure(figsize=(8,5))
plt.plot(lambda_list, ridge_test_err, '--o', label='Ridge Logistic')
plt.plot(lambda_list, transfer_test_err, '-s', label='Transfer Logistic')
plt.xlabel('Lambda (λ)')
plt.ylabel('Testing Error (1 - Accuracy)')
plt.title('Testing Error Comparison vs Lambda')
plt.legend()
plt.grid(True)
plt.show()

# ======================================================
# PRINT RESULTS TABLE
# ======================================================
print("=======================================================")
print("         Training and Testing Error Comparison")
print("=======================================================")
print("Model\t\tλ\tTrain Err\tTest Err")
print("-------------------------------------------------------")
print(f"Logistic\t-\t{ls_train_err:.4f}\t{ls_test_err:.4f}")
for lam, r_tr, r_te, t_tr, t_te in zip(lambda_list, ridge_train_err, ridge_test_err,
                                       transfer_train_err, transfer_test_err):
    print(f"Ridge\t\t{lam:.2f}\t{r_tr:.4f}\t{r_te:.4f}")
    print(f"Transfer\t{lam:.2f}\t{t_tr:.4f}\t{t_te:.4f}")
print("=======================================================")
