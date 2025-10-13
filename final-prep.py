import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ======================================================
# LOAD DATA
# ======================================================

# Load the 'crimerate.csv' dataset (skip header row)
# IMPORTANT: Features assumed standardized (mean 0, std 1)
data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)

[n, p] = np.shape(data)  # number of samples and columns (features + label)

# ======================================================
# SPLIT DATA INTO TRAIN/TEST/β* SETS
# ======================================================

num_train = int(0.01 * n)   # 1% for training (S)
num_test = int(0.25 * n)    # 25% for testing (T)

# S = [0 : num_train)
# Middle for β* = [num_train : n - num_test)
# T = [n - num_test : n)

sample_train = data[0:num_train, 0:-1]         # S features
label_train = data[0:num_train, -1]            # S labels

sample_test = data[n - num_test:, 0:-1]        # T features
label_test = data[n - num_test:, -1]           # T labels

sample_middle = data[num_train:n - num_test, 0:-1]  # For β*
label_middle = data[num_train:n - num_test, -1]

# ======================================================
# COMPUTE β* USING LEAST SQUARES ON THE MIDDLE DATASET
# ======================================================

# Formula: β* = (XᵀX)⁻¹ XᵀY
XT_X_star = sample_middle.T @ sample_middle
XT_Y_star = sample_middle.T @ label_middle
beta_star = np.linalg.inv(XT_X_star) @ XT_Y_star  # pre-trained model β*

# ======================================================
# SET UP CANDIDATE VALUES FOR LAMBDA
# ======================================================

lambda_list = [0.5, 1, 5, 10, 25, 50]

# Store errors for comparison
ls_train_errors = []
ls_test_errors = []

ridge_train_errors = []
ridge_test_errors = []

transfer_train_errors = []
transfer_test_errors = []

# ======================================================
# BASELINE 1: LEAST SQUARES (no λ)
# ======================================================

# Formula: β_LS = (XᵀX)⁻¹ XᵀY
XT_X = sample_train.T @ sample_train
XT_Y = sample_train.T @ label_train
beta_ls = np.linalg.inv(XT_X) @ XT_Y

# Predictions for LS
label_train_pred_ls = sample_train @ beta_ls
label_test_pred_ls = sample_test @ beta_ls

# Compute MSEs
ls_train_errors.append(mean_squared_error(label_train, label_train_pred_ls))
ls_test_errors.append(mean_squared_error(label_test, label_test_pred_ls))

# ======================================================
# LOOP OVER DIFFERENT λ FOR RIDGE + TRANSFER
# ======================================================

for lamda in lambda_list:

    I = np.eye(sample_train.shape[1])          # Identity matrix (p×p)
    XT_X = sample_train.T @ sample_train       # XᵀX
    XT_Y = sample_train.T @ label_train        # XᵀY
    LambdaI = lamda * I                        # λI

    # ------------------------------------------------------
    # BASELINE 2: RIDGE REGRESSION (closed form)
    # β_ridge = (XᵀX + λI)⁻¹ XᵀY
    # ------------------------------------------------------
    beta_ridge = np.linalg.inv(XT_X + LambdaI) @ XT_Y
    label_train_pred_ridge = sample_train @ beta_ridge
    label_test_pred_ridge = sample_test @ beta_ridge

    ridge_train_mse = mean_squared_error(label_train, label_train_pred_ridge)
    ridge_test_mse = mean_squared_error(label_test, label_test_pred_ridge)

    ridge_train_errors.append(ridge_train_mse)
    ridge_test_errors.append(ridge_test_mse)

    # ------------------------------------------------------
    # OUR TRANSFER LEARNING METHOD
    # Objective: J(β) = ||Y - Xβ||² + λ||β - β*||²
    # Closed form: β = (XᵀX + λI)⁻¹ (XᵀY + λβ*)
    # ------------------------------------------------------
    beta_transfer = np.linalg.inv(XT_X + LambdaI) @ (XT_Y + lamda * beta_star)

    label_train_pred_transfer = sample_train @ beta_transfer
    label_test_pred_transfer = sample_test @ beta_transfer

    transfer_train_mse = mean_squared_error(label_train, label_train_pred_transfer)
    transfer_test_mse = mean_squared_error(label_test, label_test_pred_transfer)

    transfer_train_errors.append(transfer_train_mse)
    transfer_test_errors.append(transfer_test_mse)

# ======================================================
# PLOT TEST MSEs FOR COMPARISON
# ======================================================

plt.figure(figsize=(8, 5))
# plt.axhline(y=ls_test_errors[0], color='gray', linestyle='--', label='Least Squares')
plt.plot(lambda_list, ridge_test_errors, '--o', label='Ridge Regression')
plt.plot(lambda_list, transfer_test_errors, '-s', label='Transfer Learning')
plt.xlabel('Lambda (λ)')
plt.ylabel('Test MSE')
plt.title('Model Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()

# ======================================================
# PRINT RESULTS TABLE
# ======================================================

print("=======================================================")
print("         Training and Testing MSE Comparison")
print("=======================================================")
print("Model\t\tλ\tTrain MSE\tTest MSE")
print("-------------------------------------------------------")

# LS result
print(f"Least Squares\t-\t{ls_train_errors[0]:.6f}\t{ls_test_errors[0]:.6f}")

# Ridge and Transfer results
for lam, r_tr, r_te, t_tr, t_te in zip(lambda_list, ridge_train_errors, ridge_test_errors,
                                       transfer_train_errors, transfer_test_errors):
    print(f"Ridge\t\t{lam:.2f}\t{r_tr:.6f}\t{r_te:.6f}")
    print(f"Transfer\t{lam:.2f}\t{t_tr:.6f}\t{t_te:.6f}")

print("=======================================================")
