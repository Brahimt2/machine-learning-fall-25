# hw5_als.py
import numpy as np
import matplotlib.pyplot as plt

# configuration
k = 20               # latent dimension
lambda1 = 0.1        # regularization for U
lambda2 = 0.1        # regularization for V
max_updates = 30     # number of ALS iterations
seed = 42            # reproducibility

train_file = 'rate_train.csv'
test_file = 'rate_test.csv'
figure_file = 'figure/task1_rmse.png'

def load_ratings(file_path):
    """Load space-separated CSV, return list of triples (i,j,r) and matrix shape."""
    data = np.loadtxt(file_path, delimiter=',')
    triples = []
    num_users, num_items = data.shape
    for i in range(num_users):
        for j in range(num_items):
            if data[i, j] != 0:
                triples.append((i, j, data[i, j]))
    return triples, num_users, num_items

def compute_rmse(U, V, triples):
    """Compute RMSE over given set of triples."""
    se = 0.0
    for i, j, r in triples:
        pred = np.dot(U[i], V[:, j])
        se += (r - pred) ** 2
    return np.sqrt(se / len(triples))

def als(train_triples, num_users, num_items, k, lambda1, lambda2, max_updates):
    np.random.seed(seed)
    U = np.random.normal(0, 0.1, (num_users, k))
    V = np.random.normal(0, 0.1, (k, num_items))

    # build user->items and item->users dictionaries
    user_rated = {i: [] for i in range(num_users)}
    item_rated = {j: [] for j in range(num_items)}
    M = np.zeros((num_users, num_items))
    for i, j, r in train_triples:
        user_rated[i].append(j)
        item_rated[j].append(i)
        M[i, j] = r

    rmse_history = []

    for iteration in range(1, max_updates + 1):
        # update U
        for i in range(num_users):
            Js = user_rated[i]
            if not Js:
                continue
            # V_j shape k x |Js|
            Vj = V[:, Js]
            # M_j shape |Js|
            Mj = M[i, Js]
            # A = V_j V_j^T + lambda1 * I
            A = Vj @ Vj.T + lambda1 * np.eye(k)
            # b = V_j M_j
            b = (Vj * Mj).sum(axis=1)
            U[i] = np.linalg.solve(A, b)

        # update V
        for j in range(num_items):
            Is = item_rated[j]
            if not Is:
                continue
            # U_i shape k x |Is|
            Ui = U[Is, :].T
            # M_i shape |Is|
            Mi = M[Is, j]
            # A = U_i U_i^T + lambda2 * I
            A = Ui @ Ui.T + lambda2 * np.eye(k)
            # b = U_i M_i
            b = (Ui * Mi).sum(axis=1)
            V[:, j] = np.linalg.solve(A, b)

        # compute RMSE on test set
        rmse = compute_rmse(U, V, test_triples)
        rmse_history.append(rmse)
        print(f"Iteration {iteration}: Test RMSE = {rmse:.4f}")

    return U, V, rmse_history

if __name__ == "__main__":
    train_triples, num_users, num_items = load_ratings(train_file)
    test_triples, _, _ = load_ratings(test_file)

    print(f"Number of users: {num_users}, items: {num_items}")
    print(f"Number of training entries: {len(train_triples)}, test entries: {len(test_triples)}")

    U, V, rmse_history = als(train_triples, num_users, num_items, k, lambda1, lambda2, max_updates)

    # plot RMSE vs ALS updates
    plt.figure(figsize=(8,6))
    plt.plot(range(1, max_updates + 1), rmse_history, marker='o')
    plt.xlabel("ALS Updates")
    plt.ylabel("Test RMSE")
    plt.title(f"Testing Error versus ALS Updates with k = {k}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()