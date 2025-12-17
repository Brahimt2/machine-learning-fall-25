# hw5_k.py
import numpy as np
import matplotlib.pyplot as plt

# configuration
k_values = [3, 5, 7, 15, 30]  # latent dimensions to test
lambda1 = 0.1
lambda2 = 0.1
max_updates = 30
seed = 42

train_file = 'rate_train.csv'
test_file = 'rate_test.csv'

def load_ratings(file_path):
    """Load comma-separated CSV, return list of triples (i,j,r) and matrix shape."""
    data = np.loadtxt(file_path, delimiter=',')
    triples = []
    num_users, num_items = data.shape
    for i in range(num_users):
        for j in range(num_items):
            if data[i, j] != 0:
                triples.append((i, j, data[i, j]))
    return triples, num_users, num_items

def compute_rmse(U, V, triples):
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

    for iteration in range(max_updates):
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

    return U, V

if __name__ == "__main__":
    train_triples, num_users, num_items = load_ratings(train_file)
    test_triples, _, _ = load_ratings(test_file)

    rmse_results = []

    print(f"Number of users: {num_users}, items: {num_items}")
    print(f"Number of training entries: {len(train_triples)}, test entries: {len(test_triples)}\n")

    for k in k_values:
        print(f"Running ALS for k = {k} ...")
        U, V = als(train_triples, num_users, num_items, k, lambda1, lambda2, max_updates)
        rmse = compute_rmse(U, V, test_triples)
        rmse_results.append(rmse)
        print(f"Final Test RMSE for k={k}: {rmse:.4f}\n")

    # plot RMSE vs k
    plt.figure(figsize=(8,6))
    plt.plot(k_values, rmse_results, marker='o')
    plt.xlabel("k (latent dimension)")
    plt.ylabel("Test RMSE")
    plt.title("Testing Error versus k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()