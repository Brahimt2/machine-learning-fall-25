import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)
num_train = int(0.75*n)
num_test = int(0.25*n)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# CkNN parameters
k = 3
m_values = [50, 100, 200, 400, num_train]
er_test_cknn = []

for m in m_values:
    # randomly select m points from training set
    idx = np.random.choice(num_train, m, replace=False)
    train_subset = sample_train[idx]
    label_subset = label_train[idx]

    # predict for test set
    test_preds = []
    for i in range(sample_test.shape[0]):
        distances = np.linalg.norm(train_subset - sample_test[i], axis=1)
        neighbors_idx = np.argsort(distances)[:k]
        neighbor_labels = label_subset[neighbors_idx]
        pred = np.round(np.mean(neighbor_labels))
        test_preds.append(pred)
    
    test_preds = np.array(test_preds)
    er_test_cknn.append(np.mean(test_preds != label_test))

# plot testing error vs compression rate
compression_rate = np.array(m_values) / num_train
plt.figure()
plt.plot(compression_rate, er_test_cknn, marker='o', linestyle='-', linewidth=2)
plt.xlabel('Compression Rate (m/n)')
plt.ylabel('Testing Error')
plt.title('CkNN Testing Error vs Compression Rate')
plt.grid(True)
plt.show()
