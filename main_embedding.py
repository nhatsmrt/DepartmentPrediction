import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from Source import GraphConvNet

np.random.seed(0)


with open("./Data/embeddings.emb", 'r') as f:
    content = np.array(f.readlines()[1:])
    data = np.array(list(map(lambda x: x.split(), content))).astype(np.float32)
    data = data[data[:, 0].argsort()]
    features = data[:, 1:]


departments = (pd.read_csv("./Data/department_labels.csv").sort_values(by = ['id'])[["department_id"]].values).reshape(-1)



indices = np.arange(features.shape[0])
features_train, features_test, departments_train, departments_test, indices_train, indices_test = train_test_split(
    features, departments, indices,
    test_size = 0.2
)
n_train = features_train.shape[0]


# Use k-Fold CV to select model:
kf = KFold(n_splits=3)
accs = []

# models = [
#     LogisticRegression(),
#     SVC(),
#     RandomForestClassifier(),
#     MLPClassifier(max_iter=2000)
# ]
# for model in models:
#     model_acc = []
#     for train_idx, test_idx in kf.split(features_train):
#         model.fit(features_train[train_idx], departments_train[train_idx])
#         model_acc.append(accuracy_score(departments_train[test_idx], model.predict(features_train[test_idx])))
#
#     accs.append(np.mean(model_acc))
#
# print(accs)

## Selecting the best model (here, it is LogisticRegression!)
# model = models[np.argmax(accs)]
# model.fit(features_train, departments_train)
#
# pred = model.predict(features_test)
# print(accuracy_score(departments_test, pred))



df_edges_list = pd.read_csv("./Data/edges_list.csv")
G = nx.convert_matrix.from_pandas_edgelist(
    df = df_edges_list,
    source = "from",
    target = "to"
)
adj_tilde = nx.adjacency_matrix(G).toarray() + np.eye(len(G.nodes), len(G.nodes))
deg_tilde = np.sum(adj_tilde, axis=0)
deg_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(deg_tilde))
normed_laplacian = deg_tilde_inv_sqrt * adj_tilde * deg_tilde_inv_sqrt


# epochs = [100, 250, 500, 750]
# for epoch in epochs:
#     model_acc = []
#     for train_idx, test_idx in kf.split(indices_train):
#         f_train_idx = indices_train[train_idx]
#         f_test_idx = indices_train[test_idx]
#
#         model = GraphConvNet(in_features=features.shape[1], normed_laplacian=normed_laplacian, n_class=42)
#         model.fit(features, departments_train[train_idx], f_train_idx, n_epoch=epoch, verbose = False)
#         prediction = model.predict(features)
#         model_acc.append(accuracy_score(departments_train[test_idx], prediction[f_test_idx]))
#
#     accs.append(np.mean(model_acc))
#
# print(accs)

model = GraphConvNet(in_features = features.shape[1], normed_laplacian = normed_laplacian, n_class = 42)
model.fit(features, departments_train, indices_train, n_epoch = 15000)
#
prediction = model.predict(features)
print(prediction.shape)
print(accuracy_score(departments_test, prediction[indices_test]))

model.save_weight("./Weights")

