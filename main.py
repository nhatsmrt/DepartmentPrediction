import numpy as np
import pandas as pd
import networkx as nx
from Source import GraphConvNet
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score



df_edges_list = pd.read_csv("./Data/edges_list.csv")
# labels = pd.read_csv("./Data/department_labels.csv")[["department_id"]].values - 1
df_train = pd.read_csv("./Data/train.csv").fillna(0)
df_test = pd.read_csv("./Data/test.csv").fillna(0)
df_features = pd.concat((df_train, df_test))

features = df_features.drop(["name", "department", "true_department"], axis = 1).values
departments = (df_features[["true_department"]].values).reshape(-1) - 1
n_test = df_test.shape[0]
n_train = df_train.shape[0]

features[:, :4] = features[:, :4] - 1


transformer = ColumnTransformer(
    [
        ("oh_0", OneHotEncoder(categories = [range(45)]), [0]),
        ("oh_1", OneHotEncoder(categories = [range(32)]), [1]),
        ("oh_2", OneHotEncoder(categories = [range(136)]), [2]),
        ("oh_3", OneHotEncoder(categories = [range(26)]), [3]),

    ]
)
features = transformer.fit_transform(features).toarray()
print(features.shape)

scaler = StandardScaler()
features = scaler.fit_transform(features)


pca = PCA()
features_pca = pca.fit_transform(features)

departments_train = departments[:n_train]
print(np.min(departments_train))
departments_test = departments[n_train:]
print(departments_train.shape)
print(departments_test.shape)



G = nx.convert_matrix.from_pandas_edgelist(
    df = df_edges_list,
    source = "from",
    target = "to"
)
adj_tilde = nx.adjacency_matrix(G).toarray() + np.eye(len(G.nodes), len(G.nodes))
deg_tilde = np.sum(adj_tilde, axis=0)
deg_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(deg_tilde))
normed_laplacian = deg_tilde_inv_sqrt * adj_tilde * deg_tilde_inv_sqrt

model = GraphConvNet(in_features = features.shape[1], normed_laplacian = normed_laplacian, n_class = 42)
model.fit(features, departments_train, n_train, n_epoch = 50)

prediction = model.predict(features)
print(prediction.shape)
print(accuracy_score(departments_test, prediction[n_train:]))
