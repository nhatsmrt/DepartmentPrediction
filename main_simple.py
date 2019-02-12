import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier


np.random.seed(22)

df_train = pd.read_csv("./Data/train.csv").fillna(0)
df_test = pd.read_csv("./Data/test.csv").fillna(0)

department_train = df_train[["true_department"]].values.reshape(-1)
features_train = df_train.drop(["name", "department", "true_department"], axis = 1).values
department_test = df_test[["true_department"]].values.reshape(-1)
features_test = df_test.drop(["name", "department", "true_department"], axis = 1).values

features_train[:, :4] = features_train[:, :4] - 1
features_test[:, :4] = features_test[:, :4] - 1


transformer = ColumnTransformer(
    [
        ("oh_0", OneHotEncoder(categories = [range(45)]), [0]),
        ("oh_1", OneHotEncoder(categories = [range(32)]), [1]),
        ("oh_2", OneHotEncoder(categories = [range(136)]), [2]),
        ("oh_3", OneHotEncoder(categories = [range(26)]), [3]),

    ]
)
features_train = transformer.fit_transform(features_train).toarray()
features_test = transformer.transform(features_test).toarray()

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)


pca = PCA()
features_train_pca = pca.fit_transform(features_train)
features_test_pca = pca.transform(features_test)



## Use k-Fold CV to select model:
kf = KFold(n_splits=10)
models = [LogisticRegression(C = 0.5), SVC(C = 0.5), RandomForestClassifier(), MLPClassifier(max_iter=2000)]
accs = []
for model in models:
    model_acc = []
    for train_idx, test_idx in kf.split(features_train):
        model.fit(features_train_pca[train_idx], department_train[train_idx])
        model_acc.append(accuracy_score(department_train[test_idx], model.predict(features_train_pca[test_idx])))

    accs.append(np.mean(model_acc))

print(accs)

## Selecting the best model (here, it is LogisticRegression!)
model = models[np.argmax(accs)]
model.fit(features_train_pca, department_train)

pred = model.predict(features_test_pca)
print(accuracy_score(department_test, pred))

