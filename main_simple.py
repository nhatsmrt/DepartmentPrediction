import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

np.random.seed(0)

df = pd.read_csv("./Data/features.csv").fillna(0)
department = df[["department"]].values.reshape(-1)
features = df.drop("name", axis = 1).drop("department", axis = 1).values



features_train, features_test, department_train, department_test = train_test_split(
    features,
    department,
    train_size = 0.8
)


scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)


pca = PCA()
features_train_pca = pca.fit_transform(features_train)
features_test_pca = pca.transform(features_test)


## Use k-Fold CV to select model:
kf = KFold()
models = [LogisticRegression(), SVC(), RandomForestClassifier()]
accs = []
for model in models:
    model_acc = []
    for train_idx, test_idx in kf.split(features_train):
        model.fit(features_train[train_idx], department_train[train_idx])
        model_acc.append(accuracy_score(department_train[test_idx], model.predict(features_train[test_idx])))

    accs.append(np.mean(model_acc))

print(accs)

## Selecting the best model (here, it is LogisticRegression!)
model = models[np.argmax(accs)]
model.fit(features_train_pca, department_train)

pred = model.predict(features_test_pca)
print(accuracy_score(department_test, pred))

