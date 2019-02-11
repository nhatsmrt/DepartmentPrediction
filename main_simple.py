import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC

np.random.seed(0)

df = pd.read_csv("./Data/features.csv").fillna(0)
department = df[["department"]].values.reshape(-1)

features = df.drop("name", axis = 1).drop("department", axis = 1).values
pca = PCA(n_components = 5)
features_pca = pca.fit_transform(features)


features_train, features_test, department_train, department_test = train_test_split(
    features_pca,
    department,
    train_size = 0.8
)

model = SVC()
model.fit(features_train, department_train)

pred = model.predict(features_test)
print(accuracy_score(department_test, pred))

