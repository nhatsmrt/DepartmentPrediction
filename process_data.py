import re
import numpy as np
import pandas as pd

with open("./Data/email-Eu-core.txt") as f:
    edges_list = np.array([re.sub("\\n", "", x).split() for x in f.readlines()])
    df_edges_list = pd.DataFrame({'from': edges_list[:, 0], "to": edges_list[:, 1]})
    df_edges_list.to_csv("./Data/edges_list.csv", index = False)

with open("./Data/email-Eu-core-department-labels.txt") as f:
    data = np.array([re.sub("\\n", "", x).split() for x in f.readlines()])
    df_departments = pd.DataFrame({'id': data[:, 0], "department_id": data[:, 1]})
    df_departments.to_csv("./Data/department_labels.csv", index = False)
