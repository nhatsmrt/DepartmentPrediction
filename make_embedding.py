import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

df_edges_list = pd.read_csv("./Data/edges_list.csv")
G = nx.convert_matrix.from_pandas_edgelist(
    df = df_edges_list,
    source = "from",
    target = "to"
)

node2vec = Node2Vec(G)
model = node2vec.fit(window = 10, min_count = 1, batch_words = 4)
model.wv.save_word2vec_format("./Data/embeddings.emb")
