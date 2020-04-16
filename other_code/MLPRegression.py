import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


count = pd.read_csv("data/mutation_counts.csv", header=None, index_col=0, dtype={1: float})
count = count.replace(np.nan, 0)
top_count_mutations = count[count > 10.0].dropna()

drug_responses = pd.read_csv("data/DrugResponses.csv")
inhibitors_list = drug_responses.inhibitor.unique()
drugs_list = [22, 28, 43, 46, 82, 91, 109, 120]

x_entries = os.listdir('./data_outputs/X_data/')
# print(x_entries)
y_entries = os.listdir('./data_outputs/Y_data/')
m_entries = os.listdir('./data_outputs/M_data/')
seed = 6
i = 0

for entry in x_entries:
    x = pd.read_csv('./data_outputs/X_data/' + entry, index_col=0)
    y = pd.read_csv('./data_outputs/Y_data/' + y_entries[i], index_col=0)
    m = pd.read_csv('./data_outputs/M_data/' + m_entries[i], index_col=0)
    drug = inhibitors_list[drugs_list[i]]
    # x = x.iloc[1:, :]
    gene_names = list(set(top_count_mutations.index) & set(x.columns))
    x = x[gene_names]
    norm_x = preprocessing.normalize(x)
    m = m[gene_names]
    m = m.loc[x.index]
    m = m.fillna(0)
    norm_m = preprocessing.normalize(m)
    names = x.columns
    nx = pd.DataFrame(norm_x, columns=names)
    nm = pd.DataFrame(norm_m, columns=names)
    X = pd.concat([nx, nm], axis=1)
    X = X.iloc[1:, :]
    X = X.dropna(axis=1)
    x = nx.iloc[1:, :]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    i += 1
    model = MLPRegressor(activation='logistic')
    reg = model.fit(x_train, y_train)
    file = open("./regression_outputs/MLP/" + drug + ".txt", 'w+')
    print(reg.score(x_test, y_test), file=file)
    file.close()
