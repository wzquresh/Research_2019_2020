import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
import os

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
    m = m[gene_names]
    m = m.loc[x.index]
    # m = m.fillna(0)
    X = pd.concat([x, m], axis=1)
    # X = X.iloc[1:, :]
    X = X.dropna(axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    i += 1
    model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    reg = model.fit(x_train, y_train)
    reg.score(x_train, y_train)
    file = open("./regression_outputs/Linear/" + drug + ".txt", 'w+')
    print(reg.score(x_test, y_test), file=file)
    # print("Coefficient", file=file)
    # print(reg.coef_, file=file)
    # print("Intercept", file=file)
    # print(reg.intercept_, file=file)
    # pred = reg.predict(x_test)
    # print("Predictions", file=file)
    # print(pred, file=file)
    file.close()


