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

drug_responses = pd.read_csv("DrugResponses.csv")
inhibitors_list = drug_responses.inhibitor.unique()
drugs_list = [22, 28, 43, 46, 82, 91, 109, 120]

x_entries = os.listdir('./data_outputs/X_data/')
print(x_entries)
y_entries = os.listdir('./data_outputs/Y_data/')
seed = 6
i = 0
for entry in x_entries:
    x = pd.read_csv('./data_outputs/X_data/' + entry, index_col=0)
    y = pd.read_csv('./data_outputs/Y_data/' + y_entries[i], index_col=0)
    drug = inhibitors_list[drugs_list[i]]
    x = x.iloc[1:, :]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    i += 1
    model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    reg = model.fit(x_train, y_train)
    reg.score(x_train, y_train)
    file = open("./regression_outputs/" + drug + ".txt", 'w+')
    print("Coefficient", file=file)
    print(reg.coef_, file=file)
    print("Intercept", file=file)
    print(reg.intercept_, file=file)
    pred = reg.predict(x_test)
    print("Predictions", file=file)
    print(pred, file=file)
    print(reg.score(x_test, y_test), file=file)
    # print(confusion_matrix(y_test, pred), file=file)
    # print(classification_report(y_test, pred), file=file)
    file.close()


