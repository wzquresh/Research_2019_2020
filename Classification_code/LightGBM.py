import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from SelectSamples import select_samples
import lightgbm as lgb
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
import matplotlib.pyplot as mpl
import shap


drug_responses = pd.read_csv("DrugResponses.csv")
inhibitors_list = drug_responses.inhibitor.unique()
print(inhibitors_list)

seed = 6

drug_num = int(input("drug number: "))

X, Y = select_samples(drug_num)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
# LightGBM
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
params_lgb = {'boosting_type': 'gbdt', 'objective': 'regression', 'metric': {'l2', 'l1'}, 'num_leaves': 31,
              'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5,
              'verbose': 0}
gbm = lgb.train(params_lgb, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
gbm.save_model('model.txt')
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_rms = copy.copy(y_pred)
# print(y_pred)
for i in range(0, len(y_pred)):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
# print(y_pred)


drug_model_data = open("./all_genes/" + inhibitors_list[drug_num] + "_model_stats.txt", 'a+')
print("LightGBM: ", file=drug_model_data)
print(accuracy_score(Y_test, y_pred), file=drug_model_data)
print(confusion_matrix(Y_test, y_pred), file=drug_model_data)
print(classification_report(Y_test, y_pred), file=drug_model_data)
print('The rmse of prediction is: ', mean_squared_error(Y_test, y_pred_rms)**0.5, file=drug_model_data)

explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(X_test)
# shap.force_plot(explainer.expected_value, shap_values, X_test)
shap.summary_plot(shap_values, X_test, show=False)
mpl.savefig('./all_genes/' + inhibitors_list[drug_num] +'_shap_values.png', dpi=1000, bbox_inches='tight')



