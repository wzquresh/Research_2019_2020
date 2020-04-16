import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from SelectSamples import select_samples
import lightgbm as lgb
from xgboost import XGBClassifier
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
import matplotlib.pyplot as mpl
import shap

scaler = preprocessing.StandardScaler()

drug_responses = pd.read_csv("DrugResponses.csv")
inhibitors_list = drug_responses.inhibitor.unique()
print(inhibitors_list)

seed = 6

# Drug numbers to test: 22, 28, 43, 46, 82, 91, 109, 120
drugs_list = [22, 28, 43, 46, 82, 91, 109, 120]
for num in drugs_list:
    X, Y = select_samples(num)
    names = X.columns
    # scaled_x = scaler.fit_transform(X)
    norm_x = preprocessing.normalize(X)
    X = pd.DataFrame(norm_x, columns=names)
    X.to_csv('./normalized_expression/X_' + inhibitors_list[num] + '.csv')
    Y.to_csv('./normalized_expression/Y_' + inhibitors_list[num] + '.csv')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    # LightGBM
    lgb_train = lgb.Dataset(X_train, Y_train)
    lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
    params_lgb = {'boosting_type': 'gbdt', 'objective': 'regression', 'metric': {'l2', 'l1'}, 'num_leaves': 31,
                  'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5,
                  'verbose': 0}
    gbm = lgb.train(params_lgb, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
    # gbm.save_model('model.txt')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_rms = copy.copy(y_pred)
    # print(y_pred)
    for i in range(0, len(y_pred)):
        if y_pred[i] >= 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    # print(y_pred)

    # XGBoost
    xgb = XGBClassifier()
    xgb_params = {'max_depth':[1, 2, 5, 6, 10, 50], 'learning_rate':[0.03, 0.04, 0.05], 'n_estimators':[1, 5, 10, 50, 100]}
    xgb_gs = GridSearchCV(xgb, xgb_params, cv=5)
    xgb_gs.fit(X_train, Y_train)
    xgb_best = xgb_gs.best_estimator_
    pred_xgb = xgb_best.predict(X_test)

    drug_model_data = open("./normalized_expression/" + inhibitors_list[num] + "_model_stats.txt", 'a+')

    print("XGBoost: ", file=drug_model_data)
    print(accuracy_score(Y_test, pred_xgb), file=drug_model_data)
    print(confusion_matrix(Y_test, pred_xgb), file=drug_model_data)
    print(classification_report(Y_test, pred_xgb), file=drug_model_data)

    print("LightGBM: ", file=drug_model_data)
    print(accuracy_score(Y_test, y_pred), file=drug_model_data)
    print(confusion_matrix(Y_test, y_pred), file=drug_model_data)
    print(classification_report(Y_test, y_pred), file=drug_model_data)
    print('The rmse of prediction is: ', mean_squared_error(Y_test, y_pred_rms) ** 0.5, file=drug_model_data)

    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(X_test)
    # shap.force_plot(explainer.expected_value, shap_values, X_test)
    mpl.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    mpl.savefig('./normalized_expression/' + inhibitors_list[num] + '_lgb_shap_values.png', dpi=1000, bbox_inches='tight')

    explainer_abc = shap.TreeExplainer(xgb_best)
    shap_values_abc = explainer_abc.shap_values(X_test)
    mpl.figure()
    shap.summary_plot(shap_values_abc, X_test, show=False)
    mpl.savefig('./normalized_expression/' + inhibitors_list[num] + '_xgboost_shap_values.png', dpi=1000, bbox_inches='tight')



