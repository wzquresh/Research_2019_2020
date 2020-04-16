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

seed = 6

drug_responses = pd.read_csv("DrugResponses.csv")
inhibitors_list = drug_responses.inhibitor.unique()
print(inhibitors_list)
# X, Y = select_samples(8)

def train_model(drug_num):
    X, Y = select_samples(drug_num)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

    # KNN
    knn = KNeighborsClassifier()
    params_knn = {'n_neighbors': np.arange(1, 25)}
    knn_gs = GridSearchCV(knn, params_knn, cv=5)
    knn_gs.fit(X_train, Y_train)
    knn_best = knn_gs.best_estimator_
    predictionsKNN = knn_best.predict(X_test)

    # Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=6, criterion='entropy', random_state=seed)
    params_rfc = {'n_estimators': [5, 10, 50, 100, 200]}
    rf_gs = GridSearchCV(rfc, params_rfc, cv=5)
    rf_gs.fit(X_train, Y_train)
    rf_best = rf_gs.best_estimator_
    predictionsRF = rf_best.predict(X_test)

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    predictionsLR = lr.predict(X_test)

    #SVM
    svm = SVC()
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_svc = {'C': Cs, 'gamma': gammas}
    svm_gs = GridSearchCV(svm, param_svc, cv=5)
    svm_gs.fit(X_train, Y_train)
    svm_best = svm_gs.best_estimator_
    predictionsSVM = svm_best.predict(X_test)

    # Ensemble of Classification Methods Voting Classifier
    estimators = [('rfc', rf_best), ('knn', knn_best), ('lr', lr), ('svm', svm_best)]
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(X_train, Y_train)
    predictionsEnsemble = ensemble.predict(X_test)

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

    drug_model_data = open("./model_outputs/" + inhibitors_list[drug_num] + "_model_stats.txt", 'w+')
    print("\n Model Scores: ", file=drug_model_data)
    print("KNeighbors:", file=drug_model_data)
    print(knn_gs.best_params_, file=drug_model_data)
    print(accuracy_score(Y_test, predictionsKNN), file=drug_model_data)
    print(confusion_matrix(Y_test, predictionsKNN), file=drug_model_data)
    print(classification_report(Y_test, predictionsKNN), file=drug_model_data)
    print("Random Forest:", file=drug_model_data)
    print(accuracy_score(Y_test, predictionsRF), file=drug_model_data)
    print(confusion_matrix(Y_test, predictionsRF), file=drug_model_data)
    print(classification_report(Y_test, predictionsRF), file=drug_model_data)
    print("Logistic Regression:", file=drug_model_data)
    print(accuracy_score(Y_test, predictionsLR), file=drug_model_data)
    print(confusion_matrix(Y_test, predictionsLR), file=drug_model_data)
    print(classification_report(Y_test, predictionsLR), file=drug_model_data)
    print("SVC:", file=drug_model_data)
    print(accuracy_score(Y_test, predictionsSVM), file=drug_model_data)
    print(confusion_matrix(Y_test, predictionsSVM), file=drug_model_data)
    print(classification_report(Y_test, predictionsSVM), file=drug_model_data)
    print("Ensemble Voting Classifier:", file=drug_model_data)
    print(accuracy_score(Y_test, predictionsEnsemble), file=drug_model_data)
    print(confusion_matrix(Y_test, predictionsEnsemble), file=drug_model_data)
    print(classification_report(Y_test, predictionsEnsemble), file=drug_model_data)
    print("LightGBM: ", file=drug_model_data)
    print(accuracy_score(Y_test, y_pred), file=drug_model_data)
    print(confusion_matrix(Y_test, y_pred), file=drug_model_data)
    print(classification_report(Y_test, y_pred), file=drug_model_data)
    print('The rmse of prediction is: ', mean_squared_error(Y_test, y_pred_rms)**0.5, file=drug_model_data)
    drug_model_data.close()
    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(X_test)
    # shap.force_plot(explainer.expected_value, shap_values, X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    mpl.savefig('./model_outputs/' + inhibitors_list[drug_num] +'_shap_values.png', dpi=1000, bbox_inches='tight')


print("Drugs: ")
print(inhibitors_list)
end = False
while not end:
    view_list = input("View List? y/n")
    if view_list.strip().lower() == 'y':
        print(inhibitors_list)
    train_model(int(input("Input Drug Number From List: ")))
    done = input("Done? y/n")
    if done.strip().lower() == 'y':
        end = True


