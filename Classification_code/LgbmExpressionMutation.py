import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
count = pd.read_csv("mutation_counts.csv", header=None, index_col=0, dtype={1: float})
count = count.replace(np.nan, 0)
print(count)
top_count_mutations = count[count > 10.0].dropna()
print(top_count_mutations.index)

# Use the above index to select mutation data
mutation_counts = pd.read_csv("total_count.csv", index_col=0)
print(mutation_counts.head())
mutation_counts = mutation_counts[top_count_mutations.index]
X_mutation = mutation_counts.sort_index()

drug_num = int(input("drug number: "))

gene_counts = pd.read_csv("RNAseq.csv", encoding="ISO-8859-1", dtype={'lab_id': str})
gene_counts.set_index('lab_id', inplace=True)
gene_names = gene_counts.index

# Combine X values into one dataframe
X_expression, Y = select_samples(drug_num)
X_mutation = X_mutation.loc[X_expression.index]
X = pd.concat([X_expression, X_mutation], axis=1)
X.to_csv("./combined_e_m/X_" + inhibitors_list[drug_num] + ".csv")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# LightGBM
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
params_lgb = {'boosting_type': 'gbdt', 'objective': 'regression', 'metric': {'l2', 'l1'}, 'num_leaves': 31,
              'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5,
              'verbose': 0}
gbm = lgb.train(params_lgb, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
gbm.save_model('model_w_mutation.txt')
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_rms = copy.copy(y_pred)
# print(y_pred)
for i in range(0, len(y_pred)):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
# print(y_pred)


drug_model_data = open("./combined_e_m/" + inhibitors_list[drug_num] + "_model_stats.txt", 'a+')
print("LightGBM: ", file=drug_model_data)
print(accuracy_score(Y_test, y_pred), file=drug_model_data)
print(confusion_matrix(Y_test, y_pred), file=drug_model_data)
print(classification_report(Y_test, y_pred), file=drug_model_data)
print('The rmse of prediction is: ', mean_squared_error(Y_test, y_pred_rms)**0.5, file=drug_model_data)

explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(X_test)
# shap.force_plot(explainer.expected_value, shap_values, X_test)
shap.summary_plot(shap_values, X_test, show=False)
mpl.savefig('./combined_e_m/' + inhibitors_list[drug_num] +'_shap_values.png', dpi=1000, bbox_inches='tight')



