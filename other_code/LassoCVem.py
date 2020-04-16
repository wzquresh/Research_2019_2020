import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import pickle


x_data = pd.read_csv("data/RPKM.csv", encoding="ISO-8859-1", dtype={'lab_id': str})
x_data.set_index('lab_id', inplace=True)
gene_counts_transpose = x_data.transpose()
gene_names = x_data.index
gene_count_ids = gene_counts_transpose.index

mutation_counts = pd.read_csv("data/total_count.csv", index_col=0)
X_mutation = mutation_counts.sort_index()

drug_responses = pd.read_csv("data/DrugResponses.csv")
inhibitors_list = drug_responses.inhibitor.unique()
del drug_responses['ic50']
pivot_drug_response = pd.pivot_table(drug_responses, index='lab_id', columns='inhibitor', aggfunc=np.max,  fill_value=0)
# Remove NA values
pivot_drug_response = pivot_drug_response[np.isfinite(pivot_drug_response)]

drugs_list = [22, 28, 43, 46, 82, 91, 109, 120]
for drug_num in drugs_list:
    sort_by_drug = pivot_drug_response.reindex(
        pivot_drug_response['auc'].sort_values(by=inhibitors_list[drug_num], ascending=False).index)
    sort_by_drug = sort_by_drug[sort_by_drug > 0]
    drug_response = sort_by_drug['auc'][inhibitors_list[drug_num]]
    drug_response = drug_response.dropna()
    drug_response_ids = drug_response.index
    combined_ids = list(set(gene_count_ids) & set(drug_response_ids))
    drug_sample_genetics = gene_counts_transpose.loc[combined_ids]
    drug_sample_genetics = drug_sample_genetics.sort_index()
    X_m = X_mutation.loc[combined_ids]
    X_m = X_m.dropna(axis=1)
    # mutation_names = X_m.columns
    X = pd.concat([drug_sample_genetics, X_m], axis=1, sort=True)
    coef_names = X.columns
    X.to_csv('./data_outputs/Lasso_with_mutation/X_' + inhibitors_list[drug_num] + '.csv')
    drug_response = drug_response.loc[combined_ids]
    Y = drug_response.sort_index()
    Y.to_csv('./data_outputs/Lasso_with_mutation/Y_' + inhibitors_list[drug_num] + '.csv')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    lasso = LassoCV(cv=10, random_state=0)
    lasso.fit(x_train, y_train)
    train_score = lasso.score(x_train, y_train)
    test_score = lasso.score(x_test, y_test)
    file = open("./regression_outputs/LassoCV/" + inhibitors_list[drug_num] + ".txt", 'w+')
    print("Lasso: alpha = 1", file=file)
    print("train score: " + str(train_score), file=file)
    print("test score: " + str(test_score), file=file)
    print(list(zip(lasso.coef_, coef_names)), file=file)
    pickle.dump(lasso, open("./regression_outputs/LassoCV/"+"lasso_"+inhibitors_list[drug_num]+".sav", 'wb'))
    file.close()





