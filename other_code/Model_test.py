import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


# Read in data and process for use
mutations = pd.read_csv("mutations.csv")
top_33 = mutations['symbol'].value_counts().head(33)

gene_counts = pd.read_csv("RNAseq.csv", encoding="ISO-8859-1", dtype={'lab_id': str})
gene_counts.set_index('lab_id', inplace=True)
gene_counts_transpose = gene_counts.transpose()
gene_names = gene_counts.index

top_gene_counts = gene_counts.loc[top_33.index]
top_gene_counts_t = top_gene_counts.transpose()

drug_responses = pd.read_csv("DrugResponses.csv")
del drug_responses['ic50']
pivot_drug_response = pd.pivot_table(drug_responses, index='lab_id', columns='inhibitor', aggfunc=np.max,  fill_value=0)
# Include only actual data
pivot_drug_response = pivot_drug_response[np.isfinite(pivot_drug_response)]
sort_by_drug1 = pivot_drug_response.reindex(pivot_drug_response['auc'].sort_values(by='17-AAG (Tanespimycin)', ascending=False).index)
sort_by_drug1 = sort_by_drug1[sort_by_drug1 > 0]
print(sort_by_drug1.shape)
print(sort_by_drug1.head())
drug_1_response = sort_by_drug1.iloc[:, 0]
print(drug_1_response)
# Drop patients with no response data
drug_1_response = drug_1_response.dropna()
print(drug_1_response.isnull().sum())
print(drug_1_response)

# Select gene_counts for samples with drug response data
gene_count_ids = gene_counts_transpose.index
drug_response_ids = drug_1_response.index
combined_ids = list(set(gene_count_ids) & set(drug_response_ids))
print(combined_ids)
drug_1_sample_genetics = top_gene_counts_t.loc[combined_ids]
# drug_1_sample_genetics = gene_counts_transpose.loc[combined_ids]
drug_1_sample_genetics = drug_1_sample_genetics.dropna(axis='columns')
print(drug_1_sample_genetics)
# Make sure no NA values in X data
# drug_1_response = drug_1_response[np.isfinite(drug_1_response)]
drug_1_response = drug_1_response.loc[combined_ids]


# Data: X = drug_1_sample_genetics, Y = drug_1_response
print(drug_1_sample_genetics.shape)
print(drug_1_response.shape)

# Sort
X = drug_1_sample_genetics.sort_index()
Y = drug_1_response.sort_index()
print("X: genetic information by sample")
print(X.head())
print("Y: Drug response by sample")
print(Y.head())
X.to_csv('X.csv')
Y.to_csv('Y.csv')
# Make a model of first drug
# Use genes as X and drug response as Y
# Split at 30% test size

# Prior to splitting the data, we need to determine some type of threshold value
# Then based on the threshold set the Y's to either 1 or 0
print("Tanespimycin Statistics:")
print(Y.describe())
# Based on the above, set threshold for drug 1 to 153, i.e. >153 = 1, <153 = 0
print("Threshold = 50/50 split at 153 (median)")
threshold = 153

for i in range(0, len(Y.index)):
    if Y.iloc[i] >= 153:
        Y.iloc[i] = 1
    else:
        Y.iloc[i] = 0
print(Y)
print(Y.describe())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
knn = KNeighborsClassifier()
params_knn = {'n_neighbors': np.arange(1, 25)}
knn_gs = GridSearchCV(knn, params_knn, cv=5)
knn_gs.fit(X_train, Y_train)
knn_best = knn_gs.best_estimator_
print(knn_gs.best_params_)
print('knn: {}'.format(knn_best.score(X_test, Y_test)))




