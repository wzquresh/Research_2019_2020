import os
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Step 1: Import mutation data
mutations = pd.read_csv("mutations.csv")
# Step 2: pull out top 33 mutations across cohort, by number of occurrences
top_33 = mutations['symbol'].value_counts().head(33)
print(top_33.index)
# Step 3: Import Genetic Profiles
gene_counts = pd.read_csv("RNAseq.csv", encoding="ISO-8859-1", dtype={'lab_id': str})
gene_counts.set_index('lab_id', inplace=True)
gene_counts_transpose = gene_counts.transpose()
gene_names = gene_counts.index
# Step 4: Get listing of just the top 33 genes based on the mutations
top_gene_counts = gene_counts.loc[top_33.index]   # future warning: may need to use .reindex() instead
print(top_gene_counts.head())
# Step 5: Import drug sensitivity data
drug_responses = pd.read_csv("DrugResponses.csv")


# Step 6: Cluster with drug sensitivity data

# test heat maps and cluster maps, i.e. dendrograms:
# The following dendrograms are preliminary, this is just gene count data
# following we will incorporate the drug sensitivity data as the data to cluster
# it will be clustered with: disease type, ELN2017, cytogenetics, and genetic profiles

# plt.imshow(top_gene_counts, cmap='hot')

plt.pcolor(top_gene_counts)
plt.yticks(np.arange(0.5, len(top_gene_counts.index), 1), top_gene_counts.index)
plt.xticks(np.arange(0.5, len(top_gene_counts.columns), 1), top_gene_counts.columns)
plt.savefig("plot1.png")



# top_gene_counts = top_gene_counts.set_index('lab_id')
del top_gene_counts.index.name
top_gene_counts.dropna(axis='columns')
top_gene_counts = top_gene_counts[np.isfinite(top_gene_counts)]
# Standardize:
#sns.clustermap(top_gene_counts, standard_scale=1)
# Normalize
#sns.clustermap(top_gene_counts, z_score=1)

del drug_responses['ic50']
pivot_drug_response = pd.pivot_table(drug_responses, index='lab_id', columns='inhibitor', aggfunc=np.max,  fill_value=0)
#pivot_drug_response.dropna(axis='rows')
pivot_drug_response = pivot_drug_response[np.isfinite(pivot_drug_response)]
sort_by_drug1 = pivot_drug_response.reindex(pivot_drug_response['auc'].sort_values(by='17-AAG (Tanespimycin)', ascending=False).index)
print(sort_by_drug1.head())
# html = sort_by_drug1.to_html(na_rep="")
# with open("drug_1_sort.html", "w") as f:
#     f.write(html)
# full_filename = os.path.abspath("drug_1_sort.html")
# webbrowser.open("file://{}".format(full_filename))
mask = pivot_drug_response.isnull()
plt.figure()
sns.clustermap(pivot_drug_response, mask=mask)
plt.savefig("plot2.png")
# plt.show()

# Step 7: Sort patients' profiles by genes and by drug sensitivity based on top 33 genes
# Sort patients gene expression profile data based on top 33 genes
lab_id_names = gene_counts_transpose.index.values
print(len(lab_id_names))  # 451 ids
sorted_profiles = top_gene_counts.sort_values(by=[lab_id_names[0], lab_id_names[1], lab_id_names[2], lab_id_names[3],
                                                  lab_id_names[4], lab_id_names[5], lab_id_names[6], lab_id_names[7],
                                                  lab_id_names[8], lab_id_names[9], lab_id_names[10]], ascending=False)
# sorted_profiles = top_gene_counts.sort_values(by=lab_id_names[0:5], ascending=False)
sorted_profiles.dropna(axis='rows')
sorted_profiles.dropna(axis='columns')
print(sorted_profiles)
#sns.clustermap(sorted_profiles)
# plt.show()


# Note: Cluster patients based on top 33 genes
# Data: top_gene_counts/sorted_profiles (as an experimental set)
# Method: KMeans
# API: sklearn
# Data preprocessing method: remove all columns or rows with NA values (remove patients or genes)
# Idea: This could be the test data set to create the initial model for drug selection
# Data Info: 33 features (top genes), clustering patients
print(sorted_profiles.describe())
# To start just cluster patients based on the first gene in the list
gene_1 = top_33.index[0]
print(gene_1)
gene_1_set = sorted_profiles.loc[gene_1]
print(gene_1_set.describe())
gene_1_set.dropna()
sorted_prof_t = sorted_profiles.transpose()
print(sorted_prof_t.head())
plt.figure()
sns.scatterplot(x='FLT3', y='WT1', data=sorted_prof_t)
plt.savefig("plot3.png")












