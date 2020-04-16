import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sb

count = pd.read_csv("mutation_counts.csv", header=None, index_col=0, dtype={1: float})
count = count.replace(np.nan, 0)
top_count_mutations = count[count > 10.0].dropna()
# print(top_count_mutations.index)

gene_counts = pd.read_csv("RNAseq.csv", encoding="ISO-8859-1", dtype={'lab_id': str})
gene_counts.set_index('lab_id', inplace=True)
gene_counts_transpose = gene_counts.transpose()
gene_names = gene_counts.index
gene_count_ids = gene_counts_transpose.index

drug_responses = pd.read_csv("DrugResponses.csv")
inhibitors_list = drug_responses.inhibitor.unique()
del drug_responses['ic50']
pivot_drug_response = pd.pivot_table(drug_responses, index='lab_id', columns='inhibitor', aggfunc=np.max,  fill_value=0)
# Remove NA values
pivot_drug_response = pivot_drug_response[np.isfinite(pivot_drug_response)]


gene_features = top_count_mutations.index
top_gene_counts = gene_counts.loc[gene_features]
top_gene_counts_t = top_gene_counts.transpose()


drugs_list = [22, 28, 43, 46, 82, 91, 109, 120]
for drug_num in drugs_list:
    sort_by_drug = pivot_drug_response.reindex(
            pivot_drug_response['auc'].sort_values(by=inhibitors_list[drug_num], ascending=False).index)
    sort_by_drug = sort_by_drug[sort_by_drug > 0]
    print(sort_by_drug.shape)
    print(sort_by_drug.head())
    drug_response = sort_by_drug['auc'][inhibitors_list[drug_num]]
    drug_response = drug_response.dropna()
    drug_response_ids = drug_response.index
    combined_ids = list(set(gene_count_ids) & set(drug_response_ids))
    print("Selected Samples: ")
    print(combined_ids)
    drug_sample_genetics = top_gene_counts_t.loc[combined_ids]
    drug_sample_genetics = drug_sample_genetics.dropna(axis='columns')
    drug_response = drug_response.loc[combined_ids]
    X = drug_sample_genetics.sort_index()
    Y = drug_response.sort_index()
    X.to_csv('./data_outputs/X_data/X_' + inhibitors_list[drug_num] + '.csv')
    Y.to_csv('./data_outputs/Y_data/Y_' + inhibitors_list[drug_num] + '.csv')
    xy_combine = pd.concat([X, Y], axis=1)
    # histogram of auc under drug curve:
    plt.figure()
    Y.plot(kind='hist')
    plt.savefig('./data_outputs/hist_' + inhibitors_list[drug_num] + '.png')
    plt.figure()
    sb.distplot(Y)
    plt.savefig('./data_outputs/dist_' + inhibitors_list[drug_num] + '.png')
    # sb.regplot(x='name', y='name', data=Dataset, scatter=True)
    # to use pair plot below, select a few gene features
    # sb.pairplot(xy_combine, hue='group', palette='hls')






