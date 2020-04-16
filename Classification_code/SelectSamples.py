import pandas as pd
import numpy as np
from SelectGeneFeatures import select_gene_features


gene_counts = pd.read_csv("RNAseq.csv", encoding="ISO-8859-1", dtype={'lab_id': str})
gene_counts.set_index('lab_id', inplace=True)
gene_counts_transpose = gene_counts.transpose()
gene_names = gene_counts.index
gene_count_ids = gene_counts_transpose.index


# Gene Feature Selection
gene_features = select_gene_features()
top_gene_counts = gene_counts.loc[gene_features]
top_gene_counts_t = top_gene_counts.transpose()


drug_responses = pd.read_csv("DrugResponses.csv")
inhibitors_list = drug_responses.inhibitor.unique()
del drug_responses['ic50']
pivot_drug_response = pd.pivot_table(drug_responses, index='lab_id', columns='inhibitor', aggfunc=np.max,  fill_value=0)
# Remove NA values
pivot_drug_response = pivot_drug_response[np.isfinite(pivot_drug_response)]


def select_samples(drug_num):
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
    X.to_csv('./model_outputs/X' + inhibitors_list[drug_num] + '.csv')
    Y.to_csv('./model_outputs/Y' + inhibitors_list[drug_num] + '.csv')
    Y = threshold_adjust(Y)
    return X, Y


def threshold_adjust(y):
    threshold = np.percentile(y, 80)
    for i in range(0, len(y.index)):
        if y.iloc[i] >= threshold:
            y.iloc[i] = 1
        else:
            y.iloc[i] = 0
    return y


