import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

# drug_response = pd.read_csv("DrugResponses.csv")
#
# samples = drug_response.lab_id.unique()
# drugs = drug_response.inhibitor.unique()
# drug_sample_matrix = pd.DataFrame(0, index=samples, columns=drugs)
# for row in drug_response.itertuples():
#     drug_sample_matrix.loc[row.lab_id, row.inhibitor] = row.auc
# drug_sample_matrix.to_csv("drug_sample_matrix.csv")

mutation_counts = pd.read_csv("mutation_counts.csv")
drug_sample_matrix = pd.read_csv("drug_sample_matrix.csv", index_col=0)
gene_sample_matrix = pd.read_csv("gene_sample_matrix.csv", index_col=0)
print(gene_sample_matrix.index)
gene_sample_matrix = gene_sample_matrix.reset_index().transpose().iloc[1:, :]
print(drug_sample_matrix.head())
print(gene_sample_matrix.head())
# gene_sample_matrix.index = gene_sample_matrix.iloc[0].values

model_ds = NMF()
W_ds = model_ds.fit_transform(drug_sample_matrix)
H_ds = model_ds.components_

model_gs = NMF(n_components=50)
W_gs = model_gs.fit_transform(gene_sample_matrix)
H_gs = model_gs.components_

np.savetxt('W_ds.csv', W_ds, delimiter=',')
np.savetxt('H_ds.csv', H_ds, delimiter=',')
np.savetxt('W_gs.csv', W_gs, delimiter=',')
np.savetxt('H_gs.csv', H_gs, delimiter=',')

# W_ds.to_csv('W_ds.csv')
# H_ds.to_csv('H_ds.csv')
# W_gs.to_csv('W_gs.csv')
# H_gs.to_csv('H_gs.csv')

# output = open("matrix_decomposition.txt", 'a+')
# print(W_ds, file=output)
# print(H_ds, file=output)
# print(W_gs, file=output)
# print(H_gs, file=output)
# output.close()




