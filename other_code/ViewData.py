import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb


count = pd.read_csv("data/mutation_counts.csv", header=None, index_col=0, dtype={1: float})
count = count.replace(np.nan, 0)
top_count_mutations = count[count > 120.0].dropna()


drug_responses = pd.read_csv("data/DrugResponses.csv")
inhibitors_list = drug_responses.inhibitor.unique()
drugs_list = [22, 28, 43, 46, 82, 91, 109, 120]


x_entries = os.listdir('./data_outputs/X_data/')
y_entries = os.listdir('./data_outputs/Y_data/')
m_entries = os.listdir('./data_outputs/M_data')


i = 0
for entry in x_entries:
    x = pd.read_csv('./data_outputs/X_data/' + entry, index_col=0)
    y = pd.read_csv('./data_outputs/Y_data/' + y_entries[i], index_col=0)
    m = pd.read_csv('./data_outputs/M_data/' + m_entries[i], index_col=0)
    # print(x['FLT3'].head())
    drug = inhibitors_list[drugs_list[i]]
    m = m.sort_index()
    m = m.loc[x.index]
    X = pd.concat([x, m], axis=1)
    gene_names = list(set(top_count_mutations.index) & set(x.columns))
    print(gene_names)
    pp_data = pd.concat([x[gene_names], y], axis=1)
    xy_combine = pd.concat([x, y], axis=1)
    xmy_combine = pd.concat([X, y], axis=1)
    xmy_combine.to_csv('./data_outputs/concat_data/xmy_' + inhibitors_list[drugs_list[i]] + '.csv')
    plt.figure()
    y.plot(kind='hist')
    plt.savefig('./data_outputs/hist_' + inhibitors_list[drugs_list[i]] + '.png')
    plt.figure()
    sb.distplot(y)
    plt.savefig('./data_outputs/dist_' + inhibitors_list[drugs_list[i]] + '.png')
    plt.figure()
    sb.pairplot(pp_data)
    plt.savefig('./data_outputs/pairplots/pairplt_' + inhibitors_list[drugs_list[i]] + '.png')
    i += 1


