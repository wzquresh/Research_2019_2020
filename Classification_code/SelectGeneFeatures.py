import pandas as pd
import numpy as np

# Possible options:
# 1. Top mutations
# 2. Highest counts: top 50% top 25%
# 3. Input features manually

def select_gene_features():
    mutations = pd.read_csv("mutations.csv")
    top_33 = mutations['symbol'].value_counts().head(33)

    gene_counts = pd.read_csv("RNAseq.csv", encoding="ISO-8859-1", dtype={'lab_id': str})
    gene_counts.set_index('lab_id', inplace=True)
    gene_names = gene_counts.index

    count = pd.read_csv("mutation_counts.csv", header=None, index_col=0, dtype={1: float})
    count = count.replace(np.nan, 0)
    # print(count)
    top_count_mutations = count[count > 10.0].dropna()
    # print(top_count_mutations.index)

    selector = int(input("Input Selection Type: 1, 2, 3 (1=top33, 2=all, 3=custom, 4=other) "))

    if selector == 1:
        return top_33.index
    elif selector == 2:
        return gene_names
    elif selector == 3:
        num_genes = int(input("How many genes? "))
        return mutations['symbol'].value_counts().head(num_genes)
    else:
        return top_count_mutations.index


