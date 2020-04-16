import re

import pandas as pd
import numpy as np

drug_combo_db = pd.read_csv("data/drugcombs_scored.csv")
# columns: Drug1, Drug2, ZIP, Bliss, Loewe, HSA

test_data = pd.read_csv("./data_outputs/Y_data/Y_Crizotinib (PF-2341066).csv", index_col=0)
dh = test_data[test_data['Crizotinib (PF-2341066)'] > np.percentile(test_data['Crizotinib (PF-2341066)'], 66)]
dm = test_data[(np.percentile(test_data['Crizotinib (PF-2341066)'], 66) > test_data['Crizotinib (PF-2341066)']) & (
        test_data['Crizotinib (PF-2341066)'] >= np.percentile(test_data['Crizotinib (PF-2341066)'], 33))]
dl = test_data[test_data['Crizotinib (PF-2341066)'] < np.percentile(test_data['Crizotinib (PF-2341066)'], 33)]
print(len(test_data.index))
print(len(dh.index))
print(len(dm.index))
print(len(dl.index))

print(pd.qcut(test_data['Crizotinib (PF-2341066)'], q=3))
results, bin_edges = pd.qcut(test_data['Crizotinib (PF-2341066)'], q=3, retbins=True)
print(bin_edges[2])
dh = test_data[test_data > bin_edges[2]]
dm = test_data[(bin_edges[2] > test_data) & (test_data > bin_edges[1])]
dl = test_data[test_data <= bin_edges[1]]
print(len(dh.index))
print(len(dm.index))
print(len(dl.index))

mutation_counts = pd.read_csv("data/mutation_counts.csv", index_col=0, header=None)
print(mutation_counts.loc['FLT3', 1])

families = pd.read_csv("./data/family_table.csv", index_col=0)
test_drug = "Crizotinib (PF-2341066)"
drug_fam_list = families.loc[test_drug]
print(drug_fam_list[drug_fam_list > 0].index)

drug_families = pd.read_csv("./data/DrugFamilies.csv", index_col=0)
print(drug_families.loc[test_drug, 'family'].values)

drug_2 = "A-674563"
drug_3 = "Barasertib (AZD1152-HQPA)"
drug_4 = "AZD1480"
drug_5 = "Entospletinib (GS-9973)"

# print(drug_families.loc[drug_2].values)
print(drug_families.loc[drug_2, 'family'].values)
print(pd.Series(drug_families.loc[drug_2, 'family']).values)
# print(type(drug_families.loc[drug_2, 'family']))
print(pd.Series(drug_families.loc[drug_5, 'family']).values)

if len(set(drug_families.loc[test_drug, 'family'].values) & set(
        pd.Series(drug_families.loc[drug_5, 'family']).values)) > 0:
    print("True")
else:
    print("False")

list_test = list(range(len(drug_families.index)))
print(list_test)

corr_file = open("correlation outputs.txt")
corr_table = pd.DataFrame(columns=['Drug_1', 'Drug_2', 'Correlation'])
ind = 0
for line in corr_file:
    data = re.split('Drugs|,|Score', line)
    print(data[1][2:] + " " + data[2][1:-1] + " " + data[3][2:])
    corr_table = corr_table.append({'Drug_1': data[1][2:],
                                    'Drug_2': data[2][1:-1],
                                    'Correlation': data[3][2:].rstrip('\n')}, ignore_index=True)
    ind += 1
print(corr_table)
corr_table.to_csv('correlation_table.csv')
