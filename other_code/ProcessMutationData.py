import pandas as pd


mutation_data = pd.read_csv("all_mutations.csv")
# 'ensembl_gene' vs gene 'symbol'
mutations = mutation_data.symbol.unique()
samples = mutation_data.labId.unique()
# filter data by genotyper: mutect vs varscan
mutect_data = mutation_data[mutation_data.genotyper != 'varscan']
varscan_data = mutation_data[mutation_data.genotyper != 'mutect']

mutect_count_matrix = pd.DataFrame(0, index=samples, columns=mutations)
varscan_count_matrix = pd.DataFrame(0, index=samples, columns=mutations)
total_count_matrix = pd.DataFrame(0, index=samples, columns=mutations)
# add to the counts: df.loc[samples[i], mutations[j]] += 1
# for index, row in flights.head().iterrows():
#      # access data using column names
#      print(index, row['delay'], row['distance'], row['origin'])

for row in mutect_data.itertuples():
    mutect_count_matrix.loc[row.labId, row.symbol] += 1
for row in varscan_data.itertuples():
    varscan_count_matrix.loc[row.labId, row.symbol] += 1
for row in mutation_data.itertuples():
    total_count_matrix.loc[row.labId, row.symbol] += 1

print(mutect_count_matrix.head())

mutect_count_matrix.to_csv('mutect_count.csv')
varscan_count_matrix.to_csv('varscan_count.csv')
total_count_matrix.to_csv('total_count.csv')



