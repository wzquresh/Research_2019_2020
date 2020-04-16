import pandas as pd
import numpy as np

mutation_data = pd.read_csv("mutations.csv")
mutect_data = pd.read_csv('mutect_count.csv', index_col=0)
varscan_data = pd.read_csv('varscan_count.csv')
total_counts = pd.read_csv('total_count.csv')
#
# mutect_data = mutect_data.replace(0, np.nan).dropna(axis=1, how="all")
# varscan_data = varscan_data.replace(0, np.nan).dropna(axis=1, how="all")
# total_counts = total_counts.replace(0, np.nan).dropna(axis=1, how="all")
# mutect_data.to_csv('filtered_mutect.csv')
# varscan_data.to_csv('filtered_varscan.csv')
# total_counts.to_csv('filtered_total_count.csv')

# mutation_data = pd.read_csv("mutations.csv")
# mutect_data = pd.read_csv('filtered_mutect.csv', index_col=0)
# varscan_data = pd.read_csv('filtered_varscan.csv', index_col=0)
# total_counts = pd.read_csv('filtered_total_count.csv', index_col=0)

# print(mutect_data.shape)
# print(mutect_data.describe())
# print(varscan_data.shape)
# print(varscan_data.describe())
# print(total_counts.shape)
# print(total_counts.describe())

# Select rows and columns that have maximum data

# df.loc[df['column name'] > 0]
# df.loc[df.iloc[num] > 0]

# Sort genes by count
# Upload summary counts into a vector and look at distribution of vector
mutation_counts = mutect_data.apply(pd.Series.value_counts).iloc[1, :]
mutation_counts.sort_values(ascending=False, inplace=True)
print(mutation_counts[mutation_counts > 10])
mutation_counts.to_csv("mutation_counts.csv")



# Other idea: Create a matrix of users and drugs tested == Recommender matrix