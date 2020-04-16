import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn import linear_model
from sklearn.metrics import r2_score
import os
import webbrowser

test = "Hi"
print(test)

# Input data:
# in_1 = input("Test Input: ")
# print("Input 1: " + in_1)

# reading from file
# tables to open S11
drug_families = open("DrugFamilies.csv", "r")
print(drug_families.readlines()[0])  # print out header
drug_families.close()

# Data Analysis
drug_families = pd.read_csv("DrugFamilies.csv")
print(drug_families.head())
print(drug_families.describe())

# clinical_sum = pd.read_csv("ClinicalSummary.csv", encoding="ISO-8859-1")
# print(clinical_sum.head())
# print(clinical_sum.describe())

# Gene counts data needs to be broken down more before analysis
gene_counts = pd.read_csv("RNAseq.csv", encoding="ISO-8859-1", dtype={'lab_id': str})
gene_counts.set_index('lab_id', inplace=True)
gene_counts_transpose = gene_counts.transpose()
gene_names = gene_counts.index
# columns = gene_counts_transpose.index[1:]
# gene_counts_transpose[:] = gene_counts_transpose[:].convert_objects(convert_numeric=True)
# gene_counts_transpose = gene_counts.transpose()

print(gene_counts_transpose.head()[gene_names[0]])
print(gene_counts_transpose.head().index)
# print(gene_counts_transpose.describe())
drug_responses = pd.read_csv("DrugResponses.csv")
print(drug_responses.head().lab_id)
# print(drug_responses.describe())

# compile list of lab ids in both dataframes
# use the gene counts of each of these lab_ids to
# compare against the list of drugs used on them
# will also need to compile list of each drug used on each lab id
# maybe dictionary with lab id as key and drugs/drug data and
# gene counts as values: dict = {  lab_id: [drug[data] genes[counts]], ... }
gene_count_ids = gene_counts_transpose.index
drug_response_ids = drug_responses.lab_id
combined_ids = list(set(gene_count_ids) & set(drug_response_ids))
print(combined_ids)  # list of ids in both data sets

# Next need to pull out drugs by id
# Step one get each unique drug
inhibitors_list = drug_responses.inhibitor.unique()
# Step two get gene and drug response data for each id
print(gene_counts_transpose.loc[combined_ids[0], :].head())  # example to point to one id's data
# Step three zip each patient sample with its corresponding data


# TO DO:
# Standard deviations of various data sets: np.std()

# Gene Counts Data (i.e. gene_counts/ gene_counts_transpose)
# GENE_COUNTS
# This is a distribution of the samples per gene (i.e. each gene's distribution across all samples)
# gene_count_stats = open('gene_stats.txt', 'w')
# for col in gene_counts:
#     if is_numeric_dtype(gene_counts[col]):
#         print(' % s: ' % col, file=gene_count_stats)
#         print('\t Mean = %.2f' % gene_counts[col].mean(), file=gene_count_stats)
#         print('\t Standard deviation = %.2f' % gene_counts[col].std(), file=gene_count_stats)
#         print('\t Minimum = %.2f' % gene_counts[col].min(), file=gene_count_stats)
#         print('\t Maximum = %.2f' % gene_counts[col].max(), file=gene_count_stats)
# gene_count_stats.close()
# # GENE_COUNTS_TRANSPOSE
# # This tells us the genes per sample data (i.e. the common genes and the gene outliers)
# gene_t_stats = open('gene_t_stats.txt', 'w')
# for col in gene_counts_transpose:
#     if is_numeric_dtype(gene_counts_transpose[col]):
#         print(' %s: ' % col, file=gene_t_stats)
#         print('\t Mean = %.2f' % gene_counts_transpose[col].mean(), file=gene_t_stats)
#         print('\t Standard deviation = %.2f' % gene_counts_transpose[col].std(), file=gene_t_stats)
#         print('\t Minimum = %.2f' % gene_counts_transpose[col].min(), file=gene_t_stats)
#         print('\t Maximum = %.2f' % gene_counts_transpose[col].max(), file=gene_t_stats)
# gene_t_stats.close()

# Correlation and Covariance of Gene Counts and Patient Samples
# indexing: gene_names, gene_count_ids
print("Gene counts data:")
print(gene_counts.loc[gene_names[0:5], gene_count_ids[0:5]])
print(gene_counts_transpose.loc[gene_count_ids[0:5], gene_names[0:5]])
print(gene_counts.describe())
print(gene_counts_transpose.describe())

# gene_counts_transpose.loc[gene_count_ids[0:5], gene_names[0:5]].boxplot()
# plt.show()

print("Drug response data:")
print(drug_responses.head())
print(drug_responses.describe())
print(drug_responses.cov())
print(drug_responses.corr())
# Pivot Table
pivot_drug_response = pd.pivot_table(drug_responses, index='lab_id', columns='inhibitor', aggfunc=np.max)
print(pivot_drug_response.head())
# html = pivot_drug_response.to_html(na_rep="")
# with open("review_matrix.html", "w") as f:
#     f.write(html)
# full_filename = os.path.abspath("review_matrix.html")
# webbrowser.open("file://{}".format(full_filename))
pivot_drug_response.to_csv("pivot_drug_response.csv", na_rep="")
# Ranking/Rating drug responses, determine gene sets for good drug responses
# The data we have is gene counts, and for evaluation we have a value for a drug response, or overall survival
# Then we can determine whether there is some correlation for certain genes with certain drug responses
# Step 1: get each drug data and sort it ascending or descending
# Step 2: decide on some arbitrary threshold value and say positive or negative
# Step 3: based on positive or negative values determine genes by patient that are correlated to this


# Potential Issues and Notes:
# Before correlation and covariance can be calculated, each patient sample needs full data across attributes
# This includes drug performance, gene counts, survival, etc...
# The covariance and correlation can then be calculated per patient sample
# e.g. sample 1: gene counts, drug sensitivity, overall survival
# then calculate the corr/cov for each gene against each drug/each gene against overall survival?

# print('Gene Counts Covariance: ' + gene_counts.loc[gene_names[0:3], gene_count_ids[0:10]].cov())
# print('Patient Samples Covariance: ' + gene_counts_transpose.cov())
# print('Gene Counts Correlation: ' + gene_counts.corr())
# print('Patient Samples Correlation: ' + gene_counts_transpose.corr())





