import pandas as pd
import numpy as np


# Data sets: CPM, RPKM, clinical_summary, variant_analysis, DrugResponse
# Tables to produce: two tables per drug, one with cpm, one with rpkm, both tables have clinical summary and variants
# Variants table: each gene variant followed by its personal data ------>

clinical_data = pd.read_csv("data/clinical_summary.csv")
cpm = pd.read_csv("data/RNAseq.csv", encoding="ISO-8859-1", dtype={'lab_id': str})
rpkm = pd.read_csv("data/RPKM.csv", encoding="ISO-8859-1", dtype={'lab_id': str})
mutation_data = pd.read_csv("data/variants.csv")
drug_data = pd.read_csv("data/DrugResponses.csv")


# set up drug data
inhibitors_list = drug_data.inhibitor.unique()
del drug_data['auc']
pivot_drug_response = pd.pivot_table(drug_data, index='lab_id', columns='inhibitor', aggfunc=np.max,  fill_value=0)

# set up expression data
cpm.set_index('lab_id', inplace=True)
cpm_t = cpm.transpose()
gene_count_ids = cpm_t.index
rpkm.set_index('lab_id', inplace=True)
rpkm_t = rpkm.transpose()

# set up mutation data
numerical = ["tumor_only", "total_reads", "allele_reads", "normal_total_reads", "normal_allele_reads", "t_vaf", "n_vaf",
             "exac_af"]
non_numerical = ["chrom", "pos_start", "pos_end", "ref", "alt", "genotyper", "all_consequences", "impact", "refseq",
                 "biotype", "exon", "hgvsc", "hgvsp", "cdna_position", "cds_position", "protein_position",
                 "amino_acids", "codons", "existing_variation", "variant_class", "sift", "polyphen", "short_aa_change",
                 "validation", "rna_status"]
other = ["canonical"]
symbols = mutation_data.symbol.unique()
ensembles = mutation_data.ensembl_gene.unique()
samples = mutation_data.labId.unique()
# remove troublesome columns
del mutation_data["canonical"]
mut_cols = []
for symbol in symbols:
    mut_cols.append(symbol)
    for col in numerical:
        mut_cols.append("{0}_{1}".format(symbol, col))
    # for col in non_numerical:
    #     names = mutation_data[col].unique()
    #     for name in names:
    #         mut_cols.append("{0}_{1}".format(symbol, name))

# create dataframe of mutation data
mut_dat = pd.DataFrame(0, index=samples, columns=mut_cols)
for row in mutation_data.itertuples():
    for col in numerical:
        mut_dat.loc[row.labId, "{0}_{1}".format(row.symbol, col)] = getattr(row, col)
    # for col in non_numerical:
    #     names = mutation_data.col.unique()
    #     for name in names:
    #         mut_dat.loc[row.labId, "{0}_{1}".format(row.symbol, name)] = 1

# collect data into two tables for each drug
for drug in inhibitors_list:
    print(drug)
    sort_by_drug = pivot_drug_response.reindex(
        pivot_drug_response['ic50'].sort_values(by=drug, ascending=False).index)
    sort_by_drug = sort_by_drug[sort_by_drug > 0]
    drug_response = sort_by_drug['ic50'][drug]
    cpm_table = pd.concat([clinical_data.sort_index(), cpm_t.sort_index(), mut_dat.sort_index(), drug_response.sort_index()], axis=1)
    rpkm_table = pd.concat([clinical_data.sort_index(), rpkm_t.sort_index(), mut_dat.sort_index(), drug_response.sort_index()], axis=1)
    cpm_table.to_csv("data_sets/cpm_" + drug + ".csv")
    rpkm_table.to_csv("data_sets/rpkm_" + drug + ".csv")



