import pandas as pd
import numpy as np
import os

# drug_families = pd.read_csv("data/DrugFamilies.csv")
# drugs = drug_families.inhibitor.unique()
# families = drug_families.family.unique()
# print(len(drugs))
# print(len(families))
# family_table = pd.DataFrame(0, drugs, families)
# for line in drug_families.itertuples():
#     family_table.loc[line.inhibitor, line.family] = 1
# family_table.to_csv("data/family_table.csv")

family_table = pd.read_csv("data/family_table.csv")  # possibly used to test hypothesis
drug_families = pd.read_csv("./data/DrugFamilies.csv", index_col=0)  # used to improve index
mutation_counts = pd.read_csv("data/mutation_counts.csv", index_col=0, header=None)
genes = mutation_counts.iloc[0:50, :].index

exp = pd.read_csv("data/RPKM.csv", dtype={'lab_id': str})
exp.set_index('lab_id', inplace=True)
exp = exp.transpose()
genes = list(set(genes) & set(exp.columns))
exp = exp[genes]
print(exp.describe())
# quartile groups: df.col.quantile([.25,.5,.75])
set1 = exp[exp < np.percentile(exp, 75)]
set2 = exp[exp < np.percentile(exp, 50)]
set3 = exp[exp < np.percentile(exp, 25)]
# df[df.col < percentile]

y_entries = os.listdir('./data_outputs/Y_data/')
# original test drugs: drugs_list = [22, 28, 43, 46, 82, 91, 109, 120]


def get_score(g_set, d_set, col):
    d_h = d_set[d_set[col] >= np.percentile(d_set, 66)]
    d_m = d_set[(np.percentile(d_set, 66) > d_set[col]) & (d_set[col] >= np.percentile(d_set, 33))]
    d_l = d_set[d_set[col] < np.percentile(d_set, 33)]
    dg_length = -1
    dg_score = 0
    set_index = ()
    # print(len(d_set))
    # print(d_h.index)
    # print(len(d_h.index))
    # print(len(d_m.index))
    # print(len(d_l.index))
    # print(len(g_set.index))
    # print(len(list(set(g_set.index) & set(d_h.index))))
    print(len(set(g_set.index).intersection(set(d_h.index))))
    # print(len(list(set(g_set.index) & set(d_m.index))))
    print(len(set(g_set.index).intersection(set(d_m.index))))
    # print(len(list(set(g_set.index) & set(d_l.index))))
    print(len(set(g_set.index).intersection(set(d_l.index))))
    if dg_length < len(list(set(g_set.index) & set(d_h.index))):
        dg_length = len(list(set(g_set.index) & set(d_h.index)))
        print("H: " + str(dg_length))
        set_index = list(set(g_set.index) & set(d_h.index))
        dg_score = 3
    if dg_length < len(list(set(g_set.index) & set(d_m.index))):
        dg_length = len(list(set(set_h.index) & set(d_m.index)))
        print("M: " + str(dg_length))
        set_index = list(set(g_set.index) & set(d_m.index))
        dg_score = 2
    if dg_length < len(list(set(g_set.index) & set(d_l.index))):
        dg_length = len(list(set(g_set.index) & set(d_l.index)))
        print("L: " + str(dg_length))
        set_index = list(set(g_set.index) & set(d_l.index))
        dg_score = 1
    print(dg_score)
    return dg_score, dg_length, set_index


for entry in y_entries:
    y = pd.read_csv('./data_outputs/Y_data/' + entry, index_col=0)
    y_h = y[y >= np.percentile(y, 66)]
    # print(y[(np.percentile(y, 66) > y) & (y > np.percentile(y, 33))])
    y_m = y[(np.percentile(y, 66) > y) & (y >= np.percentile(y, 33))]
    y_l = y[y < np.percentile(y, 33)]
    # d_g_t = pd.DataFrame(0, index=["h", "m", "l"], columns=genes)
    col = entry[2:-4]
    d_g_t = pd.DataFrame(0, index=[3, 2, 1], columns=genes)
    for (col_name, col_data) in exp.iteritems():
        set_h = col_data[col_data >= np.percentile(col_data, 66)]
        set_m = col_data[(np.percentile(col_data, 66) > col_data) & (col_data >= np.percentile(col_data, 33))]
        set_l = col_data[col_data < np.percentile(col_data, 33)]
        print(len(set_h))
        print(len(set_m))
        print(len(set_l))
        # count overlapping samples
        # if set maps to drug y_h,m,l store gene name with 3,2,1 and which gene set in row
        # d_g g_1 g_2 etc...
        # h   ?   ?
        # m   ?   ?
        # l   ?   ?
        # if gh in yh score = 3 stored in h
        # max(max_gh, max_gm, max_gl)
        # dg max stored in
        gh_score, gh_max, gh_index = get_score(set_h, y, col)
        gm_score, gm_max, gm_index = get_score(set_m, y, col)
        gl_score, gl_max, gl_index = get_score(set_l, y, col)
        # get highest correlation and associated score to store in d_g array
        g_max = max(gh_max, gm_max, gl_max)
        if gh_max == g_max:
            d_g_t.loc[3, col_name] = gh_score
            df = pd.DataFrame(gh_index, columns=["column"])
            df.to_csv("./drug_gene_index/" + entry[2:-4] + "_" + col_name + "_h.csv", index=False)
            # np.savetxt("./drug_gene_corr/" + entry + "_" + col_name + "h.csv", gh_index, delimiter=',', fmt='%s')
        elif gm_max == g_max:
            d_g_t.loc[2, col_name] = gm_score
            df = pd.DataFrame(gm_index, columns=["column"])
            df.to_csv("./drug_gene_index/" + entry[2:-4] + "_" + col_name + "_m.csv", index=False)
        else:
            d_g_t.loc[1, col_name] = gl_score
            df = pd.DataFrame(gl_index, columns=["column"])
            df.to_csv("./drug_gene_index/" + entry[2:-4] + "_" + col_name + "_l.csv", index=False)
        # test = {"high": (gh_score, gh_max), "med": (gm_score, gm_max), "low": (gl_score, gl_max)}
        # cycle through every gene set and find correlation between drug and each gene
    # finally store the drug-gene correlation table
    d_g_t.to_csv("drug_gene_corr/d_g_" + entry[2:-4] + ".csv")


# Next step is to find overlap between each drug-gene table to find drug-drug correlations
# in order: example -> 1-2, 1-3, 1-4, 2-3, 2-4, 3-4, etc...
# calculate correspondence scores using weights dg_score and g_max
# if a gene appears in the same row with the same score/value, add it to the total

file = open("./" + "correlation outputs.txt", 'a+')
dg_entries = os.listdir('./drug_gene_corr/')
dg_indexes = os.listdir('./drug_gene_index/')
print("Drug-Drug Scores List", file)
print("Drug names:", file)
print(dg_entries, file)
for i in range(len(dg_entries)):
    print(dg_entries[i])
    d_1 = pd.read_csv('./drug_gene_corr/' + dg_entries[i], index_col=0)
    print(d_1)
    for j in range(len(dg_entries)):
        if i+j+1 > len(dg_entries)-1:
            continue
        total = 0
        g_total = 0
        o_total = 0
        tag = ''
        # compare entry 0 to 1 etc... if two entries are equal add to total
        d_2 = pd.read_csv('./drug_gene_corr/' + dg_entries[i+j+1], index_col=0)
        for gene in genes:
            if d_1.loc[3, gene] == d_2.loc[3, gene] and d_1.loc[3, gene] > 0:
                print('./drug_gene_index/' + y_entries[i][2:-4]+'_'+gene+'_h.csv')
                d_1_h = pd.read_csv('./drug_gene_index/' + y_entries[i][2:-4]+'_'+gene+'_h.csv')
                d_2_h = pd.read_csv('./drug_gene_index/' + y_entries[i+j+1][2:-4] + '_' + gene + '_h.csv')
                overlap = len(list(set(d_1_h) & set(d_2_h)))
                o_total += overlap
                g_total += mutation_counts.loc[gene, 1] * d_1.loc[3, gene]
                total += d_1.loc[3, gene] * overlap   # Multiply score
                tag += 'h'
            if d_1.loc[2, gene] == d_2.loc[2, gene] and d_1.loc[2, gene] > 0:
                d_1_m = pd.read_csv('./drug_gene_index/' + y_entries[i][2:-4] + '_' + gene + '_m.csv')
                d_2_m = pd.read_csv('./drug_gene_index/' + y_entries[i + j + 1][2:-4] + '_' + gene + '_m.csv')
                overlap = len(list(set(d_1_m) & set(d_2_m)))
                o_total += overlap
                g_total += mutation_counts.loc[gene, 1] * d_1.loc[2, gene]
                total += d_1.loc[2, gene] * overlap
                tag += 'm'
            if d_1.loc[1, gene] == d_2.loc[1, gene] and d_1.loc[1, gene] > 0:
                d_1_l = pd.read_csv('./drug_gene_index/' + y_entries[i][2:-4] + '_' + gene + '_l.csv')
                d_2_l = pd.read_csv('./drug_gene_index/' + y_entries[i + j + 1][2:-4] + '_' + gene + '_l.csv')
                overlap = len(list(set(d_1_l) & set(d_2_l)))
                o_total += overlap
                g_total += mutation_counts.loc[gene, 1] * d_1.loc[1, gene]
                total += d_1.loc[1, gene] * overlap
                tag += 'l'
            total = total * mutation_counts.loc[gene, 1]
        # add overlaps separately from gene scores (o_total += overlap, g_total += g_score * exp_score)
        # g_score = mutations counts score
        # exp_score = d_g_table score
        # fam_score =( d_1_fam == d_2_fam or d_1_fam != d_2_fam ) 1 or -1
        # score = fam_score * auc_score * overlap + gene_score * exp_score (move parentheses as needed)
        fam_score = 0
        # print(dg_entries[i+j+1][4:-4])
        # print(drug_families.loc[dg_entries[i + j + 1][4:-4], 'family'].values)
        if dg_entries[i][4:-4] in drug_families.index and dg_entries[i + j + 1][4:-4] in drug_families.index:
            if len(set(pd.Series(drug_families.loc[dg_entries[i][4:-4], 'family']).values) & set(
                    pd.Series(drug_families.loc[dg_entries[i + j + 1][4:-4], 'family']).values)) > 0:
                print("True")
                fam_score = 1   # or length of similarity list as a multiplier
            else:
                print("False")
                fam_score = -1  # or length of differences as multiplier?
        else:
            fam_score = -1
        final_score = fam_score*(o_total + g_total)     # versus total score
        # final_score = (fam_score * o_total) + g_total  # secondary equation
        # final_score = fam_score * o_total * g_total  # final possible equation
        print("Drugs: " + dg_entries[i][4:-4] + ", " + dg_entries[i + j + 1][4:-4] + " Score: " + str(final_score), file=file)

file.close()








