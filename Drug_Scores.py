import pandas as pd


drug_scores = pd.read_csv("data/drugcombs_scored.csv")
# Drug1, Drug2
drug_correlation = pd.read_csv("correlation_table.csv")
print(drug_scores.columns)
print(drug_correlation.columns)
print(drug_correlation['Drug_1'].head())
# Drug_1, Drug_2
# if Drug_1 + Drug_2 or Drug_2 + Drug_1 == Drug1 + Drug2
# or check Drug1 and Drug2 for all matching drugs in list
test_list = pd.DataFrame(data=None, columns=['drug1', 'drug2', 'ZIP', 'Bliss'])
test_list_2 = pd.DataFrame(data=None, columns=['drug1', 'drug2', 'ZIP', 'Bliss'])
item_num = 0
# Filter by cell line to get more accurate scores: Lymphoid tissue cell line labels below
# Lines: CCRF-CEM, HL-60(TB), K-562, MOLT-4, SR, TMD8, ED-40515, SU-DIPG-XIII, DIPG25, L-1236, HDLM-2, L-428, U-HO1
for row in drug_scores.itertuples():
    print("Loop")
    for corr_row in drug_correlation.itertuples():
        if str(row.Drug1) + str(row.Drug2) == str(corr_row.Drug_1) + str(corr_row.Drug_2):  # or str(row.Drug1) + str(row.Drug2) == str(corr_row.Drug_2) + str(corr_row.Drug_1):
            # store this row into test list
            print(item_num)
            item_num += 1
            test_list = test_list.append({'drug1': row.Drug1,
                                          'drug2': row.Drug2,
                                          'ZIP': row.ZIP,
                                          'Bliss': row.Bliss}, ignore_index=True)
        if str(row.Drug1) + str(row.Drug2) == str(corr_row.Drug_2) + str(corr_row.Drug_1):
            print(item_num)
            item_num += 1
            test_list_2 = test_list_2.append({'drug1': row.Drug1,
                                          'drug2': row.Drug2,
                                          'ZIP': row.ZIP,
                                          'Bliss': row.Bliss}, ignore_index=True)
print(item_num)
test_list.to_csv("test_list.csv")
test_list_2.to_csv("test_list_2.csv")



