import pandas as pd

drug_correlation = pd.read_csv("correlation_table.csv")
test_list = pd.read_csv("test_list.csv")

corr_list = pd.DataFrame(data=None, columns=['drug1', 'drug2', 'Correlation'])

item_num = 0
for corr_row in drug_correlation.itertuples():
    print("Loop")
    for test_row in test_list.itertuples():
        if str(corr_row.Drug_1) + str(corr_row.Drug_2) == str(test_row.drug1) + str(test_row.drug2) or \
                str(corr_row.Drug_2) + str(corr_row.Drug_1) == str(test_row.drug1) + str(test_row.drug2):
            # store this row into test list
            print(item_num)
            item_num += 1
            corr_list = corr_list.append({'drug1': corr_row.Drug_1,
                                          'drug2': corr_row.Drug_2,
                                          'Correlation': corr_row.Correlation}, ignore_index=True)

corr_list.to_csv("corr_list.csv")



