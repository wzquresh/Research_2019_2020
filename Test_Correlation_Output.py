import pandas as pd

bliss_index_list = pd.read_csv("test_list.csv")
experimental_list = pd.read_csv("corr_list.csv")
full_exp_list = pd.read_csv("correlation_table.csv")

# Goal is to sort experiment list and bliss score list and count how many drug pairs are in the correct position
# Another possible comparison is position distance (if two lists are different lengths, subtract difference from score)
# Then the goal is to find the distance between any two drug pair positions
distance = 0
count = 0
bliss_index_list = bliss_index_list.sort_values('Bliss')
experimental_list = experimental_list.sort_values('Correlation')
full_exp_list = full_exp_list.sort_values('Correlation')
r_num = 0
pos_A = 0
pos_B = 0
corr_dist = pd.DataFrame(data=None, columns=['drug_pair', 'Distance'])
for row in bliss_index_list.itertuples():
    pos_B = 0
    for corr_row in experimental_list.itertuples():
        if row.drug1 != corr_row.drug1 or row.drug2 != corr_row.drug2:
            count += 1
        if row.drug1 == corr_row.drug1 and row.drug2 == corr_row.drug2:
            item_dist = pos_B - pos_A
            distance += pos_B - pos_A
            corr_dist = corr_dist.append({'drug_pair': corr_row.drug1 + "_" + corr_row.drug2,
                                          'Distance': item_dist}, ignore_index=True)
        pos_B += 1
    pos_A += 1
    # if row.drug1 + row.drug2 == experimental_list.iloc[r_num].drug1 + experimental_list.iloc[r_num].drug2:
print("Count: " + str(count))
corr_dist = corr_dist.drop_duplicates(subset='drug_pair', keep="last")
corr_dist.to_csv("corr_dist.csv")





