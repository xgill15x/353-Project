
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

# Load the dataset

data = pd.read_csv('data_sets/preprocessed_data.csv') 

#is there relation between color and result before jason color

#question: Do certain weightclasses commit to a certain type of finish/win_by methods

# Creating new columns that combine R and B win types
data['Total_win_by_Decision_Split'] = data['R_win_by_Decision_Split'] + data['B_win_by_Decision_Split']
data['Total_win_by_Decision_Unanimous'] = data['R_win_by_Decision_Unanimous'] + data['B_win_by_Decision_Unanimous']
data['Total_win_by_KO/TKO'] = data['R_win_by_KO/TKO'] + data['B_win_by_KO/TKO']
data['Total_win_by_Submission'] = data['R_win_by_Submission'] + data['B_win_by_Submission']
data['Total_win_by_TKO_Doctor_Stoppage'] = data['R_win_by_TKO_Doctor_Stoppage'] + data['B_win_by_TKO_Doctor_Stoppage']

# Selecting relevant columns for the updated contingency table
updated_finish_types = ['Total_win_by_Decision_Split', 'Total_win_by_Decision_Unanimous', 
                        'Total_win_by_KO/TKO', 'Total_win_by_Submission', 'Total_win_by_TKO_Doctor_Stoppage']
updated_contingency_table = data[['weight_class'] + updated_finish_types]

# Summing up the counts for each weight class and combined finish type
updated_contingency_table = updated_contingency_table.groupby('weight_class').sum()
print(updated_contingency_table)
# Performing the Chi-Square Test on the updated contingency table
_, p_updated, _, _ = chi2_contingency(updated_contingency_table)


#print(updated_contingency_table)
print("p-value:", p_updated)



