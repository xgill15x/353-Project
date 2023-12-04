
import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset

data = pd.read_csv('data_sets/preprocessed_data.csv') 

# Creating new columns that combine R and B win types
data['Total_win_by_Decision_Split'] = data['R_win_by_Decision_Split'] + data['B_win_by_Decision_Split']
data['Total_win_by_Decision_Unanimous'] = data['R_win_by_Decision_Unanimous'] + data['B_win_by_Decision_Unanimous']
data['Total_win_by_KO/TKO'] = data['R_win_by_KO/TKO'] + data['B_win_by_KO/TKO']
data['Total_win_by_Submission'] = data['R_win_by_Submission'] + data['B_win_by_Submission']
data['Total_win_by_TKO_Doctor_Stoppage'] = data['R_win_by_TKO_Doctor_Stoppage'] + data['B_win_by_TKO_Doctor_Stoppage']

# Selecting relevant columns for the updated contingency table
updated_finish_types = ['Total_win_by_Decision_Split', 'Total_win_by_Decision_Unanimous', 'Total_win_by_KO/TKO', 'Total_win_by_Submission', 'Total_win_by_TKO_Doctor_Stoppage']
updated_contingency_table = data[['weight_class'] + updated_finish_types]

# Summing up the counts for each weight class and combined finish type
updated_contingency_table = updated_contingency_table.groupby('weight_class').sum()
# Performing the Chi-Square Test on the updated contingency table
_, p_updated, _, _ = chi2_contingency(updated_contingency_table)
print("chi-squared p-value:", p_updated)
print(updated_contingency_table)







