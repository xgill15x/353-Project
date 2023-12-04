"""
Author: Vishaal Bharadwaj

Description:
We analyse the data to determine the best pound for pound fighter.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
ufc_data = pd.read_csv('data_sets/preprocessed_data.csv')

# List of relevant performance metrics with 'R_avg_' and 'B_avg_' prefixes
adjusted_relevant_columns = [
    'KD', 'SIG_STR_pct', 'TD_pct', 'SUB_ATT', 'REV', 'SIG_STR_att', 'SIG_STR_landed', 
    'TOTAL_STR_att', 'TOTAL_STR_landed', 'TD_att', 'TD_landed'
]


"""
Knockdowns (KD): High impact, weight = 0.20
Significant Strike Percentage (SIG_STR_pct): High accuracy is crucial, weight = 0.15
Takedown Percentage (TD_pct): Important for control, weight = 0.15
Submission Attempts (SUB_ATT): Represents finishing ability, weight = 0.10
Reversals (REV): Signifies adaptability, weight = 0.10
Significant Strikes Attempted (SIG_STR_att): Volume is important, weight = 0.10
Significant Strikes Landed (SIG_STR_landed): Effective striking, weight = 0.10
Total Strikes Attempted (TOTAL_STR_att): Overall activity, weight = 0.05
Total Strikes Landed (TOTAL_STR_landed): Overall effectiveness, weight = 0.05
Takedown Attempts (TD_att): Initiative to control, weight = 0.05
Takedowns Landed (TD_landed): Successful control, weight = 0.05
"""

# Weights for each metric based on their perceived importance
weights = {
    'KD': 0.20,
    'SIG_STR_pct': 0.10,
    'TD_pct': 0.10,
    'SUB_ATT': 0.10,
    'REV': 0.10,
    'SIG_STR_att': 0.10,
    'SIG_STR_landed': 0.10,
    'TOTAL_STR_att': 0.05,
    'TOTAL_STR_landed': 0.05,
    'TD_att': 0.05,
    'TD_landed': 0.05
}

# Ensure the weights sum to 1 (for a proper weighted average)
assert sum(weights.values()) == 1, "Weights do not sum to 1!"

# Columns for red and blue corner stats
red_columns_final = ['R_avg_' + col for col in adjusted_relevant_columns]
blue_columns_final = ['B_avg_' + col for col in adjusted_relevant_columns]

# Create separate dataframes for red and blue corner stats
red_fighter_stats = ufc_data[['R_fighter'] + red_columns_final]
blue_fighter_stats = ufc_data[['B_fighter'] + blue_columns_final]

# Rename columns to be common for both red and blue stats
common_column_names = ['fighter'] + adjusted_relevant_columns
red_fighter_stats.columns = common_column_names
blue_fighter_stats.columns = common_column_names

# Combine the red and blue stats into one dataframe
combined_fighter_stats = pd.concat([red_fighter_stats, blue_fighter_stats], axis=0)

# Group by fighter and calculate mean for each stat
fighter_stats_aggregated = combined_fighter_stats.groupby('fighter').mean().reset_index()

# Normalize the aggregated data
scaler = MinMaxScaler()
fighter_stats_aggregated[adjusted_relevant_columns] = scaler.fit_transform(fighter_stats_aggregated[adjusted_relevant_columns])

# Calculate the weighted sum for each fighter
for column in adjusted_relevant_columns:
    fighter_stats_aggregated[column] *= weights[column]
fighter_stats_aggregated['p4p_score'] = fighter_stats_aggregated[adjusted_relevant_columns].sum(axis=1)

# Sort the fighters by their pound-for-pound score in descending order
fighter_stats_aggregated = fighter_stats_aggregated.sort_values(by='p4p_score', ascending=False)

# Display the first few rows of the ranked dataframe
print(fighter_stats_aggregated.head(10))
