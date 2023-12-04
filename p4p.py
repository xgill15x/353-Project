"""
Author: Vishaal Bharadwaj

Description:
We analyse the data to determine the best pound for pound fighter.
"""


# import pandas as pd

# def calculate_p4p_score(row, weights):
#     win_percentage = row['wins'] / (row['wins'] + row['losses'] + row['draw'])
#     sig_strike_accuracy = row['SIG_STR_pct'] / 100.0
#     takedown_accuracy = row['TD_pct'] / 100.0
#     submission_accuracy = row['win_by_Submission'] / row['wins'] if row['wins'] > 0 else 0
#     ko_tko_rate = row['win_by_KO/TKO'] / row['wins'] if row['wins'] > 0 else 0

#     p4p_score = (
#         weights['win_percentage'] * win_percentage +
#         weights['sig_strike_accuracy'] * sig_strike_accuracy +
#         weights['takedown_accuracy'] * takedown_accuracy +
#         weights['submission_accuracy'] * submission_accuracy +
#         weights['ko_tko_rate'] * ko_tko_rate
#     )

#     return p4p_score

# def main():
#     # Example DataFrame (replace this with your actual DataFrame)
#     data = {
#         'wins': [20, 15, 25],
#         'losses': [5, 3, 2],
#         'draw': [2, 1, 0],
#         'SIG_STR_pct': [75, 80, 70],
#         'TD_pct': [60, 70, 50],
#         'win_by_Submission': [5, 2, 3],
#         'win_by_KO/TKO': [8, 10, 6],
#     }

#     df = pd.read_csv('recreate.csv')

#     # Example weights (adjust based on your preferences)
#     weights = {
#         'win_percentage': 0.3,
#         'sig_strike_accuracy': 0.2,
#         'takedown_accuracy': 0.2,
#         'submission_accuracy': 0.1,
#         'ko_tko_rate': 0.2
#     }

#     # Apply the function to each row in your DataFrame
#     df['p4p_score'] = df.apply(lambda row: calculate_p4p_score(row, weights), axis=1)

#     # Sort the DataFrame by the P4P score to get the rankings
#     df_ranked = df.sort_values(by='p4p_score', ascending=False)

#     # Print the ranked DataFrame
#     print("Ranked DataFrame:")
#     print(df_ranked)

# if __name__ == "__main__":
#     main()


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
ufc_data = pd.read_csv('preprocessed.csv')  # Replace with your file path

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

# Write the ranked dataframe to a CSV file
output_file_path = 'pound_for_pound_ranking.csv'  # Specify your desired output file path
fighter_stats_aggregated.to_csv(output_file_path, index=False)
