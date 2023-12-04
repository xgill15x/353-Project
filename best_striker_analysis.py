"""
Author: Vishaal Bharadwaj

Description:
We analyse the data to determine the best striker.
"""

# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# # Load your dataset
# ufc_data = pd.read_csv('preprocessed.csv')  # Replace with your file path

# # List of relevant striking metrics with 'R_avg_' and 'B_avg_' prefixes
# striking_metrics = ['KD', 'SIG_STR_landed', 'SIG_STR_pct']

# # Columns for red and blue corner stats
# red_columns_striking = ['R_avg_' + col for col in striking_metrics]
# blue_columns_striking = ['B_avg_' + col for col in striking_metrics]

# # Create separate dataframes for red and blue corner stats
# red_fighter_stats_striking = ufc_data[['R_fighter'] + red_columns_striking]
# blue_fighter_stats_striking = ufc_data[['B_fighter'] + blue_columns_striking]

# # Rename columns to be common for both red and blue stats
# common_column_names_striking = ['fighter'] + striking_metrics
# red_fighter_stats_striking.columns = common_column_names_striking
# blue_fighter_stats_striking.columns = common_column_names_striking

# # Combine the red and blue stats into one dataframe
# combined_fighter_stats_striking = pd.concat([red_fighter_stats_striking, blue_fighter_stats_striking], axis=0)

# # Group by fighter and calculate mean for each stat
# fighter_stats_aggregated_striking = combined_fighter_stats_striking.groupby('fighter').mean().reset_index()

# # Normalizing the striking metrics using MinMaxScaler
# scaler = MinMaxScaler()
# fighter_stats_aggregated_striking[striking_metrics] = scaler.fit_transform(
#     fighter_stats_aggregated_striking[striking_metrics]
# )

# # Calculating a striker score, assuming equal weight for each metric
# fighter_stats_aggregated_striking['striker_score'] = fighter_stats_aggregated_striking[striking_metrics].sum(axis=1)

# # Identifying the top 10 strikers
# top_strikers = fighter_stats_aggregated_striking.sort_values(by='striker_score', ascending=False)
# top_10_strikers = top_strikers[['fighter', 'striker_score']].head(10)

# # Display the top 10 strikers
# print(top_10_strikers)



import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
ufc_data = pd.read_csv('preprocessed.csv')  # Replace with your file path

# List of relevant striking metrics with 'R_avg_' and 'B_avg_' prefixes
striking_metrics = ['KD', 'SIG_STR_landed', 'SIG_STR_pct']

"""
Knockdowns (KD): Indicative of a fighter's power and ability to change the course of a fight, weight = 0.50.
Significant Strikes Landed (SIG_STR_landed): Reflects a fighter's effectiveness in landing meaningful strikes, weight = 0.30.
Significant Strike Percentage (SIG_STR_pct): Measures accuracy, a key component in striking efficiency, weight = 0.20.
"""

# Weights for each metric based on their perceived importance
weights_striking = {
    'KD': 0.50,
    'SIG_STR_landed': 0.30,
    'SIG_STR_pct': 0.20
}

# Columns for red and blue corner stats
red_columns_striking = ['R_avg_' + col for col in striking_metrics]
blue_columns_striking = ['B_avg_' + col for col in striking_metrics]

# Create separate dataframes for red and blue corner stats
red_fighter_stats_striking = ufc_data[['R_fighter'] + red_columns_striking]
blue_fighter_stats_striking = ufc_data[['B_fighter'] + blue_columns_striking]

# Rename columns to be common for both red and blue stats
common_column_names_striking = ['fighter'] + striking_metrics
red_fighter_stats_striking.columns = common_column_names_striking
blue_fighter_stats_striking.columns = common_column_names_striking

# Combine the red and blue stats into one dataframe
combined_fighter_stats_striking = pd.concat([red_fighter_stats_striking, blue_fighter_stats_striking], axis=0)

# Group by fighter and calculate mean for each stat
fighter_stats_aggregated_striking = combined_fighter_stats_striking.groupby('fighter').mean().reset_index()

# Normalizing the striking metrics using MinMaxScaler
scaler = MinMaxScaler()
fighter_stats_aggregated_striking[striking_metrics] = scaler.fit_transform(
    fighter_stats_aggregated_striking[striking_metrics]
)

# Calculate the weighted sum for each fighter
for column in striking_metrics:
    fighter_stats_aggregated_striking[column] *= weights_striking[column]
fighter_stats_aggregated_striking['striker_score'] = fighter_stats_aggregated_striking[striking_metrics].sum(axis=1)

# Identifying the top strikers
top_strikers = fighter_stats_aggregated_striking.sort_values(by='striker_score', ascending=False)
top_10_strikers = top_strikers[['fighter', 'striker_score']].head(10)

# Display the top 10 strikers
print(top_10_strikers)

