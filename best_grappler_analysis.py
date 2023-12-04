"""
Author: Vishaal Bharadwaj

Description:
We analyse the data to determine the best grappler.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
ufc_data = pd.read_csv('data_sets/preprocessed_data.csv')

# List of relevant grappling metrics with 'R_avg_' and 'B_avg_' prefixes
grappling_metrics = ['TD_landed', 'TD_pct', 'SUB_ATT', 'REV']

"""
Takedowns Landed (TD_landed): A crucial aspect of grappling, showing a fighter's ability to take the fight to the ground, weight = 0.40.
Takedown Accuracy (TD_pct): Reflects the effectiveness and efficiency of a fighter's takedown attempts, weight = 0.30.
Submission Attempts (SUB_ATT): Indicates a fighter's aggressiveness and skill in seeking fight-ending submissions, weight = 0.20.
Reversals (REV): Shows a fighter's ability to reverse positions, an important skill in grappling exchanges, weight = 0.10.
"""

# Weights for each metric based on their perceived importance
weights_grappling = {
    'TD_landed': 0.40,
    'TD_pct': 0.30,
    'SUB_ATT': 0.20,
    'REV': 0.10
}

# Columns for red and blue corner stats
red_columns_grappling = ['R_avg_' + col for col in grappling_metrics]
blue_columns_grappling = ['B_avg_' + col for col in grappling_metrics]

# Create separate dataframes for red and blue corner stats
red_fighter_stats_grappling = ufc_data[['R_fighter'] + red_columns_grappling]
blue_fighter_stats_grappling = ufc_data[['B_fighter'] + blue_columns_grappling]

# Rename columns to be common for both red and blue stats
common_column_names_grappling = ['fighter'] + grappling_metrics
red_fighter_stats_grappling.columns = common_column_names_grappling
blue_fighter_stats_grappling.columns = common_column_names_grappling

# Combine the red and blue stats into one dataframe
combined_fighter_stats_grappling = pd.concat([red_fighter_stats_grappling, blue_fighter_stats_grappling], axis=0)

# Group by fighter and calculate mean for each stat
fighter_stats_aggregated_grappling = combined_fighter_stats_grappling.groupby('fighter').mean().reset_index()

# Normalizing the grappling metrics using MinMaxScaler
scaler = MinMaxScaler()
fighter_stats_aggregated_grappling[grappling_metrics] = scaler.fit_transform(
    fighter_stats_aggregated_grappling[grappling_metrics]
)

# Calculate the weighted sum for each fighter
for column in grappling_metrics:
    fighter_stats_aggregated_grappling[column] *= weights_grappling[column]
fighter_stats_aggregated_grappling['grappler_score'] = fighter_stats_aggregated_grappling[grappling_metrics].sum(axis=1)

# Identifying the top grapplers
top_grapplers = fighter_stats_aggregated_grappling.sort_values(by='grappler_score', ascending=False)
top_10_grapplers = top_grapplers[['fighter', 'grappler_score']].head(10)

# Display the top 10 grapplers
print(top_10_grapplers)
