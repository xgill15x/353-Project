"""
Author: Vishaal Bharadwaj

Description:
We analyse the data to see if there is significant statistical difference in the average takedown percentage or
the average knockout values as we climb across 5 weight classes: 
Bantamweight
Featherweight
Lightweight
Welterweight
Middleweight
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, f_oneway, levene
import pingouin as pg
import scikit_posthocs as sp


# Load the dataset
file_path = 'preprocessed.csv'
ufc_data = pd.read_csv(file_path)

# Define the selected weight classes
selected_weight_classes = ['Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight', 'Middleweight']

# Filter the dataset for the selected weight classes
ufc_data_filtered = ufc_data[ufc_data['weight_class'].isin(selected_weight_classes)]

# Initialize a DataFrame to store the combined data
combined_data = pd.DataFrame()

# Iterate through each weight class and aggregate takedown percentages and knockdown averages
for weight_class in selected_weight_classes:
    # Filter data for the specific weight class
    weight_class_data = ufc_data_filtered[ufc_data_filtered['weight_class'] == weight_class]
    
    # Combine takedown percentages from both red and blue corners
    takedown_percentages = weight_class_data['R_avg_TD_pct'].tolist() + weight_class_data['B_avg_TD_pct'].tolist()
    
    # Combine knockdown averages from both red and blue corners
    knockdown_avg = weight_class_data['R_avg_KD'].tolist() + weight_class_data['B_avg_KD'].tolist()
    
    # Create a temporary DataFrame with the current weight class data
    temp_df = pd.DataFrame({
        'Weight_Class': weight_class,
        'Takedown_Percentage': takedown_percentages,
        'Knockdown_Avg': knockdown_avg
    })
    
    # Append to the combined DataFrame
    combined_data = pd.concat([combined_data, temp_df], ignore_index=True)





def plot_with_kde_from_df(data_lists, weight_classes, title_prefix):
    for data, weight_class in zip(data_lists, weight_classes):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data, label=f"{title_prefix} in {weight_class}")
        plt.title(f'{title_prefix} in {weight_class}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

def plot_without_kde_from_df(data_lists, weight_classes, title_prefix):
    for data, weight_class in zip(data_lists, weight_classes):
        plt.figure(figsize=(10, 6))
        counts, bin_edges = np.histogram(data, bins=20, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, counts, label=f"{title_prefix} in {weight_class}")
        plt.title(f'{title_prefix} in {weight_class}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()




# We note that the distribution is right-skewed for both takedown and knockdown values
# Applying square root transformation to the takedown and knockdown data
combined_data['Transformed_TD_pct'] = np.sqrt(combined_data['Takedown_Percentage'])
combined_data['Transformed_KD'] = np.sqrt(combined_data['Knockdown_Avg'])

# Preparing the data for plotting
td_original = [combined_data[combined_data['Weight_Class'] == wc]['Takedown_Percentage'].tolist() for wc in selected_weight_classes]
td_transformed = [combined_data[combined_data['Weight_Class'] == wc]['Transformed_TD_pct'].tolist() for wc in selected_weight_classes]

kd_original = [combined_data[combined_data['Weight_Class'] == wc]['Knockdown_Avg'].tolist() for wc in selected_weight_classes]
kd_transformed = [combined_data[combined_data['Weight_Class'] == wc]['Transformed_KD'].tolist() for wc in selected_weight_classes]

# plot_with_kde_from_df(td_original, selected_weight_classes, "Original Takedown Percentage")
# plot_with_kde_from_df(td_transformed, selected_weight_classes, "Transformed Takedown Percentage")
# plot_with_kde_from_df(kd_original, selected_weight_classes, "Original Knockdown Average")
# plot_with_kde_from_df(kd_transformed, selected_weight_classes, "Transformed Knockdown Average")




# Clearly our takedown data is normally distributed after the square root transformation is applied
# However our knockdown data is not normally distributed even after applying a square root transformation
# Hence we proceed with levene's test for equal variance for the takedown data to satisfy all conditions for ANOVA

# Levene's Test for equal variance
levene_td_result = levene(*td_transformed)
print("Levene's Test Result for Takedown Data:", levene_td_result)





# Since both conditions for ANOVA are satisfied by the takedown data - we proceed with ANOVA
# ANOVA Test on transformed takedown data
anova_td_result = f_oneway(*td_transformed)
print("ANOVA Test Result on Transformed Takedown Data:", anova_td_result)

# For our knockdown data - we proceed with a non-parametric test such as the Kruskal-Wallis Test
kruskal_kd_result = kruskal(*kd_original)
print("Kruskal-Wallis Test Result on Original Knockdown Data:", kruskal_kd_result)



print("\nTD\n")

# Checking the sample sizes for each weight class in the transformed takedown data
for wc, data in zip(selected_weight_classes, td_transformed):
    print(f"Sample size for {wc}: {len(data)}")

print("\nKD\n")

# Checking the sample sizes for each weight class in the original knockdown data
for wc, data in zip(selected_weight_classes, kd_original):
    print(f"Sample size for {wc}: {len(data)}")


# Since our sample sizes are different across the weight classes for takedown data - we proceed with Games-Howell test instead of Tukey
# Preparing the data for the Games-Howell test
# Prepare the data for the Games-Howell test
games_howell_data = combined_data[['Weight_Class', 'Transformed_TD_pct']]

# Conducting the Games-Howell test
posthoc_games_howell = pg.pairwise_gameshowell(data=games_howell_data, dv='Transformed_TD_pct', between='Weight_Class')
print(posthoc_games_howell)



# This will require your data to be in a long-form DataFrame with one column for KD values and another for group labels
# 'combined_data' already has the 'Weight_Class' and 'Knockdown_Avg' columns as per your previous code

# Now, melt the 'combined_data' DataFrame to create a long-form DataFrame
kd_data_long = pd.melt(combined_data, id_vars=['Weight_Class'], value_vars=['Knockdown_Avg'], var_name='Metric', value_name='Value')


# Perform Dunn's test
dunn_test_results = sp.posthoc_dunn(kd_data_long, val_col='value', group_col='Weight_Class', p_adjust='bonferroni')
print(dunn_test_results)