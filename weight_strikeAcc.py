import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# question: Is there a significant difference in the average significant strike accuracy across different weight classes?


# Load the dataset
data = pd.read_csv('data_sets/preprocessed_data.csv') 

# Combine the Red and Blue fighters' data into one column
data['combined_SIG_STR_pct'] = pd.concat([data['R_avg_SIG_STR_pct'], data['B_avg_SIG_STR_pct']], ignore_index=True)

# Drop the NaN values
data = data.dropna(subset=['combined_SIG_STR_pct', 'weight_class'])

# Prepare the data for ANOVA
grouped_data = data.groupby('weight_class')
anova_data = [group['combined_SIG_STR_pct'].tolist() for name, group in grouped_data]
# Perform the ANOVA test
anova_result = f_oneway(*anova_data)

# Output the ANOVA test result
print(f"ANOVA result: F={anova_result.statistic}, p={anova_result.pvalue}")

# If the ANOVA is significant, proceed with Tukey's HSD
if anova_result.pvalue < 0.05:
    # Perform Tukey's HSD test
    tukey = pairwise_tukeyhsd(endog=data['combined_SIG_STR_pct'],groups=data['weight_class'],alpha=0.05) 
    # Print Tukey's test result
    print(tukey.summary())
    # You can also plot the results
    fig = tukey.plot_simultaneous()    # Plot group confidence intervals
    fig.savefig('conf_int_weightStrike.png')
else:
    print("ANOVA is not significant. No need for post hoc testing.")

# Plotting the distribution of significant strike accuracy for each weight class using Seaborn and KDE
# Calculate the number of weight classes
num_groups = len(grouped_data)
# Determine the grid size for subplots based on the number of weight classes
grid_size = int(np.ceil(np.sqrt(num_groups)))
# Define the number of rows and columns for subplots in the grid
num_rows = grid_size
num_cols = grid_size
# Create a figure with subplots based on the grid size and set the figure size
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20), constrained_layout=True)
# Flatten the 2D array of subplots into a 1D array for easy iteration
axes = axes.flatten()
# Loop through each subplot and corresponding weight class data
for ax, (name, group) in zip(axes, grouped_data):
    # Combine data for significant strike accuracy from both red and blue fighters
    combined_group_data = pd.concat([group['R_avg_SIG_STR_pct'].dropna(), group['B_avg_SIG_STR_pct'].dropna()])
    # Create a histogram plot with KDE (Kernel Density Estimation)
    sns.histplot(combined_group_data, kde=True, bins=20, color='skyblue', edgecolor='black', ax=ax)
    # Set the title of the subplot to the name of the weight class
    ax.set_title(name)
    # Label the x-axis as 'Significant Strike Accuracy (%)'
    ax.set_xlabel('Significant Strike Accuracy (%)')
    # Label the y-axis as 'Frequency'
    ax.set_ylabel('Frequency')

# Hide any extra subplots that are not used
for i in range(num_groups, len(axes)):
    axes[i].set_visible(False)

# Save the entire figure as an image file named 'weight_stike_normal.png'
plt.savefig('weight_stike_normal.png')
