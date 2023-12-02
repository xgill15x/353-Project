
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

data = pd.read_csv('data_sets/preprocessed_data.csv') 


# question: Is there a significant difference in the average significant strike accuracy across different weight classes?

# Melting the dataset to combine Red and Blue fighters' data
melted_data = pd.melt(data, id_vars=['weight_class'], 
                      value_vars=['R_avg_SIG_STR_pct', 'B_avg_SIG_STR_pct'],
                      var_name='Fighter_Color', value_name='SIG_STR_pct')
print(melted_data)
# Dropping NaN values
melted_data_cleaned = melted_data.dropna(subset=['SIG_STR_pct'])

# Grouping the cleaned data by 'weight_class' and 'Fighter_Color'
grouped_melted_data = melted_data_cleaned.groupby(['weight_class', 'Fighter_Color'])

# Preparing the data for ANOVA, create dictionary, name[0] refers to the weight_class and name[1] refers to the Fighter_Color
anova_data_dict = {(name[0], name[1]): group['SIG_STR_pct'] for name, group in grouped_melted_data}
#print(anova_data_dict)

# For the ANOVA test, you can still use the values from this dictionary
anova_combined_result = f_oneway(*anova_data_dict.values())
# kruskal, make graph showing distribution
# gameshowl

print(anova_combined_result)
