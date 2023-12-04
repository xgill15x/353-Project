"""
Author: Jason Gill

Description:
This file contains a preprocessing script for the `data.csv` dataset
"""

import pandas as pd
import helper

def main():
    df = pd.read_csv('data_sets/data.csv')

    numerical_columns = df.select_dtypes(include=['number'])
    for column in numerical_columns:
        helper.fill_na_with_median(df, column)

    # Write the DataFrame to a CSV file
    helper.create_folder('data_sets')
    helper.create_folder('plots')

    df.to_csv("data_sets/preprocessed_data.csv", index=False)

if __name__=='__main__':
    main()
