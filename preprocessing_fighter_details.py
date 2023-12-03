import pandas as pd
import helper

def main():
    df = pd.read_csv('data_sets/raw_fighter_details.csv')

    df['Height'] = df['Height'].apply(helper.convert_foot_inches_to_inches)
    df['Weight'] = df['Weight'].str.replace(' lbs.', '', regex=True).astype(float) # remove lbs suffix
    df['Reach'] = df['Reach'].str.replace('"', '', regex=True).astype(float) # remove inch symbol
    
    # Convert percentage columns to numeric values between 0 and 1
    percentage_columns = ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']
    for column in percentage_columns:
        df[column] = df[column].str.replace('%', '', regex=True).astype('float') / 100.0

    numerical_columns = df.select_dtypes(include=['number'])
    for column in numerical_columns:
        helper.fill_na_with_median(df, column)

    # Write the DataFrame to a CSV file
    helper.create_folder('data_sets')
    helper.create_folder('plots')

    df.to_csv("data_sets/preprocessed_fighter_details.csv", index=False)

if __name__=='__main__':
    main()
