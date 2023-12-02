import pandas as pd

def fill_na_with_median(df, column_name):
    median = df[column_name].median()
    df[column_name] = df[column_name].fillna(value=median)

def convert_height_to_inches(height):
    if pd.isna(height):
        return None
    feet, inches = map(int, height.replace('"', '').split("'")) # seperate feet and inches, then convert to int
    return feet * 12 + inches

def main():
    df = pd.read_csv('raw_fighter_details.csv')

    df['Height'] = df['Height'].apply(convert_height_to_inches)
    df['Weight'] = df['Weight'].str.replace(' lbs.', '').astype(float) # remove lbs suffix
    df['Reach'] = df['Reach'].str.replace('"', '', regex=True).astype(float) # remove inch symbol
    
    # Convert percentage columns to numeric values between 0 and 1
    percentage_columns = ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']
    for column in percentage_columns:
        df[column] = df[column].str.replace('%', '').astype('float') / 100.0

    numerical_columns = df.select_dtypes(include=['number'])
    for column in numerical_columns:
        fill_na_with_median(df, column)

    # Write the DataFrame to a CSV file
    df.to_csv("preprocessed_fighter_details.csv", index=False)

if __name__=='__main__':
    main()
