import pandas as pd

def fill_na_with_median(df, column_name):
    median = df[column_name].median()
    df[column_name] = df[column_name].fillna(value=median)

def main():
    df = pd.read_csv('data.csv')

    # df = df[~(df['Winner'] == 'Draw')] # We don't care about fights ending in a draw
    df.reset_index(inplace=True)

    numerical_columns = df.select_dtypes(include=['number'])
    for column in numerical_columns[1:]:    # skips index
        fill_na_with_median(df, column)

    # Write the DataFrame to a CSV file
    df.to_csv("preprocessed.csv", index=False)

if __name__=='__main__':
    main()


# Another test comment - Jason
