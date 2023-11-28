import pandas as pd

df = pd.read_csv('data.csv')

df = df[['R_fighter', 'R_Stance', 'B_fighter', 'B_Stance', 'Winner']]

df = df.dropna()

print(df)