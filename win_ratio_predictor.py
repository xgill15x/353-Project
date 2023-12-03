import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor

import helper

def main():
    fighter_win_ratios = helper.fighter_win_loss_stats('data_sets/preprocessed_data.csv')[['Name', 'Win_ratio']] # remove no. of wins and losses - only extract win ratio
    # print(fighter_win_ratios)

    fighters_df = pd.read_csv('data_sets/preprocessed_fighter_details.csv')
    fighters_df = fighters_df.drop(columns=['Stance', 'DOB'])
    fighters_df = fighters_df.rename(columns={'fighter_name': 'Name'})
    # print(fighter_df)

    full_fighter_details = fighters_df.merge(fighter_win_ratios, on='Name')
    # print(full_fighter_details)

    y = full_fighter_details['Win_ratio'].values

    stat_columns = full_fighter_details.select_dtypes(include=['number'])
    X = stat_columns.values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = make_pipeline(MinMaxScaler(), KNeighborsRegressor(n_neighbors=5)) # 5 seems to give best results
    model.fit(X_train, y_train)
    print(model.score(X_valid, y_valid))

    y_pred = model.predict(X_valid)
    
    helper.plot_win_ratio_predictor(X_valid, y_valid, y_pred)

if __name__=='__main__':
    main()
