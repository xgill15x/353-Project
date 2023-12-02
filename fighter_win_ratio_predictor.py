import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt
import seaborn as sns

def further_preprocessing(data):
    data = data[~(data['Winner'] == 'Draw')] # We don't care about fights ending in a draw
    data.reset_index(drop=True, inplace=True)
    return data

def plot_results(X_valid, y_valid, y_pred):
    X_range = np.arange(0, X_valid.shape[0]).reshape(-1, 1)
    
    plt.figure(figsize=(20, 6))
    plt.plot(X_range, y_valid, color='blue', label='True values')
    plt.plot(X_range, y_pred, color='red', label='Predicted values', alpha=0.5)

    plt.xlabel('X_valid points (labeled by ID #)')
    plt.ylabel('Win ratios')
    plt.legend()
    plt.title('KNN Regression: True vs Predicted values')

    plt.gca().set_aspect(125, adjustable='box')  # Adjust the aspect ratio
    plt.show()

def fighter_win_loss_stats(preprocessed_data):
    win_count_map = {}
    lose_count_map = {}

    for index, row in preprocessed_data.iterrows():
        winner = row['Winner'][0]
        loser = 'B' # default value checked in following conditional

        if (winner == 'B'):
            loser = 'R'

        winner_name = row[winner + "_fighter"]
        loser_name = row[loser + "_fighter"]

        win_count_map[winner_name] = win_count_map.get(winner_name, 0) + 1  # if it exists increment by 1, if not init to 1
        lose_count_map[loser_name] = lose_count_map.get(loser_name, 0) + 1  # to track the amount of losses for a winner later on
        
    # contains names and win/lose counts (each fighter will be a datapoint)
    win_count_df = pd.DataFrame(list(win_count_map.items()), columns=['Name', 'Win_count']) # contains winner names and no. of wins
    lose_count_df = pd.DataFrame(list(lose_count_map.items()), columns=['Name', 'Lose_count']) #

    fighter_data = win_count_df.merge(lose_count_df, on='Name', how='outer')
    fighter_data['Lose_count'] = fighter_data['Lose_count'].fillna(0) # if name not found in losses map, default to 0 losses
    fighter_data['Win_count'] = fighter_data['Win_count'].fillna(0) # if name not found in wins map, default to 0 wins
    fighter_data['Win_ratio'] = fighter_data['Win_count'] / (fighter_data['Win_count'] + fighter_data['Lose_count'])
    
    return fighter_data

def main():
    fights_df = pd.read_csv('preprocessed_data.csv')
    fights_df = further_preprocessing(fights_df)
    fighter_win_ratios = fighter_win_loss_stats(fights_df)[['Name', 'Win_ratio']]

    fighters_df = pd.read_csv('preprocessed_fighter_details.csv')
    fighters_df = fighters_df.drop(columns=['Stance', 'DOB'])
    fighters_df = fighters_df.rename(columns={'fighter_name': 'Name'})

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

    print("X_valid shape:", X_valid.shape)
    print("y_valid shape:", y_valid.shape)
    
    plot_results(X_valid, y_valid, y_pred)

if __name__=='__main__':
    main()
