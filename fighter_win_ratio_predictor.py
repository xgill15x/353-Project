import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor

def further_preprocessing(stance_data):
    stance_data = stance_data[~(stance_data['Winner'] == 'Draw')] # We don't care about fights ending in a draw
    stance_data.dropna(inplace=True)
    stance_data.reset_index(drop=True, inplace=True)
    return stance_data

def fighter_win_loss_stats(preprocessed_data):
    # Extract stances and names of all the winners
    winners_info = []
    losers_info = []
    win_count_map = {}
    lose_count_map = {}

    for index, row in preprocessed_data.iterrows():
        winner = row['Winner'][0]
        loser = 'B' # default value checked in following conditional

        if (winner == 'B'):
            loser = 'R'

        winner_name = row[winner + "_fighter"]
        winner_stance = row[winner + "_Stance"]

        loser_name = row[loser + "_fighter"]
        loser_stance = row[loser + "_Stance"]

        win_count_map[winner_name] = win_count_map.get(winner_name, 0) + 1  # if it exists increment by 1, if not init to 1
        lose_count_map[loser_name] = lose_count_map.get(loser_name, 0) + 1  # to track the amount of losses for a winner later on

        winners_info.append({"Name": winner_name, "Stance": winner_stance})
        losers_info.append({"Name": loser_name, "Stance": loser_stance})
        
    winners_df = pd.DataFrame(winners_info) # contains winner names and stances
    losers_df = pd.DataFrame(losers_info) # similar to above comment
    winners_df.drop_duplicates(inplace=True)
    losers_df.drop_duplicates(inplace=True)
    
    # contains names and win/lose counts (each fighter will be a datapoint)
    win_count_df = pd.DataFrame(list(win_count_map.items()), columns=['Name', 'Win_count']) # contains winner names and no. of wins
    lose_count_df = pd.DataFrame(list(lose_count_map.items()), columns=['Name', 'Lose_count']) #
    
    winners_and_count = winners_df.merge(win_count_df, on='Name') # merges winner's stances and win counts into single df
    losers_and_count = losers_df.merge(lose_count_df, on='Name')
    
    fighter_data = pd.merge(winners_and_count, losers_and_count, on=['Name', 'Stance'], how='outer') # combined df with all fighters stances and win/lose counts
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

if __name__=='__main__':
    main()
