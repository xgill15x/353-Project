import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fill_na_with_median(df, column_name):
    median = df[column_name].median()
    df[column_name] = df[column_name].fillna(value=median)

def convert_foot_inches_to_inches(height):
    if pd.isna(height):
        return None
    feet, inches = map(int, height.replace('"', '').split("'")) # seperate feet and inches, then convert to int
    return feet * 12 + inches

def further_preprocessing_for_removing_draws(data):
    data = data[~(data['Winner'] == 'Draw')] # We don't care about fights ending in a draw
    data.reset_index(drop=True, inplace=True)
    return data

def further_preprocessing_for_stance_analysis(data):
    data = data[~(data['Winner'] == 'Draw')] # We don't care about fights ending in a draw
    data.dropna(inplace=True) # Used to drop columns with no stance (stance is only column with null values)
    data.reset_index(drop=True, inplace=True)
    return data

# ------------------------------------------------------------------------------------------

# RETURN DATAFRAME W/ FIGHTER NAMES, WINS, LOSSES, WIN RATIO
def fighter_win_loss_stats(preprocessed_data_csv):
    fights_df = pd.read_csv(preprocessed_data_csv)
    preprocessed_data_without_draws = further_preprocessing_for_removing_draws(fights_df)

    win_count_map = {}
    lose_count_map = {}

    for index, row in preprocessed_data_without_draws.iterrows():
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

# ------------------------------------------------------------------------------------------

# RETURN DATAFRAME W/ FIGHTER NAMES, WINS, LOSSES, WIN RATIO, STANCE
def fighter_win_loss_stats_with_stance(preprocessed_data_csv):
    
    df = pd.read_csv(preprocessed_data_csv)
    preprocessed_data_for_stance_analysis = further_preprocessing_for_stance_analysis(df)
    
    # Extract stances and names of all the winners
    winners_info = []
    losers_info = []
    win_count_map = {}
    lose_count_map = {}

    for index, row in preprocessed_data_for_stance_analysis.iterrows():
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

# ------------------------------------------------------------------------------------------

# RETURN DATAFRAME W/ FIGHT HEAD STRIKES, BODY STRIKES, LEG STRIKES OF WINNING FIGHTERS OF EACH MATCH
def fight_strike_stats_for_winners(raw_total_fight_data_csv):
    df = pd.read_csv(raw_total_fight_data_csv, sep=';')
    df = further_preprocessing_for_removing_draws(df)

    winner_strike_stats = []
    for index, row in df.iterrows():
        winner_name = row['Winner']
        winner_colour = ''

        if (winner_name == row['R_fighter']):
            winner_colour = 'R'
        else:
            winner_colour = 'B'

        winner_head_strikes = row[winner_colour + "_HEAD"]
        winner_body_strikes = row[winner_colour + "_BODY"]
        winner_leg_strikes = row[winner_colour + "_LEG"]

        winner_strike_stats.append({"Name": winner_name, "Head_strikes": winner_head_strikes, "Body_strikes": winner_body_strikes, "Leg_strikes": winner_leg_strikes})
    
    return pd.DataFrame(winner_strike_stats) # contains winner names and strike stats

# ------------------------------------------------------------------------------------------

# RETURN DATAFRAME W/ RED AND BLUE WINS SEPERATED
def seperate_colour_wins(preprocessed_data_csv):
    df = pd.read_csv(preprocessed_data_csv)
    fight_data = further_preprocessing_for_removing_draws(df)
    red_wins = fight_data[fight_data['Winner'] == 'Red']
    blue_wins = fight_data[fight_data['Winner'] == 'Blue']

    return red_wins, blue_wins

# ------------------------------------------------------------------------------------------

def draw_stance_plots(types_list):
    # Create histograms
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    sns.histplot(types_list[0], kde=True)
    plt.title('Orthodox Stance')

    plt.subplot(2, 3, 2)
    sns.histplot(types_list[1], kde=True)
    plt.title('Southpaw Stance')

    plt.subplot(2, 3, 3)
    sns.histplot(types_list[2], kde=True)
    plt.title('Switch Stance')

    plt.subplot(2, 3, 4)
    sns.histplot(types_list[3], kde=True)
    plt.title('Open Stance')

    plt.subplot(2, 3, 5)
    sns.histplot(types_list[4], kde=True)
    plt.title('Sideways Stance')

    plt.tight_layout()
    plt.show()

def draw_strike_plots(types_list):
    # Create histograms
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    sns.histplot(types_list[0], kde=True)
    plt.title('Head strike counts in Head-strike heavy wins')

    plt.subplot(2, 3, 2)
    sns.histplot(types_list[1], kde=True)
    plt.title('Body strike counts in Body-strike heavy wins')

    plt.subplot(2, 3, 3)
    sns.histplot(types_list[2], kde=True)
    plt.title('Leg strike counts in Leg-strike heavy wins')

    plt.tight_layout()
    plt.show()

def draw_colour_wins_plots(types_list):
    # Create histograms
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    sns.histplot(types_list[0], kde=True)
    plt.title('Red wins per event')

    plt.subplot(2, 3, 2)
    sns.histplot(types_list[1], kde=True)
    plt.title('Blue wins per event')

    plt.tight_layout()
    plt.show()

def plot_win_ratio_predictor(X_valid, y_valid, y_pred):
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