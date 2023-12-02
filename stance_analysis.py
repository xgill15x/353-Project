import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg

import matplotlib.pyplot as plt
import seaborn as sns

def further_preprocessing(data):
    data = data[~(data['Winner'] == 'Draw')] # We don't care about fights ending in a draw
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def get_unique_stances(stance_data):
    return stance_data['R_Stance'].unique() # B_Stance contains the same values

def draw_plots(types_list):
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

def fighter_win_loss_stance_stats(preprocessed_data):
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
    df = pd.read_csv('preprocessed_data.csv')
    df = further_preprocessing(df)

    stance_data = fighter_win_loss_stance_stats(df)
    # print(stance_data)

    # Only want to keep people who have the competence to win with their stance (as we want to compare stances with ppl who represent them best)
    stance_data = stance_data[(stance_data['Win_ratio']) > 0]
    # print(stance_data)

    stance_counts = stance_data.groupby('Stance').count()
    # print(stance_counts)

    orthodox = pd.Series(stance_data[stance_data['Stance'] == "Orthodox"]['Win_ratio'])
    southpaw = pd.Series(stance_data[stance_data['Stance'] == "Southpaw"]['Win_ratio'])
    switch = pd.Series(stance_data[stance_data['Stance'] == "Switch"]['Win_ratio'])
    open_stance = pd.Series(stance_data[stance_data['Stance'] == "Open Stance"]['Win_ratio'])
    sideways = pd.Series(stance_data[stance_data['Stance'] == "Sideways"]['Win_ratio'])

    # pre-transformed average win ratios (left for debugging/curiosity) ----------------------------------
    # orthodox_avg = orthodox.mean()
    # southpaw_avg = southpaw.mean()
    # switch_avg = switch.mean()
    # open_stance_avg = open_stance.mean()
    # sideways_avg = sideways.mean()
    # print(orthodox_avg, southpaw_avg, switch_avg, open_stance_avg, sideways_avg)
    # -----------------------------------------------------------

    # transformations to deal with right-skewedness
    orthodox = np.sqrt(orthodox)
    southpaw = np.sqrt(southpaw)
    switch = np.sqrt(switch)
    open_stance = np.sqrt(open_stance)
    sideways = np.sqrt(sideways)

    stances_list = [orthodox, southpaw, switch, open_stance, sideways]
    draw_plots(stances_list)

    # dealing with unequal sample sizes: https://www.statology.org/anova-unequal-sample-size/
    # data has equal variance and roughly normal distribution => anova oneway
    levene_result = stats.levene(orthodox, southpaw, switch)
    anova_result = stats.f_oneway(orthodox, southpaw, switch)

    # Display the result
    print("Levene p-value:", levene_result.pvalue)
    print("Anova p-value:", anova_result.pvalue)

    stance_win_ratios = pd.concat([orthodox, southpaw, switch], axis=1, keys=['orthodox', 'southpaw', 'switch'])    
    melted_df = stance_win_ratios.melt().dropna()

    posthoc = pg.pairwise_gameshowell(data=melted_df, dv='value', between='variable')

    print(posthoc)

if __name__=='__main__':
    main()
