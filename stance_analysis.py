import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg

import matplotlib.pyplot as plt
import seaborn as sns

def further_preprocessing(stance_data):
    stance_data = stance_data[~(stance_data['Winner'] == 'Draw')] # We don't care about fights ending in a draw
    stance_data.dropna(inplace=True)
    stance_data.reset_index(drop=True, inplace=True)
    return stance_data

def get_unique_stances(stance_data):
    return stance_data['R_Stance'].unique() # B_Stance contains the same values

def draw_plots(stance_data_list):
    # Create histograms
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    sns.histplot(stance_data_list[0], kde=True)
    plt.title('Orthodox Stance')

    plt.subplot(2, 3, 2)
    sns.histplot(stance_data_list[1], kde=True)
    plt.title('Southpaw Stance')

    plt.subplot(2, 3, 3)
    sns.histplot(stance_data_list[2], kde=True)
    plt.title('Switch Stance')

    plt.subplot(2, 3, 4)
    sns.histplot(stance_data_list[3], kde=True)
    plt.title('Open Stance')

    plt.subplot(2, 3, 5)
    sns.histplot(stance_data_list[4], kde=True)
    plt.title('Sideways Stance')

    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv('preprocessed.csv')

    # further preprocessing 
    stance_data = further_preprocessing(df)

    # Extract stances and names of all the winners
    winners_info = []
    losers_info = []
    win_count_map = {}
    lose_count_map = {}

    for index, row in stance_data.iterrows():
        winner = row['Winner'][0]
        loser = 'B' # default value check in following conditional

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
    losers_df = pd.DataFrame(losers_info)
    winners_df.drop_duplicates(inplace=True)
    losers_df.drop_duplicates(inplace=True)
    
    # contains names and win/lose counts (each fighter will be a datapoint)
    win_count_df = pd.DataFrame(list(win_count_map.items()), columns=['Name', 'Win_count']) # contains winner names and no. of wins
    lose_count_df = pd.DataFrame(list(lose_count_map.items()), columns=['Name', 'Lose_count']) # contains loser names and no. of losses
    
    winners_and_count = winners_df.merge(win_count_df, on='Name')
    losers_and_count = losers_df.merge(lose_count_df, on='Name')
    
    win_lose_count_df = pd.merge(winners_and_count, losers_and_count, on=['Name', 'Stance'], how='outer')
    win_lose_count_df['Lose_count'] = win_lose_count_df['Lose_count'].fillna(0) # if name not found in losses map, default to 0 losses
    win_lose_count_df['Win_count'] = win_lose_count_df['Win_count'].fillna(0) # if name not found in losses map, default to 0 losses
    win_lose_count_df['Win_ratio'] = win_lose_count_df['Win_count'] / (win_lose_count_df['Win_count'] + win_lose_count_df['Lose_count'])

    # Only want to keep people who have the competence to win with their stance (as we're comparing stances using ppl who represent them best)
    win_lose_count_df = win_lose_count_df[(win_lose_count_df['Win_ratio']) > 0]
    # print(win_lose_count_df)

    stance_counts = win_lose_count_df.groupby('Stance').count()
    # print(stance_counts)

    orthodox = pd.Series(win_lose_count_df[win_lose_count_df['Stance'] == "Orthodox"]['Win_ratio'])
    southpaw = pd.Series(win_lose_count_df[win_lose_count_df['Stance'] == "Southpaw"]['Win_ratio'])
    switch = pd.Series(win_lose_count_df[win_lose_count_df['Stance'] == "Switch"]['Win_ratio'])
    open_stance = pd.Series(win_lose_count_df[win_lose_count_df['Stance'] == "Open Stance"]['Win_ratio'])
    sideways = pd.Series(win_lose_count_df[win_lose_count_df['Stance'] == "Sideways"]['Win_ratio'])

    # transformations to deal with right-skewedness
    orthodox = np.sqrt(orthodox)
    southpaw = np.sqrt(southpaw)
    switch = np.sqrt(switch)
    open_stance = np.sqrt(open_stance)
    sideways = np.sqrt(sideways)

    stance_data_list = [orthodox, southpaw, switch, open_stance, sideways]
    draw_plots(stance_data_list)

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
