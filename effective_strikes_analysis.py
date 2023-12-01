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

def draw_plots(stance_data_list):
    # Create histograms
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    sns.histplot(stance_data_list[0], kde=True)
    plt.title('Head strike counts in Head-strike heavy wins')

    plt.subplot(2, 3, 2)
    sns.histplot(stance_data_list[1], kde=True)
    plt.title('Body strike counts in Body-strike heavy wins')

    plt.subplot(2, 3, 3)
    sns.histplot(stance_data_list[2], kde=True)
    plt.title('Leg strike counts in Leg-strike heavy wins')

    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv('preprocessed.csv')

    df = pd.read_csv("raw_total_fight_data.csv", sep=';')
    # further preprocessing 
    df = further_preprocessing(df)

    # print(df)

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

    winner_strike_stats = pd.DataFrame(winner_strike_stats) # contains winner names and stances
    winner_strike_stats['Head_strikes'] = winner_strike_stats['Head_strikes'].str.split(' ').str[0].astype(int)
    winner_strike_stats['Body_strikes'] = winner_strike_stats['Body_strikes'].str.split(' ').str[0].astype(int)
    winner_strike_stats['Leg_strikes'] = winner_strike_stats['Leg_strikes'].str.split(' ').str[0].astype(int)

    # print(winner_strike_stats)

    won_by_head = []
    won_by_body = []
    won_by_leg = []
    
    for index, row in winner_strike_stats.iterrows():
        if ((row['Head_strikes'] > row['Leg_strikes']) & (row['Head_strikes'] > row['Body_strikes'])):
            won_by_head.append({"Head_strikes": row['Head_strikes'], "Body_strikes": row['Body_strikes'], "Leg_strikes": row['Leg_strikes']})
        
        elif ((row['Body_strikes'] > row['Head_strikes']) & (row['Body_strikes'] > row['Leg_strikes'])):
            won_by_body.append({"Head_strikes": row['Head_strikes'], "Body_strikes": row['Body_strikes'], "Leg_strikes": row['Leg_strikes']})

        elif ((row['Leg_strikes'] > row['Head_strikes']) & (row['Leg_strikes'] > row['Body_strikes'])):
            won_by_leg.append({"Name": row['Name'],"Head_strikes": row['Head_strikes'], "Body_strikes": row['Body_strikes'], "Leg_strikes": row['Leg_strikes']})
        

    won_by_head = pd.DataFrame(won_by_head)
    head_strikes = won_by_head['Head_strikes']

    won_by_body = pd.DataFrame(won_by_body)
    body_strikes = won_by_body['Body_strikes']

    won_by_leg = pd.DataFrame(won_by_leg)
    leg_strikes = won_by_leg['Leg_strikes']

    head_strikes = np.log(head_strikes)
    body_strikes = np.log(body_strikes)
    leg_strikes = np.log(leg_strikes)

    strike_types_list = [head_strikes, body_strikes, leg_strikes]
    draw_plots(strike_types_list)

    # dealing with unequal sample sizes: https://www.statology.org/anova-unequal-sample-size/
    # data has unequal variance and roughly normal distribution => kruskal-wallis test
    levene_result = stats.levene(head_strikes, body_strikes, leg_strikes)
    kruskal_result = stats.kruskal(head_strikes, body_strikes, levene_result)

    print("Levene p-value:", levene_result.pvalue)
    print("Kruskal p-value:", kruskal_result.pvalue)
    
    strike_types_and_counts = pd.concat([head_strikes, body_strikes, leg_strikes], axis=1, keys=['head_strikes', 'body_strikes', 'leg_strikes'])    
    melted_df = strike_types_and_counts.melt().dropna()

    posthoc = pg.pairwise_gameshowell(data=melted_df, dv='value', between='variable')

    print(posthoc)

if __name__=='__main__':
    main()
