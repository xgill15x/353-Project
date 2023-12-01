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

def draw_plots(types_list):
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

def determine_dominant_strike(row):
    if ((row['Head_strikes'] > row['Leg_strikes']) & (row['Head_strikes'] > row['Body_strikes'])):
        return 'Head'
    elif ((row['Body_strikes'] > row['Head_strikes']) & (row['Body_strikes'] > row['Leg_strikes'])):
        return 'Body'
    elif ((row['Leg_strikes'] > row['Head_strikes']) & (row['Leg_strikes'] > row['Body_strikes'])):
        return 'Leg'

def main():
    df = pd.read_csv("raw_total_fight_data.csv", sep=';')
    df = further_preprocessing(df)

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

    winner_strike_stats = pd.DataFrame(winner_strike_stats) # contains winner names and strike stats
    
    # removing attempted strikes from the expressinon - we only care about the strikes that landed
    winner_strike_stats['Head_strikes'] = winner_strike_stats['Head_strikes'].str.split(' ').str[0].astype(int)
    winner_strike_stats['Body_strikes'] = winner_strike_stats['Body_strikes'].str.split(' ').str[0].astype(int)
    winner_strike_stats['Leg_strikes'] = winner_strike_stats['Leg_strikes'].str.split(' ').str[0].astype(int)
    
    # adding dominant strike stat
    winner_strike_stats['dominant_strike'] = winner_strike_stats.apply(determine_dominant_strike, axis=1)
    winner_strike_stats.dropna(inplace=True)
    # print(winner_strike_stats)

    head_strikes = winner_strike_stats[winner_strike_stats['dominant_strike'] == 'Head']['Head_strikes']
    body_strikes = winner_strike_stats[winner_strike_stats['dominant_strike'] == 'Body']['Body_strikes']
    leg_strikes = winner_strike_stats[winner_strike_stats['dominant_strike'] == 'Leg']['Leg_strikes']

    # pre-transformed average strike count in matches where that strike was dominantly used (left for debugging/curiosity) ----------------------------------
    # head_strike_count_avg = head_strikes.mean()
    # body_strikes_count_avg = body_strikes.mean()
    # leg_strike_count_avg = leg_strikes.mean()
    # print(head_strike_count_avg, body_strikes_count_avg, leg_strike_count_avg)
    # -----------------------------------------------------------

    # transforming strike values for normality
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
