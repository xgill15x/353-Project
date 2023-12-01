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
    plt.title('Red wins per event')

    plt.subplot(2, 3, 2)
    sns.histplot(types_list[1], kde=True)
    plt.title('Blue wins per event')

    plt.tight_layout()
    plt.show()

def seperate_colour_wins(fight_data):
    red_wins = fight_data[fight_data['Winner'] == 'Red']
    blue_wins = fight_data[fight_data['Winner'] == 'Blue']

    return red_wins, blue_wins

def wins_per_event(red_wins, blue_wins):
    
    red_wins_per_event = red_wins.groupby('date').count()['R_fighter']
    blue_wins_per_event = blue_wins.groupby('date').count()['B_fighter']

    colour_wins_per_event = pd.merge(red_wins_per_event, blue_wins_per_event, left_index=True, right_index=True)
    colour_wins_per_event = colour_wins_per_event.rename(columns={'R_fighter': 'red_wins_per_event', 'B_fighter': 'blue_wins_per_event'})

    return colour_wins_per_event['red_wins_per_event'], colour_wins_per_event['blue_wins_per_event']

def main():
    df = pd.read_csv("preprocessed.csv")
    df = further_preprocessing(df)

    red_wins, blue_wins = seperate_colour_wins(df)

    red_wins_per_event, blue_wins_per_event = wins_per_event(red_wins, blue_wins)

    # pre-transformed average colour wins per event (left for debugging/curiosity) ----------------------------------
    # red_wins_per_event_avg = red_wins_per_event.mean()
    # blue_wins_per_event_avg = blue_wins_per_event.mean()
    # print(red_wins_per_event_avg, blue_wins_per_event_avg)
    # -----------------------------------------------------------

    red_wins_per_event = np.sqrt(red_wins_per_event)
    blue_wins_per_event = np.sqrt(blue_wins_per_event)
    
    draw_plots([red_wins_per_event, blue_wins_per_event])

    levene_result = stats.levene(red_wins_per_event, blue_wins_per_event)
    t_test_result = stats.ttest_ind(red_wins_per_event, blue_wins_per_event)

    # Display the result
    print("Levene p-value:", levene_result.pvalue)
    print("T_test p-value:", t_test_result.pvalue)

if __name__=='__main__':
    main()
