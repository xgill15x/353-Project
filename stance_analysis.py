import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg

import helper

def get_unique_stances(stance_data):
    return stance_data['R_Stance'].unique() # B_Stance contains the same values

def main():
    stance_data = helper.fighter_win_loss_stats_with_stance('preprocessed_data.csv')
    # print(stance_data)

    # Only want to keep people who have the competence to win with their stance (as we want to compare stances with ppl who represent them best)
    stance_data = stance_data[(stance_data['Win_ratio']) > 0]

    # stance_counts = stance_data.groupby('Stance').count()
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
    helper.draw_stance_plots(stances_list)

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
