"""
Author: Jason Gill

Description:
Our goal in this program is to determine the most effective type of strike that helps win UFC fights. 
The strike types to be analyzed are head strikes, body strikes and leg strikes. Further, we want to compare the differences 
between how effective each strike type is compared to the others.
"""

import pandas as pd
from scipy import stats
import pingouin as pg

import helper

def determine_dominant_strike(row):
    if ((row['Head_strikes'] > row['Leg_strikes']) & (row['Head_strikes'] > row['Body_strikes'])):
        return 'Head'
    elif ((row['Body_strikes'] > row['Head_strikes']) & (row['Body_strikes'] > row['Leg_strikes'])):
        return 'Body'
    elif ((row['Leg_strikes'] > row['Head_strikes']) & (row['Leg_strikes'] > row['Body_strikes'])):
        return 'Leg'

def main():
    winner_strike_stats = helper.fight_strike_stats_for_winners("data_sets/raw_total_fight_data.csv")
    # print(winner_strike_stats)
    
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

    # transforming strike values for normality - (commented bc still doesn't lead to best normality and variances are unequal, non-parametric tests used instead)
    # head_strikes = np.log(head_strikes)
    # body_strikes = np.log(body_strikes)
    # leg_strikes = np.log(leg_strikes)

    strike_types_list = [head_strikes, body_strikes, leg_strikes]
    helper.draw_strike_plots(strike_types_list)

    # dealing with unequal sample sizes: https://www.statology.org/anova-unequal-sample-size/
    # data has unequal variance and roughly normal distribution => kruskal-wallis test
    levene_result = stats.levene(head_strikes, body_strikes, leg_strikes)
    kruskal_result = stats.kruskal(head_strikes, body_strikes, levene_result)

    print("Levene p-value:", levene_result.pvalue)
    print("Kruskal p-value:", kruskal_result.pvalue)
    
    strike_types_and_counts = pd.concat([head_strikes, body_strikes, leg_strikes], axis=1, keys=['head_strikes', 'body_strikes', 'leg_strikes'])    
    melted_df = strike_types_and_counts.melt().dropna()

    # games-howell handles unequal sample sizes much better than tukey
    # http://bayes.acs.unt.edu:8083/BayesContent/class/Jon/ISSS_SC/Module009/isss_m91_onewayanova/node7.html
    posthoc = pg.pairwise_gameshowell(data=melted_df, dv='value', between='variable')

    print(posthoc)

if __name__=='__main__':
    main()
