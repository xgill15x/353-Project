import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats import multicomp

import matplotlib.pyplot as plt
import seaborn as sns

def further_preprocessing(stance_data):
    stance_data = stance_data[~(stance_data['Winner'] == 'Draw')] # We don't care about fights ending in a draw
    stance_data.dropna(inplace=True)
    stance_data.reset_index(drop=True, inplace=True)
    return stance_data

def get_unique_stances(stance_data):
    return stance_data['R_Stance'].unique() # B_Stance contains the same values


def main():
    df = pd.read_csv('preprocessed.csv')

    # stance_data = df[['R_fighter', 'B_fighter', 'R_Stance', 'B_Stance', 'Winner', 'wins']]

    # stance_data = stance_data[~(stance_data['Winner'] == 'Draw')]
    # stance_data.dropna(inplace=True)
    # stance_data.reset_index(drop=True, inplace=True)


    # further preprocessing 
    stance_data = further_preprocessing(df)

    # print(stance_data)

    # Extract stances and names of all the winners
    winners_info = []
    win_count_map = {}
    lose_count_map = {}

    for index, row in stance_data.iterrows():
        winner = row['Winner'][0]
        loser = 'B' # default value check in following conditional

        if (winner == 'B'):
            loser = 'R'

        # print(winner)
        winner_name = row[winner + "_fighter"]
        winner_stance = row[winner + "_Stance"]

        loser_name = row[loser + "_fighter"]

        win_count_map[winner_name] = win_count_map.get(winner_name, 0) + 1  # if it exists increment by 1, if not init to 1
        lose_count_map[loser_name] = lose_count_map.get(loser_name, 0) + 1  # to track the amount of losses for a winner later on

        winners_info.append({"Name": winner_name, "Stance": winner_stance})
        
    # Convert the result to a new DataFrame
    winners_df = pd.DataFrame(winners_info) # contains winner names and stances
    
    win_count_df = pd.DataFrame(list(win_count_map.items()), columns=['Name', 'Win_count']) # contains winner names and no. of wins
    lose_count_df = pd.DataFrame(list(lose_count_map.items()), columns=['Name', 'Lose_count']) # contains loser names and no. of losses
    win_lose_count_df = pd.merge(win_count_df, lose_count_df, on='Name', how='left')
    win_lose_count_df['Lose_count'] = win_lose_count_df['Lose_count'].fillna(0) # if name not found in losses map, default to 0 losses
    win_lose_count_df['Win_ratio'] = win_lose_count_df['Win_count'] / (win_lose_count_df['Win_count'] + win_lose_count_df['Lose_count'])

    winners_df = pd.merge(winners_df, win_lose_count_df, on="Name", how="inner")

    stance_counts = winners_df.groupby('Stance').count()
    print(stance_counts)

    orthodox = pd.Series(winners_df[winners_df['Stance'] == "Orthodox"]['Win_ratio'])
    southpaw = pd.Series(winners_df[winners_df['Stance'] == "Southpaw"]['Win_ratio'])
    switch = pd.Series(winners_df[winners_df['Stance'] == "Switch"]['Win_ratio'])
    open_stance = pd.Series(winners_df[winners_df['Stance'] == "Open Stance"]['Win_ratio'])
    sideways = pd.Series(winners_df[winners_df['Stance'] == "Sideways"]['Win_ratio'])

    # # Create histograms
    # plt.figure(figsize=(12, 8))

    # plt.subplot(2, 3, 1)
    # sns.histplot(orthodox, kde=True)
    # plt.title('Orthodox Stance')

    # plt.subplot(2, 3, 2)
    # sns.histplot(southpaw, kde=True)
    # plt.title('Southpaw Stance')

    # plt.subplot(2, 3, 3)
    # sns.histplot(switch, kde=True)
    # plt.title('Switch Stance')

    # plt.subplot(2, 3, 4)
    # sns.histplot(open_stance, kde=True)
    # plt.title('Open Stance')

    # plt.subplot(2, 3, 5)
    # sns.histplot(sideways, kde=True)
    # plt.title('Sideways Stance')

    # plt.tight_layout()
    # plt.show()

    anova_result = f_oneway(orthodox, southpaw, switch, open_stance, sideways)

    # Display the result
    print("ANOVA p-value:", anova_result.pvalue)

    orthodox = pd.Series(winners_df[winners_df['Stance'] == "Orthodox"]['Win_ratio'])
    southpaw = pd.Series(winners_df[winners_df['Stance'] == "Southpaw"]['Win_ratio'])
    switch = pd.Series(winners_df[winners_df['Stance'] == "Switch"]['Win_ratio'])
    open_stance = pd.Series(winners_df[winners_df['Stance'] == "Open Stance"]['Win_ratio'])
    sideways = pd.Series(winners_df[winners_df['Stance'] == "Sideways"]['Win_ratio'])


    # JASON THIS IS WHERE TO PICK UP TOMORROW - ISSUE: STANCES ARE NOT OF THE SAME LENGTH FOR TUKEY DOWN BELOW, FIGURE IT OUT

    data_df = pd.DataFrame({'orthodox': orthodox.values, 'southpaw': southpaw.values, 'switch': switch.values, 
                        'open_stance': open_stance.values, 'sideways': sideways.values})
    
    print(data_df)

    # data_melt = pd.melt(data_df)
    # posthoc = multicomp.pairwise_tukeyhsd(
    # data_melt['value'], data_melt['variable'],
    # alpha=0.05)

    # count = winners_df.groupby('Stance').count()



    
    # print(win_lose_count_df)

    # print(winners_df)

    # df = pd.DataFrame(list(my_map.items()), columns=['Key', 'Value'])

    return

    # print(winners_df)

    print(winners_df.groupby('Winner').count())

    # orthodox_winners = winners_df[winners_df['Winner'] == 'Orthodox']['Winner']
    # southpaw_winners = winners_df[winners_df['Stance'] == 'Southpaw']['Winner']

    # print(orthodox_winners)

    # red_winners = stance_data[stance_data['Winner'] == 'Red'][['R_fighter', 'R_Stance']]
    # print(red_winners)
    

    # blue_winners = stance_data[stance_data['Winner'] == 'Blue'][['B_fighter', 'B_Stance']]
    # print(blue_winners)

    return

    print(stance_data)

    print(get_unique_stances(stance_data))
    unique_wins = stance_data['Winner'].unique()


    print(unique_wins)



if __name__=='__main__':
    main()


# Another test comment - Jason
