import pandas as pd

def calculate_p4p_score(row, weights):
    win_percentage = row['wins'] / (row['wins'] + row['losses'] + row['draw'])
    sig_strike_accuracy = row['SIG_STR_pct'] / 100.0
    takedown_accuracy = row['TD_pct'] / 100.0
    submission_accuracy = row['win_by_Submission'] / row['wins'] if row['wins'] > 0 else 0
    ko_tko_rate = row['win_by_KO/TKO'] / row['wins'] if row['wins'] > 0 else 0

    p4p_score = (
        weights['win_percentage'] * win_percentage +
        weights['sig_strike_accuracy'] * sig_strike_accuracy +
        weights['takedown_accuracy'] * takedown_accuracy +
        weights['submission_accuracy'] * submission_accuracy +
        weights['ko_tko_rate'] * ko_tko_rate
    )

    return p4p_score

def main():
    # Example DataFrame (replace this with your actual DataFrame)
    data = {
        'wins': [20, 15, 25],
        'losses': [5, 3, 2],
        'draw': [2, 1, 0],
        'SIG_STR_pct': [75, 80, 70],
        'TD_pct': [60, 70, 50],
        'win_by_Submission': [5, 2, 3],
        'win_by_KO/TKO': [8, 10, 6],
    }

    df = pd.read_csv('recreate.csv')

    # Example weights (adjust based on your preferences)
    weights = {
        'win_percentage': 0.3,
        'sig_strike_accuracy': 0.2,
        'takedown_accuracy': 0.2,
        'submission_accuracy': 0.1,
        'ko_tko_rate': 0.2
    }

    # Apply the function to each row in your DataFrame
    df['p4p_score'] = df.apply(lambda row: calculate_p4p_score(row, weights), axis=1)

    # Sort the DataFrame by the P4P score to get the rankings
    df_ranked = df.sort_values(by='p4p_score', ascending=False)

    # Print the ranked DataFrame
    print("Ranked DataFrame:")
    print(df_ranked)

if __name__ == "__main__":
    main()
