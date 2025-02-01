import os
import pandas as pd

def player_in_over(ball_by_ball_dir, over_by_over_dir, inning_number):
    """
    Processes ball-by-ball and over-by-over cricket match data for the specified inning.

    Parameters:
    - ball_by_ball_dir (str): Directory containing ball-by-ball files.
    - over_by_over_dir (str): Directory containing over-by-over files.
    - inning_number (int): Inning number (1 or 2) for processing data.
    """
    matches_with_multiple_bowlers = []
    over_by_over_files = [f for f in os.listdir(over_by_over_dir) if f.endswith('.csv')]

    for over_by_over_file in over_by_over_files:
        # Extract match_id from over-by-over filenames like match_<match_id>_innings2_data.csv
        match_id = over_by_over_file.split('_')[1]

        # Construct ball-by-ball filename dynamically using the match_id and inning_number
        ball_by_ball_file = os.path.join(ball_by_ball_dir, f"innings{inning_number}_{match_id}.csv")
        over_by_over_file_path = os.path.join(over_by_over_dir, over_by_over_file)

        if os.path.exists(ball_by_ball_file):
            ball_by_ball_df = pd.read_csv(ball_by_ball_file)
            ball_by_ball_df['over_number'] = ball_by_ball_df['ball'].apply(lambda x: int(x))
            over_by_over_df = pd.read_csv(over_by_over_file_path)

            batters, bowlers, balls_faced, balls_bowled = [], [], [], []
            for _, row in over_by_over_df.iterrows():
                over = int(row['over'])  # Ensure 'over' is treated as integer
                over_data = ball_by_ball_df[ball_by_ball_df['over_number'] == over]
                over_batters = over_data['striker'].unique()
                over_bowlers = over_data['bowler'].unique()

                striker_ball_count = {striker: (over_data['striker'] == striker).sum() for striker in over_batters}
                bowler_ball_count = {bowler: (over_data['bowler'] == bowler).sum() for bowler in over_bowlers}

                batters.append(', '.join(over_batters))
                bowlers.append(', '.join(over_bowlers))
                balls_faced.append(', '.join([str(striker_ball_count[striker]) for striker in over_batters]))
                balls_bowled.append(', '.join([str(bowler_ball_count[bowler]) for bowler in over_bowlers]))

                if len(over_bowlers) > 1:
                    if match_id not in matches_with_multiple_bowlers:
                        matches_with_multiple_bowlers.append(match_id)

            over_by_over_df['batters'] = batters
            over_by_over_df['bowlers'] = bowlers
            over_by_over_df['balls_faced'] = balls_faced
            over_by_over_df['balls_bowled'] = balls_bowled
            over_by_over_df.to_csv(over_by_over_file_path, index=False)
        else:
            print(f"Ball-by-ball file for match {match_id} in innings {inning_number} not found.")

    if matches_with_multiple_bowlers:
        print(f"Matches with multiple bowlers in an over for innings {inning_number}:")
        for match_id in matches_with_multiple_bowlers:
            print(f"Match {match_id}")
    else:
        print(f"No matches with multiple bowlers in an over for innings {inning_number}.")


player_in_over(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test", 
               r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test_processed", 
               inning_number=2)

player_in_over(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\train", 
               r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\train_processed", 
               inning_number=2)

player_in_over(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test", 
               r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test_processed", 
               inning_number=1)

player_in_over(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train", 
               r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train_processed", 
               inning_number=1)
