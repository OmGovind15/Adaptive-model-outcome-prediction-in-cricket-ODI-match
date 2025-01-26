import pandas as pd
import os

def player_in_over(ball_by_ball_dir, over_by_over_dir):
    """
    Processes ball-by-ball and over-by-over cricket match data.
    
    Parameters:
    - ball_by_ball_dir (str): Directory containing ball-by-ball files.
    - over_by_over_dir (str): Directory containing over-by-over files.
    
    The function processes each match, extracting batter and bowler details for each over,
    and updates the over-by-over CSV files with this information.
    """
    # List to track matches with multiple bowlers in an over
    matches_with_multiple_bowlers = []

    # Get all the over-by-over files in the over_by_over_dir
    over_by_over_files = [f for f in os.listdir(over_by_over_dir) if f.endswith('.csv')]

    # Loop through each over-by-over file and process the corresponding ball-by-ball file
    for over_by_over_file in over_by_over_files:
        # Extract the match number from the over_by_over file name (e.g., match_64814_cumulative_data.csv -> 64814)
        match_id = over_by_over_file.split('_')[1].split('.')[0]

        # Define the paths for the ball-by-ball and over-by-over files using the extracted match_id
        ball_by_ball_file = os.path.join(ball_by_ball_dir, f"innings1_{match_id}.csv")
        over_by_over_file_path = os.path.join(over_by_over_dir, over_by_over_file)

        # Check if the corresponding ball-by-ball file exists
        if os.path.exists(ball_by_ball_file):
            # Load the ball-by-ball dataset
            ball_by_ball_df = pd.read_csv(ball_by_ball_file)
            
            # Extract the over number from the 'ball' column (before the decimal point)
            ball_by_ball_df['over_number'] = ball_by_ball_df['ball'].apply(lambda x: int(x))

            # Load the over-by-over dataset
            over_by_over_df = pd.read_csv(over_by_over_file_path)

            # Create empty lists for batters, bowlers, and ball counts
            batters = []
            bowlers = []
            balls_faced = []
            balls_bowled = []

            # For each row in the over-by-over dataset, determine the batters and bowlers
            for index, row in over_by_over_df.iterrows():
                over = row['over']
                
                # Get the ball-by-ball data for the current over by matching over_number
                over_data = ball_by_ball_df[ball_by_ball_df['over_number'] == over]
                
                # Get the strikers and bowlers for each ball in the over
                over_batters = over_data['striker'].unique()
                over_bowlers = over_data['bowler'].unique()

                # Count how many balls each striker has faced in the over
                striker_ball_count = {striker: (over_data['striker'] == striker).sum() for striker in over_batters}
                
                # Count how many balls each bowler has bowled in the over
                bowler_ball_count = {bowler: (over_data['bowler'] == bowler).sum() for bowler in over_bowlers}
                
                # Append the batters to the list without ball count
                batters.append(', '.join(over_batters))
                
                # Append the bowlers to the list without ball count
                bowlers.append(', '.join(over_bowlers))
                
                # Append the ball counts for batters
                balls_faced.append(', '.join([str(striker_ball_count[striker]) for striker in over_batters]))
                
                # Append the ball counts for bowlers
                balls_bowled.append(', '.join([str(bowler_ball_count[bowler]) for bowler in over_bowlers]))

                # Check if there are multiple bowlers for this over, and if so, store the match ID
                if len(over_bowlers) > 1:
                    if match_id not in matches_with_multiple_bowlers:
                        matches_with_multiple_bowlers.append(match_id)

            # Add the batters, bowlers, balls faced, and balls bowled columns to the over-by-over DataFrame
            over_by_over_df['batters'] = batters
            over_by_over_df['bowlers'] = bowlers
            over_by_over_df['balls_faced'] = balls_faced
            over_by_over_df['balls_bowled'] = balls_bowled

            # Save the updated DataFrame back to the same over-by-over file
            over_by_over_df.to_csv(over_by_over_file_path, index=False)

            # Check the updated DataFrame (optional)
            print(f"Updated data for match {match_id}:")
            print(over_by_over_df.head())
        else:
            print(f"Ball-by-ball file for match {match_id} not found.")

    # Print the matches with multiple bowlers
    if matches_with_multiple_bowlers:
        print("Matches with multiple bowlers in an over:")
        for match_id in matches_with_multiple_bowlers:
            print(f"Match {match_id}")
    else:
        print("No matches with multiple bowlers in an over.")

# Run the function for both the train and test datasets
player_in_over(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train", r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train_processed")
player_in_over(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test", r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test_processed")

player_in_over(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\inning2\train", r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\train_processed")
player_in_over(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test", r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test_processed")