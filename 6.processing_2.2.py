import pandas as pd
import os

def player_in_over(ball_by_ball_dir, over_by_over_dir, innings_num):
    """
    Processes ball-by-ball and over-by-over cricket match data for the specified innings.

    Parameters:
    - ball_by_ball_dir (str): Directory containing ball-by-ball files for the specified innings.
    - over_by_over_dir (str): Directory containing over-by-over files for the specified innings.
    - innings_num (int): The innings number (1 or 2) being processed.
    """
    # Get all the over-by-over files in the over_by_over_dir
    over_by_over_files = [f for f in os.listdir(over_by_over_dir) if f.endswith('.csv')]

    # Loop through each over-by-over file and process the corresponding ball-by-ball file
    for over_by_over_file in over_by_over_files:
        # Extract the match ID from the over-by-over file name
        match_id = over_by_over_file.split('_')[1].split('.')[0]

        # Define the paths for the ball-by-ball and over-by-over files using the extracted match_id
        ball_by_ball_file = os.path.join(ball_by_ball_dir, f"innings{innings_num}_{match_id}.csv")
        over_by_over_file_path = os.path.join(over_by_over_dir, over_by_over_file)

        # Check if the corresponding ball-by-ball file exists
        if os.path.exists(ball_by_ball_file):
            # Load the ball-by-ball dataset
            ball_by_ball_df = pd.read_csv(ball_by_ball_file)
            
            # Extract the over number from the 'ball' column (before the decimal point)
            ball_by_ball_df['over_number'] = ball_by_ball_df['ball'].apply(lambda x: int(x))

            # Load the over-by-over dataset
            over_by_over_df = pd.read_csv(over_by_over_file_path)

            # Create empty lists for batters, bowlers, and balls faced
            batters = []
            bowlers = []
            balls_faced = []

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
                
                # Format the batters and ball counts as strings for easier storage
                batter_ball_counts = [f"{striker} ({striker_ball_count[striker]})" for striker in over_batters]
                
                # Append the results to the lists
                batters.append(', '.join(batter_ball_counts))
                bowlers.append(', '.join(over_bowlers))
                balls_faced.append(', '.join([str(striker_ball_count[striker]) for striker in over_batters]))

            # Add the batters, bowlers, and balls faced columns to the over-by-over DataFrame
            over_by_over_df['batters'] = batters
            over_by_over_df['bowlers'] = bowlers
            over_by_over_df['balls_faced'] = balls_faced

            # Save the updated DataFrame back to the same over-by-over file
            over_by_over_df.to_csv(over_by_over_file_path, index=False)

            # Check the updated DataFrame (optional)
            print(f"Updated data for match {match_id}, innings {innings_num}:")
            print(over_by_over_df.head())
        else:
            print(f"Ball-by-ball file for match {match_id}, innings {innings_num} not found.")

# Process innings 1 and innings 2 files
player_in_over(
    r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train",
    r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train_processed",
    innings_num=1
)

player_in_over(
    r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test",
    r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test_processed",
    innings_num=1
)

player_in_over(
    r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\train",
    r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\train_processed",
    innings_num=2
)

player_in_over(
    r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test",
    r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test_processed",
    innings_num=2
)
