import pandas as pd
import os
from tqdm import tqdm

def process_and_save_match_data(input_folder, output_folder):
    """
    Processes cricket match data to calculate cumulative runs, cumulative wickets, balls remaining, and wickets remaining 
    for each over, and saves the processed data as individual match CSV files.

    Args:
    - input_folder (str): Path to the folder containing input CSV files.
    - output_folder (str): Path to the folder to save processed match CSV files.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get all the CSV file names from the folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    # Process each CSV file
    for file in tqdm(csv_files, desc="Processing CSV files"):
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(os.path.join(input_folder, file))

        # Filter the data for Innings 1
        df_innings_1 = df[df['innings'] == 1]

        # Calculate extras for each row
        df_innings_1['extras'] = df_innings_1[['wides', 'noballs', 'byes', 'legbyes', 'penalty']].sum(axis=1)

        # Calculate total runs (runs_off_bat + extras)
        df_innings_1['total_runs'] = df_innings_1['runs_off_bat'] + df_innings_1['extras']

        # Calculate total wickets (count rows where 'wicket_type' is not null)
        df_innings_1['wickets'] = df_innings_1['wicket_type'].notnull().astype(int)

        # Extract the over number from the ball column (the integer part of the ball number)
        df_innings_1['over'] = df_innings_1['ball'].apply(lambda x: int(x))

        # Group by match_id, batting_team, bowling_team, and over to get total runs and total wickets for each over
        over_data = df_innings_1.groupby(['match_id', 'batting_team', 'bowling_team', 'over'], as_index=False).agg(
            total_runs=('total_runs', 'sum'),
            total_wickets=('wickets', 'sum')
        )
        over_data['innings'] = 1
        # Calculate cumulative runs and wickets by applying cumulative sum
        over_data['cumulative_runs'] = over_data.groupby(['match_id', 'batting_team'])['total_runs'].cumsum()
        over_data['cumulative_wickets'] = over_data.groupby(['match_id', 'batting_team'])['total_wickets'].cumsum()

        # Add the "balls_remaining" and "wickets_remaining" columns
        over_data['balls_remaining'] = (50 * 6) - ((over_data['over'])*6)-6
        over_data['wickets_remaining'] = 10 - over_data['cumulative_wickets']

        # Save the cumulative data for each match as a CSV file
        for match_id in over_data['match_id'].unique():
            match_data = over_data[over_data['match_id'] == match_id]
            match_filename = f"match_{match_id}_cumulative_data.csv"
            match_file_path = os.path.join(output_folder, match_filename)

            # Include 'batting_team', 'bowling_team', 'over', 'cumulative_runs', 'cumulative_wickets', 'balls_remaining', and 'wickets_remaining' columns
            match_data[['innings','batting_team', 'bowling_team', 'over', 'cumulative_runs', 'cumulative_wickets', 'balls_remaining', 'wickets_remaining']].to_csv(match_file_path, index=False)

# Example usage:


#process_and_save_match_data(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test", r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test_processed")
#process_and_save_match_data(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train", r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train_processed")

process_and_save_match_data(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test", r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test_processed")
process_and_save_match_data(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\train", r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train_processed")