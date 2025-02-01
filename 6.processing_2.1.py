import pandas as pd
import os
from tqdm import tqdm

def process_and_save_match_data(input_folder_innings1, input_folder_innings2, output_folder):
    """
    Processes cricket match data for both innings to calculate trail runs for the second innings 
    and saves the processed data as individual match CSV files.

    Args:
    - input_folder_innings1 (str): Path to the folder containing input CSV files for innings 1.
    - input_folder_innings2 (str): Path to the folder containing input CSV files for innings 2.
    - output_folder (str): Path to the folder to save processed match CSV files.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get all CSV files from both innings
    csv_files_innings1 = [f for f in os.listdir(input_folder_innings1) if f.endswith('.csv')]
    csv_files_innings2 = [f for f in os.listdir(input_folder_innings2) if f.endswith('.csv')]

    # Process innings 1 files
    match_total_runs = {}
    for file in tqdm(csv_files_innings1, desc="Processing Innings 1 CSV files"):
        # Load innings 1 data
        df_innings1 = pd.read_csv(os.path.join(input_folder_innings1, file))

        # Calculate cumulative runs in innings 1
        df_innings1['extras'] = df_innings1[['wides', 'noballs', 'byes', 'legbyes', 'penalty']].sum(axis=1)
        df_innings1['total_runs'] = df_innings1['runs_off_bat'] + df_innings1['extras']
        match_id = df_innings1['match_id'].iloc[0]

        # Get total runs scored in the first innings
        total_runs_innings1 = df_innings1['total_runs'].sum()
        match_total_runs[match_id] = total_runs_innings1

    # Process innings 2 files
    for file in tqdm(csv_files_innings2, desc="Processing Innings 2 CSV files"):
        # Load innings 2 data
        df_innings2 = pd.read_csv(os.path.join(input_folder_innings2, file))

        # Calculate extras and total runs for innings 2
        df_innings2['extras'] = df_innings2[['wides', 'noballs', 'byes', 'legbyes', 'penalty']].sum(axis=1)
        df_innings2['total_runs'] = df_innings2['runs_off_bat'] + df_innings2['extras']
        df_innings2['wickets'] = df_innings2['wicket_type'].notnull().astype(int)
        df_innings2['over'] = df_innings2['ball'].apply(lambda x: int(x))

        # Group by over to calculate total runs and wickets for the second innings
        over_data = df_innings2.groupby(['match_id', 'batting_team', 'bowling_team', 'over'], as_index=False).agg(
            total_runs=('total_runs', 'sum'),
            total_wickets=('wickets', 'sum')
        )
        over_data['innings'] = 2
        over_data['cumulative_runs'] = over_data.groupby(['match_id', 'batting_team'])['total_runs'].cumsum()
        over_data['cumulative_wickets'] = over_data.groupby(['match_id', 'batting_team'])['total_wickets'].cumsum()

        # Add balls remaining and wickets remaining
        over_data['balls_remaining'] = (50 * 6) - (over_data['over'] * 6) - 6
        over_data['wickets_remaining'] = 10 - over_data['cumulative_wickets']

        # Add trail runs
        match_id = over_data['match_id'].iloc[0]
        if match_id in match_total_runs:
            total_runs_innings1 = match_total_runs[match_id]
            over_data['trail_runs'] = total_runs_innings1 - over_data['cumulative_runs']
        else:
            over_data['trail_runs'] = None  # Handle cases where no innings 1 data is found

        # Save the processed second innings data
        for match_id in over_data['match_id'].unique():
            match_data = over_data[over_data['match_id'] == match_id]
            match_filename = f"match_{match_id}_innings2_data.csv"
            match_file_path = os.path.join(output_folder, match_filename)

            match_data[['innings', 'batting_team', 'bowling_team', 'over', 'cumulative_runs',
                        'cumulative_wickets', 'balls_remaining', 'wickets_remaining', 'trail_runs']].to_csv(
                match_file_path, index=False
            )


process_and_save_match_data(
    input_folder_innings1=r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train",
    input_folder_innings2=r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\train",
    output_folder=r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\train_processed"
)
process_and_save_match_data(
    input_folder_innings1=r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test",
    input_folder_innings2=r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test",
    output_folder=r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test_processed"
)
