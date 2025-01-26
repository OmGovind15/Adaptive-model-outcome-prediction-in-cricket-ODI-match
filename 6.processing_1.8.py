import os
import csv
import pandas as pd

def process_match_files(info_folder, processed_folder):
    """
    Process match files to add toss_result and match_result columns.

    Parameters:
    - info_folder (str): Path to the folder containing match info files.
    - processed_folder (str): Path to the folder containing processed match files.

    Returns:
    None
    """
    def get_toss_and_match_winner(info_file_path):
        """
        Read toss and match winner from the info file.

        Parameters:
        - info_file_path (str): Path to the info CSV file.

        Returns:
        - tuple: Toss winner and match winner.
        """
        toss_winner, match_winner = None, None
        with open(info_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0].startswith("info"):
                    if row[1] == "toss_winner":
                        toss_winner = row[2]
                    elif row[1] == "winner":
                        match_winner = row[2]
        return toss_winner, match_winner

    for file_name in os.listdir(processed_folder):
        if file_name.endswith(".csv") and "match_" in file_name:
            # Extract match number
            match_number = file_name.split('_')[1]

            # Construct paths
            info_file_path = os.path.join(info_folder, f"{match_number}_info.csv")
            processed_file_path = os.path.join(processed_folder, file_name)

            # Skip if info file doesn't exist
            if not os.path.exists(info_file_path):
                print(f"Info file not found for match {match_number}. Skipping...")
                continue

            # Get toss and match winner
            toss_winner, match_winner = get_toss_and_match_winner(info_file_path)

            # Read processed file
            df = pd.read_csv(processed_file_path)

            # Add toss_result and match_result columns
            df["toss_result"] = df["bowling_team"].apply(lambda x: 1 if x == toss_winner else 0)
            df["match_result"] = df["bowling_team"].apply(lambda x: 1 if x == match_winner else 0)

            # Save back to the same file
            df.to_csv(processed_file_path, index=False)
            print(f"Processed and updated file: {file_name}")

process_match_files(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\Top_teams\train",r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\train_processed" )
process_match_files(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\Top_teams\test",r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test_processed")