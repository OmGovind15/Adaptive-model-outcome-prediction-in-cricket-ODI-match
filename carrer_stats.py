import os
import pandas as pd
from tqdm import tqdm

# Function to load and process each match file
def process_match_file(file_path):
    # Load the match data from the CSV file
    df = pd.read_csv(file_path)
    
    # Convert ball numbers to over and ball number
    def convert_ball_to_over_and_ball(ball):
        over = int(ball)  # Add 1 to the integer part to get the over number
        ball_in_over = round((ball - int(ball)) * 10)  # Decimal part represents the ball in that over
        return over, ball_in_over

    # Apply the function to extract over and ball
    df['over'], df['ball_in_over'] = zip(*df['ball'].apply(convert_ball_to_over_and_ball))
    
    # Extract match_id directly from the dataframe (assuming the match_id is present in the first row of each match)
    match_id = df['match_id'].iloc[0]

    # Separate Wickets and other player stats for each innings
    wickets_innings_1 = df[df['player_dismissed'].notnull() & (df['innings'] == 1)][['match_id', 'innings', 'player_dismissed', 'over', 'ball_in_over', 'wicket_type']]
    wickets_innings_2 = df[df['player_dismissed'].notnull() & (df['innings'] == 2)][['match_id', 'innings', 'player_dismissed', 'over', 'ball_in_over', 'wicket_type']]

    # Runs per player for each innings
    runs_per_player_innings_1 = df[df['innings'] == 1].groupby('striker')['runs_off_bat'].sum().reset_index()
    runs_per_player_innings_2 = df[df['innings'] == 2].groupby('striker')['runs_off_bat'].sum().reset_index()

    # Total Runs and Wickets per Over for each innings
    total_over_stats_innings_1 = df[df['innings'] == 1].groupby('over').agg({'runs_off_bat': 'sum', 'player_dismissed': 'count'}).reset_index()
    total_over_stats_innings_2 = df[df['innings'] == 2].groupby('over').agg({'runs_off_bat': 'sum', 'player_dismissed': 'count'}).reset_index()

    # Renaming columns for clarity
    total_over_stats_innings_1.columns = ['over', 'total_runs', 'total_wickets']
    total_over_stats_innings_2.columns = ['over', 'total_runs', 'total_wickets']

    # Bowling Statistics per Bowler
    bowling_stats_innings_1 = df[df['innings'] == 1].groupby('bowler').agg({
        'runs_off_bat': 'sum', 
        'player_dismissed': 'count', 
        'ball': 'count'}).reset_index()
    bowling_stats_innings_2 = df[df['innings'] == 2].groupby('bowler').agg({
        'runs_off_bat': 'sum', 
        'player_dismissed': 'count', 
        'ball': 'count'}).reset_index()

    # Renaming columns for clarity
    bowling_stats_innings_1.columns = ['bowler', 'total_runs_conceded', 'wickets_taken', 'balls_bowled']
    bowling_stats_innings_2.columns = ['bowler', 'total_runs_conceded', 'wickets_taken', 'balls_bowled']

    # Return the statistics for this match
    return {
        'match_id': match_id,
        'wickets_innings_1': wickets_innings_1,
        'wickets_innings_2': wickets_innings_2,
        'runs_per_player_innings_1': runs_per_player_innings_1,
        'runs_per_player_innings_2': runs_per_player_innings_2,
        'total_over_stats_innings_1': total_over_stats_innings_1,
        'total_over_stats_innings_2': total_over_stats_innings_2,
        'bowling_stats_innings_1': bowling_stats_innings_1,
        'bowling_stats_innings_2': bowling_stats_innings_2
    }

# Define the folder paths for innings 1 and innings 2
innings1_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train"
innings2_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\train"

# Initialize empty dictionaries to hold aggregated statistics for each innings
career_batting_stats_innings_1 = {}
career_batting_stats_innings_2 = {}
career_bowling_stats_innings_1 = {}
career_bowling_stats_innings_2 = {}
career_matches_played_innings_1 = {}
career_matches_played_innings_2 = {}

# Initialize a dictionary to track which matches the player has already been counted for each innings
matches_played_tracker_innings_1 = {}
matches_played_tracker_innings_2 = {}

# Iterate through the files in the innings1 and innings2 folders
for folder in [innings1_folder, innings2_folder]:
    for file_name in tqdm(os.listdir(folder), desc="Processing matches", unit="match"):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder, file_name)

            # Process the current match file
            match_stats = process_match_file(file_path)

            match_id= match_stats['match_id']

            # Aggregate Runs and Wickets for each Player for Innings 1
            for player in match_stats['runs_per_player_innings_1']['striker']:
                runs = match_stats['runs_per_player_innings_1'][match_stats['runs_per_player_innings_1']['striker'] == player]['runs_off_bat'].values[0]
                career_batting_stats_innings_1[player] = career_batting_stats_innings_1.get(player, 0) + runs

                # Mark the player as having played this match if not already counted
                if player not in matches_played_tracker_innings_1:
                    matches_played_tracker_innings_1[player] = set()
                matches_played_tracker_innings_1[player].add(match_id)

            for player in match_stats['runs_per_player_innings_2']['striker']:
                runs = match_stats['runs_per_player_innings_2'][match_stats['runs_per_player_innings_2']['striker'] == player]['runs_off_bat'].values[0]
                career_batting_stats_innings_2[player] = career_batting_stats_innings_2.get(player, 0) + runs

                # Mark the player as having played this match if not already counted
                if player not in matches_played_tracker_innings_2:
                    matches_played_tracker_innings_2[player] = set()
                matches_played_tracker_innings_2[player].add(match_id)

            # Aggregate Wickets for each Player (Bowling) for Innings 1
            for player in match_stats['bowling_stats_innings_1']['bowler']:
                wickets = match_stats['bowling_stats_innings_1'][match_stats['bowling_stats_innings_1']['bowler'] == player]['wickets_taken'].values[0]
                career_bowling_stats_innings_1[player] = career_bowling_stats_innings_1.get(player, 0) + wickets

                # Mark the player as having played this match if not already counted
                if player not in matches_played_tracker_innings_1:
                    matches_played_tracker_innings_1[player] = set()
                matches_played_tracker_innings_1[player].add(match_id)

            # Aggregate Wickets for each Player (Bowling) for Innings 2
            for player in match_stats['bowling_stats_innings_2']['bowler']:
                wickets = match_stats['bowling_stats_innings_2'][match_stats['bowling_stats_innings_2']['bowler'] == player]['wickets_taken'].values[0]
                career_bowling_stats_innings_2[player] = career_bowling_stats_innings_2.get(player, 0) + wickets

                # Mark the player as having played this match if not already counted
                if player not in matches_played_tracker_innings_2:
                    matches_played_tracker_innings_2[player] = set()
                matches_played_tracker_innings_2[player].add(match_id)

# Convert the aggregated statistics into DataFrames for each innings
batting_df_innings_1 = pd.DataFrame(list(career_batting_stats_innings_1.items()), columns=['player', 'total_runs_innings_1'])
batting_df_innings_2 = pd.DataFrame(list(career_batting_stats_innings_2.items()), columns=['player', 'total_runs_innings_2'])

bowling_df_innings_1 = pd.DataFrame(list(career_bowling_stats_innings_1.items()), columns=['player', 'total_wickets_innings_1'])
bowling_df_innings_2 = pd.DataFrame(list(career_bowling_stats_innings_2.items()), columns=['player', 'total_wickets_innings_2'])

# Calculate the number of matches played for each player by counting unique match_ids for each innings
matches_played_df_innings_1 = pd.DataFrame([(player, len(matches)) for player, matches in matches_played_tracker_innings_1.items()], columns=['player', 'matches_played_innings_1'])
matches_played_df_innings_2 = pd.DataFrame([(player, len(matches)) for player, matches in matches_played_tracker_innings_2.items()], columns=['player', 'matches_played_innings_2'])

# Merge the DataFrames for innings 1 and innings 2
final_stats_innings_1 = pd.merge(batting_df_innings_1, bowling_df_innings_1, on='player', how='outer')
final_stats_innings_1 = pd.merge(final_stats_innings_1, matches_played_df_innings_1, on='player', how='outer')

final_stats_innings_2 = pd.merge(batting_df_innings_2, bowling_df_innings_2, on='player', how='outer')
final_stats_innings_2 = pd.merge(final_stats_innings_2, matches_played_df_innings_2, on='player', how='outer')

# Merge stats from both innings
final_stats = pd.merge(final_stats_innings_1, final_stats_innings_2, on='player', how='outer')

# Display or save the aggregated career statistics
print("\nCareer Statistics (Innings 1 and 2):")
print(final_stats)

# Optionally, save to CSV
final_stats.to_csv(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\career_stats_separated_innings.csv", index=False)
