import pandas as pd
import os
from tqdm import tqdm

# Define the path to the folder containing all the files
folder_path = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train_processed"

# Load the player stats data (replace with your actual file path)
player_stats = pd.read_csv(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\career_stats_separated_innings.csv")

# Create a function to get the batting average for a player
def get_batting_average(player, player_stats):
    player_data = player_stats[player_stats['player'] == player]
    if not player_data.empty:
        runs = player_data['total_runs_innings_1'].values[0]
        matches = player_data['matches_played_innings_1'].values[0]
        return runs / matches if matches > 0 else 0
    else:
        return 0

# Create a function to calculate the weighted batting average
def get_weighted_batting_average(batting_bowler_list, balls_faced_list, player_stats):
    total_weighted_runs = 0
    total_balls = 0
    not_found_count = 0
    
    for batter, balls_faced in zip(batting_bowler_list, balls_faced_list):
        batting_avg = get_batting_average(batter, player_stats)
        total_weighted_runs += batting_avg * balls_faced  # Weighted by balls faced
        total_balls += balls_faced  # Sum of all balls faced
    
    weighted_batting_avg = total_weighted_runs / total_balls if total_balls > 0 else 0
    return weighted_batting_avg, not_found_count

# Create a function to get the bowling average
def get_bowling_average(batting_bowler_list, player_stats):
    wickets = []
    matches = []
    not_found_count = 0
    for p in batting_bowler_list:
        player_data = player_stats[player_stats['player'] == p]
        if not player_data.empty:
            wickets.append(player_data['total_wickets_innings_1'].values[0])
            matches.append(player_data['matches_played_innings_1'].values[0])
        else:
            wickets.append(0)
            matches.append(0)
            not_found_count += 1  # Increment the count for not found players
    
    total_wickets = sum(wickets)
    total_matches = sum(matches)
    bowling_avg = total_wickets / total_matches if total_matches > 0 else 0
    return bowling_avg, not_found_count

# Loop through all the files in the folder with tqdm progress bar
total_not_found_batting = 0
total_not_found_bowling = 0

files = [filename for filename in os.listdir(folder_path) if filename.endswith(".csv")]

for filename in tqdm(files, desc="Processing files", unit="file"):
    # Construct the full file path
    file_path = os.path.join(folder_path, filename)
    
    # Load the batting-bowling data
    batting_bowling_data = pd.read_csv(file_path)
    
    # Split the balls faced for each batter (assuming the balls are comma-separated in the 'balls_faced' column)
    balls_faced_list = [list(map(int, balls.split(', '))) for balls in batting_bowling_data['balls_faced']]
    
    # Add columns for weighted batting and bowling averages
    batting_bowling_data['weighted_batting_average'], not_found_batting = zip(*batting_bowling_data.apply(
        lambda row: get_weighted_batting_average(row['batters'].split(', '), balls_faced_list[batting_bowling_data.index.get_loc(row.name)], player_stats),
        axis=1
    ))

    batting_bowling_data['bowling_average'], not_found_bowling = zip(*batting_bowling_data['bowlers'].apply(
        lambda x: get_bowling_average([x], player_stats)
    ))

    # Update the total not found counts
    total_not_found_batting += sum(not_found_batting)
    total_not_found_bowling += sum(not_found_bowling)

    # Save the updated batting-bowling data back to the same CSV file
    batting_bowling_data.to_csv(file_path, index=False)

# Print the total count of players not found
print(f"\nTotal players not found for batting: {total_not_found_batting}")
print(f"Total players not found for bowling: {total_not_found_bowling}")
