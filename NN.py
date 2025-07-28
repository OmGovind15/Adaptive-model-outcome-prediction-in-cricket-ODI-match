import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import pandas as pd
from torch.utils.data import random_split
import pandas as pd
from openpyxl import load_workbook

class MatchSequenceDataset(Dataset):
    def __init__(self, folder_paths, sequence_len=50):
        if isinstance(folder_paths, str):
            folder_paths = [folder_paths]

        self.sequence_len = sequence_len
        self.samples = []

        for folder in folder_paths:
            for f in sorted(os.listdir(folder)):
                if f.endswith('.csv'):
                    full_path = os.path.join(folder, f)
                    data = self.load_file(full_path)
                    self.samples.append((f, data))  # Save filename too

    def load_file(self, file_path):
        df = pd.read_csv(file_path)

        match_result = int(df['match_result'].iloc[0])
        batting_result = 1 - match_result

        if 'toss_result' in df.columns:
            df['toss_result'] = 1 - df['toss_result']

        df = df.drop(columns=[
            'weighted_batting_average', 'weighted_bowling_average',
            'bowling_team_win_percentage',
            'bowling_team_win_percentage.1' if 'bowling_team_win_percentage.1' in df.columns else None,
            'match_result'
        ], errors='ignore')

        use_trail = int(df['innings'].iloc[0]) == 2
        df['runs_input'] = df['trail_runs'] if use_trail and 'trail_runs' in df.columns else df['cumulative_runs']

        # ---  Feature Engineering Based on Innings ---
        df['balls_remaining'] = (49 - df['over']) * 6
        df['run_rate'] = df['cumulative_runs'] / (df['over'] + 1).clip(lower=1)
        df['momentum_factor'] = df['cumulative_runs'].diff(periods=3).fillna(0)

        if int(df['innings'].iloc[0]) == 2:
            df['required_run_rate'] = df['trail_runs'] / (49 - df['over']).clip(lower=1)
            df['projected_runs'] = df['run_rate'] * 50
            df['rate_gap'] = df['required_run_rate'] - df['run_rate']
            df['pressure_index'] = (
                300 * df['rate_gap']) / ((df['wickets_remaining'] + 1) * (df['momentum_factor'] + 1) * (df['balls_remaining'] + 1))
            numeric_cols = ['trail_runs', 'wickets_remaining', 'rate_gap', 'pressure_index', 'momentum_factor']
        else:
            df['projected_runs'] = df['run_rate'] * 50
            df['pressure_index'] = (
                300) / ((df['wickets_remaining'] + 1) * (df['momentum_factor'] + 1) * (df['balls_remaining'] + 1))
            numeric_cols = ['cumulative_runs', 'wickets_remaining', 'projected_runs', 'pressure_index', 'momentum_factor']

        # ---  Feature Engineered Inputs for Model  ---
        x_numeric = torch.tensor(df[numeric_cols].values, dtype=torch.float32)

        b1 = torch.tensor(df['b1_idx'].values, dtype=torch.long)
        b2 = torch.tensor(df['b2_idx'].values, dtype=torch.long)
        bowler = torch.tensor(df['bowler_idx'].values, dtype=torch.long)

        return {
            'features': x_numeric,
            'b1': b1,
            'b2': b2,
            'bowler': bowler,
            'innings': int(df['innings'].iloc[0]),
            'batting_team': df['batting_team'].iloc[0],
            'bowling_team': df['bowling_team'].iloc[0],
            'match_result': batting_result
        }


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, item = self.samples[idx]
        return {
            'filename': filename,
            'features': item['features'],
            'b1': item['b1'],
            'b2': item['b2'],
            'bowler': item['bowler'],
            'innings': item['innings'],
            'batting_team': item['batting_team'],
            'bowling_team': item['bowling_team'],
            'match_result': item['match_result']  # ✅
        }


def pad_collate_fn(batch):
    max_len = max(item['features'].shape[0] for item in batch)

    def pad_seq(seq, pad_val=0):
        pad_size = (0, 0, 0, max_len - seq.shape[0])
        return F.pad(seq, pad_size, value=pad_val)

    features = torch.stack([pad_seq(item['features']) for item in batch])
    b1 = torch.stack([F.pad(item['b1'], (0, max_len - item['b1'].shape[0]), value=0) for item in batch])
    b2 = torch.stack([F.pad(item['b2'], (0, max_len - item['b2'].shape[0]), value=0) for item in batch])
    bowler = torch.stack([F.pad(item['bowler'], (0, max_len - item['bowler'].shape[0]), value=0) for item in batch])
    match_results = torch.tensor([item['match_result'] for item in batch], dtype=torch.float32)

    return {
        'features': features,  # [B, T, D]
        'b1': b1,
        'b2': b2,
        'bowler': bowler,
        'match_result': match_results  # [B]
    }

# Load full dataset from folders

full_dataset = MatchSequenceDataset([
    "C:\\Users\\91878\\Documents\\projects\\New folder\\odis_male_csv2\\innings1\\training_data_innings1_encoded",
    "C:\\Users\\91878\\Documents\\projects\\New folder\\odis_male_csv2\\innings2\\training_data_innings2_encoded"
])

# Split into training and validation sets (e.g., 80% train, 20% val)
train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate_fn)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate_fn)

import torch.nn as nn
from torch.utils.data import DataLoader
class WinPredictor(nn.Module):
    def __init__(self, num_players, input_dim=5, emb_dim=32, hidden_dim=64):
        super().__init__()
        self.player_emb = nn.Embedding(num_players, emb_dim)
        total_input = input_dim + emb_dim * 3

        self.mlp = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_numeric, b1_idx, b2_idx, bowler_idx):
        b1_emb = self.player_emb(b1_idx)
        b2_emb = self.player_emb(b2_idx)
        bowler_emb = self.player_emb(bowler_idx)

        x = torch.cat([x_numeric, b1_emb, b2_emb, bowler_emb], dim=-1)
        return self.mlp(x).squeeze(-1)  # [B*T]
def temporal_consistency_loss(p_win_seq):
    return ((p_win_seq[:, 1:] - p_win_seq[:, :-1]) ** 2).mean()


from tqdm import tqdm  # Add this at the top of your script

# Model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WinPredictor(num_players=1500).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
best_val_loss = float('inf')
patience = 3                # Number of epochs to wait for improvement
counter = 0                 # Count epochs without improvement
for epoch in range(25):
    # === Train ===
    model.train()
    total_train_loss = 0
    print(f"\n Epoch {epoch+1}")

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        features = batch['features'].to(device)
        b1 = batch['b1'].to(device)
        b2 = batch['b2'].to(device)
        bowler = batch['bowler'].to(device)
        labels = batch['match_result'].to(device)

        B, T, D = features.shape
        x = features.view(B*T, D)
        b1 = b1.view(B*T)
        b2 = b2.view(B*T)
        bowler = bowler.view(B*T)

        p_win = model(x, b1, b2, bowler).view(B, T)

        label_seq = labels.unsqueeze(1).expand(-1, T)
        mask = (features.abs().sum(dim=-1) > 0).float()
        bce_raw = F.binary_cross_entropy(p_win, label_seq, reduction='none')
        bce_loss = (bce_raw * mask).sum() / mask.sum()
        temp_loss = temporal_consistency_loss(p_win)

        total_batch_loss = bce_loss + 0.01 * temp_loss

        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        total_train_loss += total_batch_loss.item()

    # === Validation ===
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            b1 = batch['b1'].to(device)
            b2 = batch['b2'].to(device)
            bowler = batch['bowler'].to(device)
            labels = batch['match_result'].to(device)

            B, T, D = features.shape
            x = features.view(B*T, D)
            b1 = b1.view(B*T)
            b2 = b2.view(B*T)
            bowler = bowler.view(B*T)

            p_win = model(x, b1, b2, bowler).view(B, T)

            label_seq = labels.unsqueeze(1).expand(-1, T)
            mask = (features.abs().sum(dim=-1) > 0).float()
            bce_raw = F.binary_cross_entropy(p_win, label_seq, reduction='none')
            bce_loss = (bce_raw * mask).sum() / mask.sum()
            temp_loss = temporal_consistency_loss(p_win)

            total_val_loss += (bce_loss + 0.01 * temp_loss).item()

    # Save best model
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        torch.save(model.state_dict(), "best_model.pt")
        print(" Saved Best Model")
        counter = 0  # reset counter
    else:
        counter += 1
        print(f" No improvement. Early stopping counter: {counter}/{patience}")
        if counter >= patience:
            print(" Early stopping triggered.")
            break


    print(f"Epoch {epoch+1} | Train Loss: {total_train_loss:.4f} | Val Loss: {total_val_loss:.4f}")

test_dataset = MatchSequenceDataset([
    "C:\\Users\\91878\\Documents\\projects\\New folder\\odis_male_csv2\\innings1\\encoded_test_inning1",
    "C:\\Users\\91878\\Documents\\projects\\New folder\\odis_male_csv2\\innings2\\encoded_test_inning2"
])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate_fn)
#  Load best model before testing
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
# Initialize counters for each innings
total_correct_innings1 = 0
total_correct_innings2 = 0
total_innings1 = 0
total_innings2 = 0

model.eval()
with torch.no_grad():
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        features = sample['features'].unsqueeze(0).to(device)  # [1, T, D]
        b1 = sample['b1'].unsqueeze(0).to(device)
        b2 = sample['b2'].unsqueeze(0).to(device)
        bowler = sample['bowler'].unsqueeze(0).to(device)
        label = sample['match_result']
        innings = sample['innings']  # 1 or 2

        B, T, D = features.shape
        x = features.view(B*T, D)
        b1 = b1.view(B*T)
        b2 = b2.view(B*T)
        bowler = bowler.view(B*T)

        p_win = model(x, b1, b2, bowler).view(B, T)
        seq_len = (features.abs().sum(dim=-1) > 0).sum(dim=1)[0].item()
        final_pred = p_win[0, seq_len - 1].item()
        predicted = 1.0 if final_pred > 0.5 else 0.0

        if innings == 1:
            total_innings1 += 1
            total_correct_innings1 += (predicted == label)
        else:
            total_innings2 += 1
            total_correct_innings2 += (predicted == label)

# Print results
print("\n[TEST ACCURACY BREAKDOWN]")
print(f"Innings 1 Accuracy: {100 * total_correct_innings1 / total_innings1:.2f}% ({total_correct_innings1}/{total_innings1})")
print(f"Innings 2 Accuracy: {100 * total_correct_innings2 / total_innings2:.2f}% ({total_correct_innings2}/{total_innings2})")
total_correct = total_correct_innings1 + total_correct_innings2
total_samples = total_innings1 + total_innings2

total_accuracy = 100 * total_correct / total_samples

print(f"\nTotal Test Accuracy: {total_accuracy:.2f}% ({total_correct}/{total_samples})")


import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt

# Load best model
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Track correct and total counts per over and innings
overwise_stats = {
    1: defaultdict(lambda: {'correct': 0, 'total': 0}),
    2: defaultdict(lambda: {'correct': 0, 'total': 0})
}

with torch.no_grad():
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        features = sample['features'].unsqueeze(0).to(device)
        b1 = sample['b1'].unsqueeze(0).to(device)
        b2 = sample['b2'].unsqueeze(0).to(device)
        bowler = sample['bowler'].unsqueeze(0).to(device)
        label = sample['match_result']
        innings = sample['innings']

        T = features.shape[1]
        # Predict win probability for each over
        p_win = model(features.view(T, -1), b1.view(-1), b2.view(-1), bowler.view(-1))
        preds = (p_win > 0.5).float().cpu()

        for t in range(len(preds)):
            pred = preds[t].item()
            over = t + 1  # overs are 1-indexed
            overwise_stats[innings][over]['total'] += 1
            overwise_stats[innings][over]['correct'] += int(pred == label)

# --- Plotting ---
plt.figure(figsize=(10, 5))
for innings in [1, 2]:
    overs = sorted(overwise_stats[innings].keys())
    accuracies = [100 * overwise_stats[innings][o]['correct'] / overwise_stats[innings][o]['total'] for o in overs]
    plt.plot(overs, accuracies, label=f"Innings {innings}")

plt.title("Over-wise Test Accuracy per Innings")
plt.xlabel("Over Number")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()
plt.xticks(range(1, 51))
plt.tight_layout()
plt.show()

# --- Group innings by match ID ---
from collections import defaultdict

# Group files by match ID 
match_groups = defaultdict(dict)
for i in range(len(train_dataset)):
    sample = train_dataset[i]
    match_id = sample['filename'].split('_')[1]
    innings = sample['innings']
    match_groups[match_id][innings] = sample

# --- Pick one match with both innings ---
for match_id, innings_dict in match_groups.items():
    if 1 in innings_dict and 2 in innings_dict:
        sample1 = innings_dict[1]
        sample2 = innings_dict[2]
        break  # Stop after first full match

import joblib

player_encoder = joblib.load("C:\\Users\\91878\\Downloads\\player_encoder_combined.pkl")

player_id_to_name = {i: name for i, name in enumerate(player_encoder.classes_)}

import shap
import numpy as np
import matplotlib.pyplot as plt

# ---- SHAP-compatible wrapper ----
def shap_model_wrapper(x_full):
    x_numeric = torch.tensor(x_full[:, :5], dtype=torch.float32).to(device)
    b1 = torch.tensor(x_full[:, 5], dtype=torch.long).to(device)
    b2 = torch.tensor(x_full[:, 6], dtype=torch.long).to(device)
    bowler = torch.tensor(x_full[:, 7], dtype=torch.long).to(device)

    with torch.no_grad():
        return model(x_numeric, b1, b2, bowler).cpu().numpy()
def build_phase_aware_background(train_dataset, player_id_to_name, overs_per_phase=5):
    phase_indices = {
        "early": list(range(0, 10)),
        "middle": list(range(10, 35)),
        "death": list(range(35, 50))
    }

    background_data = []

    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        features = sample['features']
        b1 = sample['b1']
        b2 = sample['b2']
        bowler = sample['bowler']

        for phase, indices in phase_indices.items():
            sampled = np.random.choice(indices, min(overs_per_phase, len(indices)), replace=False)
            for idx in sampled:
                if idx >= len(features):  # skip if sample too short
                    continue
                f = features[idx]
                x = torch.cat([
                    f.unsqueeze(0),
                    torch.tensor([[b1[idx].item(), b2[idx].item(), bowler[idx].item()]], dtype=torch.float32)
                ], dim=1)
                background_data.append(x)

    if not background_data:
        raise ValueError("No background data could be generated.")

    background_tensor = torch.cat(background_data, dim=0).numpy()  # shape: [N, 8]
    print(f" Background samples: {background_tensor.shape}")
    return background_tensor

# ---- Function to run SHAP and visualize for all overs ----
import os

# Define file path once (globally or at the top)
excel_path = "shap_explanation_outputs.xlsx"



def explain_all_overs(sample, background_data, innings_name="Innings"):
    num_overs = sample['features'].shape[0]
    static_feature_names = [
        'cumulative_runs', 'wickets_remaining', 'projected_runs',
        'pressure_index', 'momentum_factor'
    ]
    shap_data = []
    player_shap_contrib = {}
    shap_detailed_rows = []


    for over_idx in range(num_overs):
        x_numeric = sample['features'][over_idx].unsqueeze(0)
        b1 = sample['b1'][over_idx].item()
        b2 = sample['b2'][over_idx].item()
        bowler = sample['bowler'][over_idx].item()

        x_shap = torch.cat([
            x_numeric,
            torch.tensor([[b1, b2, bowler]], dtype=torch.float32)
        ], dim=1).numpy()

        # Explain
        explainer = shap.KernelExplainer(shap_model_wrapper, background_data)
        shap_values = explainer.shap_values(x_shap)
        values = shap_values[0]  # shape: (8,)

        pred = shap_model_wrapper(x_shap)[0]
        shap_data.append({
            'over': over_idx + 1,
            'shap_static': values[:5],
            'shap_players': values[5:],
            'prediction': pred
        })
                # Add detailed SHAP row for each role
        for role, pid, shap_val in zip(['b1', 'b2', 'bowler'], [b1, b2, bowler], values[5:]):
            pname = player_id_to_name.get(pid, f"player_{pid}")
            shap_detailed_rows.append({
                "Over": over_idx + 1,
                "Role": role,
                "Player": pname,
                "Player_ID": pid,
                "SHAP_Value": shap_val,
                "Prediction": pred,
                "Innings": innings_name
            })
        


        # Print raw SHAP values
        print(f"\n[SHAP] {innings_name} - Over {over_idx+1:02d}")
        print(f"  Prediction: {pred:.4f}")
       

        # Accumulate for bar plot
        for i, pid in enumerate([b1, b2, bowler]):
            pname = player_id_to_name.get(pid, f"player_{pid}")
            if pname not in player_shap_contrib:
                player_shap_contrib[pname] = []
            player_shap_contrib[pname].append(values[5+i])

    # --- Plotting (as before, unchanged) ---
    shap_matrix = np.stack([d['shap_static'] for d in shap_data])
    overs = np.array([d['over'] for d in shap_data])
    preds = np.array([d['prediction'] for d in shap_data])
    static_df = pd.DataFrame(shap_matrix, columns=static_feature_names)
    static_df.insert(0, "Over", overs)
    import seaborn as sns

    # Pivot static_df to make features the rows and overs the columns
    static_heatmap_df = static_df.set_index("Over").T

    # Optional: center the colormap at 0
    plt.figure(figsize=(16, 3.5))
    sns.heatmap(
        static_heatmap_df,
        cmap="RdYlGn",
        center=0,
        linewidths=0.3,
        cbar_kws={'label': 'SHAP Value (Impact on Win Probability)'}
    )
    plt.title(f"SHAP Heatmap: Static Feature Impact Per Over - {innings_name}")
    plt.xlabel("Over Number")
    plt.ylabel("Static Feature")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    for i, name in enumerate(static_feature_names):
        plt.plot(overs, shap_matrix[:, i], label=name)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"SHAP Contributions (Static Features) - {innings_name}")
    plt.xlabel("Over")
    plt.ylabel("SHAP Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(overs, preds, label="Win Probability")
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.title(f"Win Probability Over Time - {innings_name}")
    plt.xlabel("Over")
    plt.ylabel("Win Probability")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    # --- Save all player SHAPs to Excel ---

    player_df = pd.DataFrame([
        {"Player": pname, "Mean_SHAP": np.mean(vals), "Total_Overs": len(vals)}
        for pname, vals in player_shap_contrib.items()
    ])

    avg_shap = {k: np.mean(v) for k, v in player_shap_contrib.items()}
    top_players = sorted(avg_shap.items(), key=lambda x: -abs(x[1]))[:10]

    plt.figure(figsize=(8, 4))
    plt.barh([p[0] for p in top_players], [p[1] for p in top_players])
    plt.title(f"Top 10 Player Contributions - {innings_name}")
    plt.xlabel("Avg SHAP Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    excel_path = "shap_explanation_outputs.xlsx"
    sheet_name = f"{innings_name}_Player_SHAP".replace(" ", "_")
    static_sheet_name = f"{innings_name}_Static_SHAP".replace(" ", "_")
    shap_detailed_df = pd.DataFrame(shap_detailed_rows)
# ---  Generate Over × Player Heatmap ---
    import seaborn as sns

        # Pivot to get Player × Over matrix
    heatmap_df = shap_detailed_df.pivot_table(
            index="Player", columns="Over", values="SHAP_Value", aggfunc='sum', fill_value=0
        )

        # Optional: sort by total impact
    heatmap_df["TotalImpact"] = heatmap_df.sum(axis=1)
    heatmap_df = heatmap_df.sort_values("TotalImpact", ascending=False).drop("TotalImpact", axis=1)

        # Plot
    plt.figure(figsize=(16, len(heatmap_df) * 0.5))
    sns.heatmap(
            heatmap_df,
            cmap="RdYlGn",
            center=0,
            linewidths=0.3,
            cbar_kws={'label': 'SHAP Value (Impact on Win Probability)'}
        )
    plt.title(f"SHAP Heatmap: Player Impact Per Over - {innings_name}")
    plt.xlabel("Over Number")
    plt.ylabel("Player")
    plt.tight_layout()
    plt.show()
    detailed_sheet_name = f"{innings_name}_SHAP_Detail".replace(" ", "_")
    with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        player_df.to_excel(writer, sheet_name=sheet_name, index=False)
        static_df.to_excel(writer, sheet_name=static_sheet_name, index=False)
        shap_detailed_df.to_excel(writer, sheet_name=detailed_sheet_name, index=False)



# --- Predict win probabilities ---
def predict_win_prob(sample, model, device):
    model.eval()
    with torch.no_grad():
        features = sample['features'].unsqueeze(0).to(device)
        b1 = sample['b1'].unsqueeze(0).to(device)
        b2 = sample['b2'].unsqueeze(0).to(device)
        bowler = sample['bowler'].unsqueeze(0).to(device)

        T = features.shape[1]
        preds = model(features.view(T, -1), b1.view(-1), b2.view(-1), bowler.view(-1))
        return preds.detach().cpu().numpy()

# Predict
p1 = predict_win_prob(sample1, model, device)
p2 = predict_win_prob(sample2, model, device)
# --- Print Full Match Info ---
print(f"\n Match ID: {match_id}")
print(f"Innings 1 → Batting: {sample1['batting_team']} | Bowling: {sample1['bowling_team']}")
print(f"Innings 2 → Batting: {sample2['batting_team']} | Bowling: {sample2['bowling_team']}")
print(f"Match Result: {'WIN' if sample2['match_result'] == 1 else 'LOSS'} for {sample2['batting_team']} (2nd innings batting team)")

# Print probabilities
print("\n Win Probabilities Over Overs:")
print(f"Innings 1 ({sample1['batting_team']}):")
for i, prob in enumerate(p1):
    print(f"  Over {i+1:2d}: {prob:.4f}")

print(f"\nInnings 2 ({sample2['batting_team']}):")
for i, prob in enumerate(p2):
    print(f"  Over {i+1:2d}: {prob:.4f}")

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(p1, label=f"Innings 1: {sample1['batting_team']} vs {sample1['bowling_team']}")
plt.plot(p2, label=f"Innings 2: {sample2['batting_team']} vs {sample2['bowling_team']}")
plt.axhline(0.5, color='gray', linestyle='--', label='Neutral (0.5)')
plt.xlabel("Over")
plt.ylabel("Win Probability")
plt.title(f"Win Probability Over Time - Match {match_id}")
plt.legend()
plt.grid()
plt.show()

background_data = build_phase_aware_background(train_dataset, player_id_to_name, overs_per_phase=5)
import openpyxl

excel_path = "shap_explanation_outputs.xlsx"

# Ensure the Excel file exists and has at least one sheet
if not os.path.exists(excel_path):
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Create an initial dummy sheet to satisfy openpyxl's requirement
        pd.DataFrame({"Init": [0]}).to_excel(writer, sheet_name="Init", index=False)

explain_all_overs(sample1, background_data, innings_name=f"Innings 1 - {sample1['batting_team']}")
explain_all_overs(sample2, background_data, innings_name=f"Innings 2 - {sample2['batting_team']}")

# --- Player Embedding Clustering for All Players in the Match ---
def get_unique_players_from_match(sample_innings1, sample_innings2):
    all_ids = torch.cat([
        sample_innings1['b1'], sample_innings1['b2'], sample_innings1['bowler'],
        sample_innings2['b1'], sample_innings2['b2'], sample_innings2['bowler']
    ])
    return list(set(all_ids.tolist()))
def get_embeddings_for_players(model, player_ids):
    device = next(model.parameters()).device
    ids_tensor = torch.tensor(player_ids, dtype=torch.long).to(device)
    embeddings = model.player_emb(ids_tensor).detach().cpu().numpy()
    return embeddings

def visualize_player_embeddings(player_ids, embeddings, player_id_to_name):
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(player_ids)-1))
    reduced = tsne.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=min(5, len(player_ids)), random_state=42).fit(embeddings)
    labels = kmeans.labels_

    plt.figure(figsize=(12, 6))
    used_labels = set()
    for i, pid in enumerate(player_ids):
        name = player_id_to_name.get(pid, f"Player {pid}")
        label = f"Cluster {labels[i]}"
        plt.scatter(reduced[i, 0], reduced[i, 1], c=f"C{labels[i]}", label=label if label not in used_labels else None)
        plt.text(reduced[i, 0]+0.3, reduced[i, 1], name, fontsize=8)
        used_labels.add(label)

    plt.title("Player Embedding Clusters (Match-level)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

# --- Run the clustering ---
player_ids = get_unique_players_from_match(sample1, sample2)
embeddings = get_embeddings_for_players(model, player_ids)
visualize_player_embeddings(player_ids, embeddings, player_id_to_name)
