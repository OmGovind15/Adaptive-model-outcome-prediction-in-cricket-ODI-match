import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. Define the folder paths
train_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train_processed"
test_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test_processed"

# 2. Load all CSV files from the folders (train, valid, test)
def load_data_from_folder(folder_path):
    all_data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            match_data = pd.read_csv(os.path.join(folder_path, file))
            match_number = file.split('.')[0]  # Assuming match number is in file name
            match_data['match_number'] = match_number
            all_data.append(match_data)
    return pd.concat(all_data, ignore_index=True)

train_data = load_data_from_folder(train_folder)
test_data = load_data_from_folder(test_folder)

# 3. Feature Engineering Function (to apply consistently to all datasets)
def feature_engineering(data):
    data['balls_remaining'] = (49 - data['over']) * 6
    data['run_rate'] = data['cumulative_runs'] / (data['over'] + 1)
    data['projected_runs'] = data['run_rate'] * 50
    data['momentum_factor'] = data['cumulative_runs'].diff(periods=3).fillna(0)
    data['pressure_index'] = 300 / ((data['wickets_remaining'] + 1)*(data['momentum_factor']+1)*(data['balls_remaining']+1))
    data['match_phase'] = pd.cut(data['over'], bins=[0, 10, 30, 50], labels=['early', 'middle', 'final'])
    match_phase_dummies = pd.get_dummies(data['match_phase'], prefix='match_phase')
    data = pd.concat([data, match_phase_dummies], axis=1)
    features = ['cumulative_runs', 'toss_result', 'wickets_remaining', 'bowling_team_win_percentage',
                'projected_runs', 'pressure_index', 'momentum_factor', 'weighted_batting_average', 'weighted_bowling_average']
    data.drop(columns=['match_phase'], inplace=True)
    return data, features

train_data, features = feature_engineering(train_data)
test_data, features = feature_engineering(test_data)
train_data.dropna(subset=['match_result'], inplace=True)
test_data.dropna(subset=['match_result'], inplace=True)

# 4. Split the data (optional, as data is already separated)
scaler = StandardScaler()
train_data[features] = scaler.fit_transform(train_data[features])
test_data[features] = scaler.transform(test_data[features])

# 6. Define models and parameter grids for hyperparameter tuning using GridSearchCV
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
param_grid_xgb = {'n_estimators': [50, 100, 150], 'max_depth': [3, 6, 10], 'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]}
param_grid_lr = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs', 'saga']}
param_grid_lda = {'solver': ['svd', 'lsqr', 'eigen'], 'shrinkage': ['auto', None]}
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
param_grid_nb = {}

models = {'RandomForest': RandomForestClassifier(random_state=42),
          'XGBoost': xgb.XGBClassifier(random_state=42),
          'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
          'LDA': LDA(),
          'SVM': SVC(probability=True, random_state=42),
          'NaiveBayes': GaussianNB()}

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
grid_search_rf = GridSearchCV(models['RandomForest'], param_grid_rf, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search_xgb = GridSearchCV(models['XGBoost'], param_grid_xgb, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search_lr = GridSearchCV(models['LogisticRegression'], param_grid_lr, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search_lda = GridSearchCV(models['LDA'], param_grid_lda, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search_svm = GridSearchCV(models['SVM'], param_grid_svm, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search_nb = GridSearchCV(models['NaiveBayes'], param_grid_nb, cv=kf, scoring='accuracy', n_jobs=-1)

# 7. Train and evaluate models for each over with tqdm for progress
over_models = {}
model_usage = {}
model_accuracies = {model: [] for model in models.keys()}
best_model_accuracies = []
cv_scores = {model_name: [] for model_name in models.keys()}
training_accuracies = {model: [] for model in models.keys()}
validation_accuracies = {model: [] for model in models.keys()}
kf = KFold(n_splits=5, shuffle=True, random_state=42) 
for over in tqdm(train_data['over'].unique(), desc="Training models for each over"):
    train_over_data = train_data[train_data['over'] == over]
    X_over = train_over_data[features].values
    y_over = train_over_data['match_result'].values

    fold_training_accuracies = {model: [] for model in models.keys()}
    fold_validation_accuracies = {model: [] for model in models.keys()}

    # Perform 5-fold cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_over)):
        X_train, X_val = X_over[train_idx], X_over[val_idx]
        y_train, y_val = y_over[train_idx], y_over[val_idx]
        
        for model_name, grid_search in zip(models.keys(), 
                                           [grid_search_rf, grid_search_xgb, grid_search_lr, 
                                            grid_search_lda, grid_search_svm, grid_search_nb]):
            # Train the model
            grid_search.fit(X_train, y_train)
            
            # Training accuracy
            train_acc = accuracy_score(y_train, grid_search.best_estimator_.predict(X_train))
            fold_training_accuracies[model_name].append(train_acc)
            
            # Validation accuracy
            val_acc = accuracy_score(y_val, grid_search.best_estimator_.predict(X_val))
            fold_validation_accuracies[model_name].append(val_acc)
    
    # Aggregate fold results for this "over"
    for model_name in models.keys():
        training_accuracies[model_name].append(fold_training_accuracies[model_name])
        validation_accuracies[model_name].append(fold_validation_accuracies[model_name])
        # Calculate mean validation accuracy and store it
        mean_val_acc = sum(fold_validation_accuracies[model_name]) / len(fold_validation_accuracies[model_name])
        cv_scores[model_name].append(mean_val_acc)
    
    # Test on separate test data
    test_over_data = test_data[test_data['over'] == over]
    X_test = test_over_data[features]
    y_test = test_over_data['match_result']

    # Determine the best model based on mean validation accuracy
    best_model = None
    best_score = 0
    best_model_name = ''
    
    for model_name, grid_search in zip(models.keys(), [grid_search_rf, grid_search_xgb, grid_search_lr, grid_search_lda, grid_search_svm, grid_search_nb]):
        mean_val_acc = cv_scores[model_name][-1]  # Use the calculated mean validation accuracy
        
        y_pred = grid_search.best_estimator_.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name].append(accuracy)
        
        if mean_val_acc > best_score:
            best_score = mean_val_acc
            best_model = grid_search.best_estimator_
            best_model_name = model_name
    
    over_models[over] = best_model
    model_usage[over] = best_model_name
    best_model_test_accuracy = model_accuracies[best_model_name][-1]
    best_model_accuracies.append(best_model_test_accuracy)

# 8. Calculate the average accuracy for each model
overall_model_accuracies = {model: np.mean(model_accuracies[model]) for model in models.keys()}
best_model_overall_accuracy = np.mean(best_model_accuracies)
# Initialize a list to store the rows for the DataFrame
results_data = []

# Map each over to its index for lookup
over_indices = {over: i for i, over in enumerate(train_data['over'].unique())}

# Iterate over all unique overs


for over in train_data['over'].unique():
    row = {'Over': over}
    over_index = over_indices[over]  # Get index of the current over

    for model_name, grid_search in zip(models.keys(), [grid_search_rf, grid_search_xgb, grid_search_lr, grid_search_lda, grid_search_svm, grid_search_nb]):
        # Validation Accuracies (list for the current over)
        if over_index < len(validation_accuracies[model_name]):
            row[f'{model_name}_Validation_Accuracies'] = validation_accuracies[model_name][over_index]  # List of 5-fold accuracies
            row[f'{model_name}_CV_Score'] = sum(validation_accuracies[model_name][over_index]) / len(validation_accuracies[model_name][over_index])  # Mean
        else:
            row[f'{model_name}_Validation_Accuracies'] = None
            row[f'{model_name}_CV_Score'] = None
    for model_name, grid_search in zip(models.keys(), [grid_search_rf, grid_search_xgb, grid_search_lr, grid_search_lda, grid_search_svm, grid_search_nb]):
        # Training Accuracies (list for the current over)
        if over_index < len(training_accuracies[model_name]):
            row[f'{model_name}_training_Accuracies'] = training_accuracies[model_name][over_index]  # List of 5-fold accuracies
            row[f'{model_name}_T_Score'] = sum(training_accuracies[model_name][over_index]) / len(training_accuracies[model_name][over_index])  # Mean
        else:
            row[f'{model_name}_training_Accuracies'] = None
            row[f'{model_name}_T_Score'] = None
        # Test accuracy (hard predictions)
        if over_index < len(model_accuracies[model_name]):
            row[f'{model_name}_Test_Accuracy'] = model_accuracies[model_name][over_index]
        else:
            row[f'{model_name}_Test_Accuracy'] = None

        # Soft accuracy (mean deviation from true label)
        test_over_data = test_data[test_data['over'] == over]
        X_test = test_over_data[features]
        y_test = test_over_data['match_result']

        if hasattr(grid_search.best_estimator_, "predict_proba"):
            y_prob = grid_search.best_estimator_.predict_proba(X_test)
            y_test_one_hot = np.eye(len(np.unique(y_test)))[y_test]
            mean_deviation = np.mean(np.abs(y_test_one_hot - y_prob))
            soft_accuracy = 1 - mean_deviation
            row[f'{model_name}_Soft_Accuracy'] = soft_accuracy

            # If this model is the best model for the current over, update Best_Model_Soft_Accuracy
            if model_name == model_usage.get(over, "Unknown"):
                row['Best_Model_Soft_Accuracy'] = soft_accuracy
        else:
            row[f'{model_name}_Soft_Accuracy'] = None
            if model_name == model_usage.get(over, "Unknown"):
                row['Best_Model_Soft_Accuracy'] = None

    # Best model and its test accuracy
    best_model_name = model_usage.get(over, "Unknown")
    row['Best_Model_Selected'] = best_model_name
    row['Best_Model_Test_Accuracy'] = (
        best_model_accuracies[over_index] if over_index < len(best_model_accuracies) else None
    )

    # Append the row to results_data
    results_data.append(row)


# Create a DataFrame from the collected data
results_df = pd.DataFrame(results_data)

# Save the DataFrame to an Excel file
results_df.to_excel('model_details.xlsx', index=False)

# Path to Excel file
excel_path = "model_details.xlsx"

# Open Excel writer in append mode (if file already exists)
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
    results_df.to_excel(writer, index=False, sheet_name='Results')

# Monitoring:
print("Data written to Excel:")
print(results_df)
print("Excel file created successfully: 'model_performance_summary.xlsx'")

# Plot overall results
plt.figure(figsize=(14, 7))
for model_name, accuracies in model_accuracies.items():
    plt.plot(train_data['over'].unique(), accuracies, label=f'{model_name} Test Accuracy')

# Plot the best model test accuracies with black dashed line
plt.plot(train_data['over'].unique(), best_model_accuracies, 'k--', label='Best Model Test Accuracy')

plt.axhline(y=best_model_overall_accuracy, color='r', linestyle='--', label=f'Best Model Overall Accuracy: {best_model_overall_accuracy:.4f}')
plt.xlabel('Over')
plt.ylabel('Accuracy')
plt.title('Model Test Accuracies per Over')
plt.legend()
plt.grid(True)
plt.show()

