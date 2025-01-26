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
    
    # One-hot encode 'match_phase'
    match_phase_dummies = pd.get_dummies(data['match_phase'], prefix='match_phase')
    data = pd.concat([data, match_phase_dummies], axis=1)

    
    features = [
        'cumulative_runs', 'toss_result', 
        'wickets_remaining', 'bowling_team_win_percentage',
        'projected_runs', 'pressure_index', 'momentum_factor', 'weighted_batting_average', 'weighted_bowling_average'
    ]
    
    # Remove the original categorical column
    data.drop(columns=['match_phase'], inplace=True)
    
    return data, features

# Apply feature engineering to all datasets
train_data, features = feature_engineering(train_data)
  # Reuse feature engineering function without redoing the feature list
test_data, _ = feature_engineering(test_data)   # Same as above

# Drop missing values
target = 'match_result'
train_data.dropna(subset=[target], inplace=True)

test_data.dropna(subset=[target], inplace=True)

# 4. Split the data (optional, as data is already separated)
# unique_matches = data['match_number'].unique()
# train_matches, temp_matches = train_test_split(unique_matches, test_size=0.3, random_state=42)
# valid_matches, test_matches = train_test_split(temp_matches, test_size=0.5, random_state=42)

# 5. Apply scaling to all datasets using the same scaler
scaler = StandardScaler()
train_data[features] = scaler.fit_transform(train_data[features])

test_data[features] = scaler.transform(test_data[features])

# 6. Define models and parameter grids for hyperparameter tuning using GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

param_grid_xgb = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]  # Fraction of features used for training each tree
}

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],  # L2 regularization
    'solver': ['lbfgs', 'saga']  # Solvers suitable for multi-class classification
}

param_grid_lda = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': ['auto', None]
}

param_grid_svm = {
    'C': [0.1, 1, 10],              # Regularization parameter
    'kernel': ['linear', 'rbf'],    # Kernel types
    'gamma': ['scale', 'auto']      # Kernel coefficient
}

param_grid_nb = {
    # No tunable hyperparameters for GaussianNB; keeping empty grid for consistency
}

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'LDA': LDA(),
    'SVM': SVC(probability=True, random_state=42),3
    'NaiveBayes': GaussianNB()
}
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
# Initialize GridSearchCV for each model
grid_search_rf = GridSearchCV(models['RandomForest'], param_grid_rf, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search_xgb = GridSearchCV(models['XGBoost'], param_grid_xgb, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search_lr = GridSearchCV(models['LogisticRegression'], param_grid_lr, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search_lda = GridSearchCV(models['LDA'], param_grid_lda, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search_svm = GridSearchCV(models['SVM'], param_grid_svm, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search_nb = GridSearchCV(models['NaiveBayes'], param_grid_nb, cv=kf, scoring='accuracy', n_jobs=-1)

# 7. Train and evaluate models for each over with tqdm for progress
over_models = {}
model_usage = {}  # Track which model is used for which over
model_accuracies = {model: [] for model in ['RandomForest', 'XGBoost', 'LogisticRegression', 'LDA', 'SVM', 'NaiveBayes']}
best_model_accuracies = []
cv_scores = {model_name: [] for model_name in models.keys()}  # Store cross-validation scores
for over in tqdm(train_data['over'].unique(), desc="Training models for each over"):
    train_over_data = train_data[train_data['over'] == over]
    X_train = train_over_data[features]
    y_train = train_over_data[target]
    
    # Train each model on the current over's training data
    grid_search_rf.fit(X_train, y_train)
    grid_search_xgb.fit(X_train, y_train)
    grid_search_lr.fit(X_train, y_train)
    grid_search_lda.fit(X_train, y_train)
    grid_search_svm.fit(X_train, y_train)
    grid_search_nb.fit(X_train, y_train)

    # Evaluate each model on the test set
    test_over_data = test_data[test_data['over'] == over]
    X_test = test_over_data[features]
    y_test = test_over_data[target]
    
    
    best_model = None
    best_score = 0
    best_model_name = ''
    
    for model_name, grid_search in zip(['RandomForest', 'XGBoost', 'LogisticRegression', 'SVM', 'LDA', 'NaiveBayes'],
                                       [grid_search_rf, grid_search_xgb, grid_search_lr, grid_search_svm, grid_search_lda, grid_search_nb]):
        mean_cv_score = grid_search.best_score_
        cv_scores[model_name].append(mean_cv_score)
        
        
        y_pred = grid_search.best_estimator_.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name].append(accuracy)
        
        
        
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model = grid_search.best_estimator_
            best_model_name = model_name
    
    over_models[over] = best_model
    model_usage[over] = best_model_name  # Store which model was used for the over
    best_model_test_accuracy = model_accuracies[best_model_name][-1]  # Get the test accuracy for the selected best model
    best_model_accuracies.append(best_model_test_accuracy)  # Store the test accuracy of the best model
    print(cv_scores)
    print(model_accuracies)
    print(f"Best model for Over {over}: {best_model_name} (valid Accuracy: {best_score:.4f}, Test accuracy: {best_model_test_accuracy:.4f})")
    
overall_model_accuracies = {model: np.mean(model_accuracies[model]) for model in model_accuracies}
best_model_overall_accuracy = np.mean(best_model_accuracies)
# 3. Print the overall accuracy of each model on the test set
print("\nOverall Accuracy of Each Model on Test Set:")
for model_name, accuracy in overall_model_accuracies.items():
    print(f"{model_name}: {accuracy:.4f}")

# 4. Print the overall accuracy of the best model on the validation set
print(f"\nOverall Accuracy of the Best Model on Validation Set: {best_model_overall_accuracy:.4f}")

# 5. Print the model_accuracies and best_model_accuracies
print("\nModel Accuracies for Each Model on Test Set:")
for model_name, accuracies in model_accuracies.items():
    print(f"{model_name}: {accuracies}")

print("\nBest Model Accuracies:")
print(best_model_accuracies)
# 5. Plot both model_accuracies and best_model_accuracies on the same plot
plt.figure(figsize=(10, 6))

# Plot model accuracies for each model
for model_name, accuracy_list in model_accuracies.items():
    plt.plot(range(len(accuracy_list)), accuracy_list, label=model_name)

# Plot best model accuracies
plt.plot(range(len(best_model_accuracies)), best_model_accuracies, label='Best Model', linestyle='--', color='black')

# Add labels and title
plt.xlabel('Overs')
plt.ylabel('Accuracy')
plt.title('Model Accuracies for Each Over and Best Model Accuracy')
plt.legend(loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
