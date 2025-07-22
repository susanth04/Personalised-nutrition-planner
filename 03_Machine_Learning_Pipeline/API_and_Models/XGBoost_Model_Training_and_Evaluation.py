import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Load dataset
data = pd.read_csv(r"C:\RESEARCH-PROJECT\IHMP\DONE\integrated_multiomics_dataset.tsv", sep='\t')
print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")

# Check for zeros in nutrient columns
zero_calories = data[data['avg_daily_calories'] == 0]
if len(zero_calories) > 0:
    print(f"Warning: {len(zero_calories)} rows with avg_daily_calories = 0")
    # Optional: Replace zeros with mean (uncomment to apply)
    # data['avg_daily_calories'] = data['avg_daily_calories'].replace(0, data['avg_daily_calories'].mean())

# Define features and target
exclude_cols = ['Dataset', 'Sample', 'Subject', 'Study.Group', 'Gender', 'smoking status',
               'condition_numeric', 'normalized_inflammation', 
               'breakfast_day1', 'breakfast_day2', 'breakfast_day3',
               'lunch_day1', 'lunch_day2', 'lunch_day3',
               'dinner_day1', 'dinner_day2', 'dinner_day3']
features = [col for col in data.columns if col not in exclude_cols]
X = data[features]
y = data['normalized_inflammation']

# Split data, preserving indices
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_indices = X_train.index
test_indices = X_test.index

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define XGBoost model and parameter grid
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV MAE: {-grid_search.best_score_}")

# Predict and evaluate
y_pred = best_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MAE: {mae}")
print(f"Test R²: {r2}")

# Save model
best_model.save_model("xgboost_nutrition_tuned.json")
print("Model saved as 'xgboost_nutrition_tuned.json'")

# Feature importance with column names
importance = best_model.get_booster().get_score(importance_type='gain')
# Map f0, f1, ... to feature names
importance_mapped = {features[int(k.replace('f', ''))]: v for k, v in importance.items()}
sorted_importance = sorted(importance_mapped.items(), key=lambda x: x[1], reverse=True)
print("\nTop 5 Feature Importances:")
for feature, score in sorted_importance[:5]:
    print(f"{feature}: {score}")

# Save predictions
test_results = pd.DataFrame({
    'Sample': data.loc[test_indices, 'Sample'],
    'True_normalized_inflammation': y_test.values,
    'Predicted_normalized_inflammation': y_pred
})
test_results.to_csv("xgboost_predictions.csv", index=False)
print("Predictions saved as 'xgboost_predictions.csv'")