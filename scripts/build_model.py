import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# load training data
print("Loading training data...")
train_df = pd.read_csv('../data/earnings_train.csv')
print(f"Training data shape: {train_df.shape}")

# prepare the data
train_df_prep = train_df.copy()
train_df_prep['DISTRICT_CODE'] = train_df_prep['DISTRICT_CODE'].fillna(0)

# separate features and target
X = train_df_prep.drop('WAGE_YEAR4', axis=1)
y = train_df_prep['WAGE_YEAR4']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# encode categorical variables
label_encoders = {}
X_encoded = X.copy()
categorical_cols = ['DISTRICT_TYPE', 'DISTRICT_NAME', 'ACADEMIC_YEAR', 'DEMO_CATEGORY', 
                     'STUDENT_POPULATION', 'AWARD_CATEGORY']

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le

print("Categorical columns encoded")

# split data
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# try linear regression
print("\nTraining Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred_val = lr_model.predict(X_val)
lr_rmse_val = np.sqrt(mean_squared_error(y_val, lr_pred_val))
lr_r2_val = r2_score(y_val, lr_pred_val)
print(f"  Validation RMSE: {lr_rmse_val:.2f}, R²: {lr_r2_val:.4f}")

# try decision tree
print("\nTraining Decision Tree...")
dt_model = DecisionTreeRegressor(random_state=42, max_depth=20, min_samples_split=10)
dt_model.fit(X_train, y_train)
dt_pred_val = dt_model.predict(X_val)
dt_rmse_val = np.sqrt(mean_squared_error(y_val, dt_pred_val))
dt_r2_val = r2_score(y_val, dt_pred_val)
print(f"  Validation RMSE: {dt_rmse_val:.2f}, R²: {dt_r2_val:.4f}")

# try knn
print("\nTraining KNN...")
k_values = [3, 5, 7, 10, 15]
best_k = 5
best_knn_rmse = float('inf')
best_knn_model = None

for k in k_values:
    knn_temp = KNeighborsRegressor(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    knn_temp_pred = knn_temp.predict(X_val)
    knn_temp_rmse = np.sqrt(mean_squared_error(y_val, knn_temp_pred))
    if knn_temp_rmse < best_knn_rmse:
        best_knn_rmse = knn_temp_rmse
        best_k = k
        best_knn_model = knn_temp

knn_model = best_knn_model
knn_pred_val = knn_model.predict(X_val)
knn_rmse_val = np.sqrt(mean_squared_error(y_val, knn_pred_val))
knn_r2_val = r2_score(y_val, knn_pred_val)
print(f"  Validation RMSE: {knn_rmse_val:.2f}, R²: {knn_r2_val:.4f} (best k={best_k})")

# compare models
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"{'Model':<20} {'Validation RMSE':<20} {'Validation R²':<15}")
print("-" * 60)
print(f"{'Linear Regression':<20} {lr_rmse_val:<20.2f} {lr_r2_val:<15.4f}")
print(f"{'Decision Tree':<20} {dt_rmse_val:<20.2f} {dt_r2_val:<15.4f}")
print(f"{'KNN (k=' + str(best_k) + ')':<20} {knn_rmse_val:<20.2f} {knn_r2_val:<15.4f}")

# select best model
models = {
    'Linear Regression': (lr_model, lr_rmse_val),
    'Decision Tree': (dt_model, dt_rmse_val),
    'KNN': (knn_model, knn_rmse_val)
}

best_model_name = min(models, key=lambda x: models[x][1])
best_model = models[best_model_name][0]
best_rmse = models[best_model_name][1]

print(f"\nBest model: {best_model_name} (Validation RMSE: {best_rmse:.2f})")

# train final model on all data
print(f"\nTraining final {best_model_name} model on all training data...")
final_model = best_model.__class__(**best_model.get_params())
final_model.fit(X_encoded, y)

# load and prepare test data
print("\nLoading test data...")
test_df = pd.read_csv('../data/earnings_test_features.csv')
print(f"Test data shape: {test_df.shape}")

test_df_prep = test_df.copy()
test_df_prep['DISTRICT_CODE'] = test_df_prep['DISTRICT_CODE'].fillna(0)

X_test = test_df_prep.copy()
for col in categorical_cols:
    le = label_encoders[col]
    known_classes = set(le.classes_)
    X_test[col] = X_test[col].astype(str).apply(
        lambda x: x if x in known_classes else le.classes_[0]
    )
    X_test[col] = le.transform(X_test[col])

# make predictions
print("Making predictions...")
test_predictions = final_model.predict(X_test)
test_predictions = np.maximum(test_predictions, 0)  # no negative wages

print(f"\nPredictions made: {len(test_predictions)}")
print(f"  Min: {test_predictions.min():.2f}")
print(f"  Max: {test_predictions.max():.2f}")
print(f"  Mean: {test_predictions.mean():.2f}")
print(f"  Median: {np.median(test_predictions):.2f}")

# save predictions
predictions_df = pd.DataFrame({'WAGE_YEAR4': test_predictions})
predictions_df.to_csv('../preds.csv', index=False)
print("\nPredictions saved to preds.csv")
print(f"\nFirst 10 predictions:")
print(predictions_df.head(10))

