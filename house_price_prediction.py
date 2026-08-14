# ============================================================
# HOUSE PRICE PREDICTION
# CloudExify Data Science - Month 2 Project 3
# ============================================================

# ============================================================
# Step 1: Import Libraries
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# ============================================================
# Step 2: Load and Explore Data
# ============================================================

df = pd.read_csv('house_prices.csv')

print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)

print(f"Total Rows:    {df.shape[0]:,}")
print(f"Total Columns: {df.shape[1]}")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df[
    [
        'price',
        'bedrooms',
        'baths',
        'Area Size',
        'Area Type',
        'city',
        'property_type',
        'purpose'
    ]
].isnull().sum())


# ============================================================
# Step 3: Data Preprocessing
# ============================================================

# Keep only required columns
required_columns = [
    'price',
    'bedrooms',
    'baths',
    'Area Size',
    'Area Type',
    'city',
    'property_type',
    'purpose'
]

df_clean = df[required_columns].copy()

# Remove rows with missing values
df_clean = df_clean.dropna()

print("\nAfter removing missing values:")
print(f"Rows remaining: {len(df_clean):,}")


# ------------------------------------------------------------
# 3.1 Filter properties for sale
# ------------------------------------------------------------

df_clean = df_clean[
    df_clean['purpose'].astype(str).str.strip().str.lower() == 'for sale'
].copy()

print(f"\nProperties for sale: {len(df_clean):,}")


# ------------------------------------------------------------
# 3.2 Convert all areas to Marla
# ------------------------------------------------------------

df_clean['Area_Marla'] = np.where(
    df_clean['Area Type'].astype(str).str.strip().str.lower() == 'kanal',
    df_clean['Area Size'] * 20,
    df_clean['Area Size']
)


# ------------------------------------------------------------
# 3.3 Remove unrealistic values / outliers
# ------------------------------------------------------------

df_clean = df_clean[
    (df_clean['price'] >= 500000) &
    (df_clean['price'] <= 200000000) &
    (df_clean['Area_Marla'] > 0.5) &
    (df_clean['Area_Marla'] <= 100) &
    (df_clean['bedrooms'] >= 1) &
    (df_clean['bedrooms'] <= 10) &
    (df_clean['baths'] >= 1) &
    (df_clean['baths'] <= 10)
].copy()

print(f"\nRows after outlier filtering: {len(df_clean):,}")


# ============================================================
# Step 4: Feature Selection
# ============================================================

features = [
    'Area_Marla',
    'bedrooms',
    'baths',
    'city',
    'property_type'
]

X = df_clean[features].copy()
y = df_clean['price'].copy()


# ------------------------------------------------------------
# Encode categorical variables
# ------------------------------------------------------------

X = pd.get_dummies(
    X,
    columns=['city', 'property_type'],
    drop_first=True,
    dtype=int
)

print("\n--- Final Features ---")
print(X.columns.tolist())


# ============================================================
# Step 5: Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"Training Set: {len(X_train):,}")
print(f"Testing Set:  {len(X_test):,}")


# ============================================================
# Step 6: Linear Regression
# ============================================================

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr_train = lr_model.predict(X_train)
y_pred_lr_test = lr_model.predict(X_test)

# Metrics
r2_lr_train = r2_score(y_train, y_pred_lr_train)
r2_lr_test = r2_score(y_test, y_pred_lr_test)

rmse_lr = np.sqrt(
    mean_squared_error(y_test, y_pred_lr_test)
)

mae_lr = mean_absolute_error(
    y_test,
    y_pred_lr_test
)

print("\n" + "=" * 60)
print("LINEAR REGRESSION RESULTS")
print("=" * 60)

print(f"R² Score (Train): {r2_lr_train:.4f}")
print(f"R² Score (Test):  {r2_lr_test:.4f}")
print(f"RMSE:             PKR {rmse_lr:,.0f}")
print(f"MAE:              PKR {mae_lr:,.0f}")


# ============================================================
# Step 7: Random Forest Regression
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf_train = rf_model.predict(X_train)
y_pred_rf_test = rf_model.predict(X_test)

# Metrics
r2_rf_train = r2_score(y_train, y_pred_rf_train)
r2_rf_test = r2_score(y_test, y_pred_rf_test)

rmse_rf = np.sqrt(
    mean_squared_error(y_test, y_pred_rf_test)
)

mae_rf = mean_absolute_error(
    y_test,
    y_pred_rf_test
)

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(f"R² Score (Train): {r2_rf_train:.4f}")
print(f"R² Score (Test):  {r2_rf_test:.4f}")
print(f"RMSE:             PKR {rmse_rf:,.0f}")
print(f"MAE:              PKR {mae_rf:,.0f}")


# ============================================================
# Step 8: Model Comparison
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Random Forest'
    ],
    'Test R²': [
        r2_lr_test,
        r2_rf_test
    ],
    'RMSE': [
        rmse_lr,
        rmse_rf
    ],
    'MAE': [
        mae_lr,
        mae_rf
    ]
})

print(comparison.to_string(index=False))


# Select best model based on test R²
if r2_rf_test > r2_lr_test:
    best_model = rf_model
    best_model_name = "Random Forest"
else:
    best_model = lr_model
    best_model_name = "Linear Regression"

print(f"\nBest Model: {best_model_name}")


# ============================================================
# Step 9: Feature Importance
# ============================================================

# Feature importance is available for Random Forest
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(
    'Importance',
    ascending=False
)

print("\n" + "=" * 60)
print("TOP FEATURE IMPORTANCES")
print("=" * 60)

print(importance_df.head(10).to_string(index=False))


# ============================================================
# Step 10: Visualization 1
# Average House Price by City
# ============================================================

plt.figure(figsize=(10, 6))

avg_price_city = (
    df_clean.groupby('city')['price']
    .mean()
    .sort_values(ascending=False)
    / 1e6
)

avg_price_city.plot(kind='bar')

plt.title('Average House Price by City')
plt.xlabel('City')
plt.ylabel('Average Price (Million PKR)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ============================================================
# Step 11: Visualization 2
# House Price Distribution
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df_clean['price'] / 1e6,
    bins=40,
    edgecolor='black',
    alpha=0.7
)

plt.title('House Price Distribution')
plt.xlabel('Price (Million PKR)')
plt.ylabel('Number of Houses')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ============================================================
# Step 12: Visualization 3
# House Price vs Area
# ============================================================

plt.figure(figsize=(10, 6))

sample_plot = df_clean.sample(
    min(2000, len(df_clean)),
    random_state=42
)

plt.scatter(
    sample_plot['Area_Marla'],
    sample_plot['price'] / 1e6,
    alpha=0.4
)

plt.title('House Price vs Area Size')
plt.xlabel('Area Size (Marla)')
plt.ylabel('Price (Million PKR)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ============================================================
# Step 13: Visualization 4
# Feature Importance
# ============================================================

top_features = importance_df.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features['Feature'],
    top_features['Importance']
)

plt.title('Top 10 Feature Importances - Random Forest')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ============================================================
# Step 14: Visualization 5
# Actual vs Predicted Prices
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test / 1e6,
    y_pred_rf_test / 1e6,
    alpha=0.5
)

min_price = min(
    y_test.min(),
    y_pred_rf_test.min()
) / 1e6

max_price = max(
    y_test.max(),
    y_pred_rf_test.max()
) / 1e6

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    'r--',
    linewidth=2
)

plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Price (Million PKR)')
plt.ylabel('Predicted Price (Million PKR)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ============================================================
# Step 15: Predict Price of a New House
# ============================================================

sample_house = pd.DataFrame([{
    'Area_Marla': 10.0,
    'bedrooms': 4,
    'baths': 4,
    'city_Islamabad': 1,
    'property_type_House': 1
}])

# Make sure sample has exactly the same columns as training data
sample_house = sample_house.reindex(
    columns=X.columns,
    fill_value=0
)

predicted_price = best_model.predict(sample_house)[0]

print("\n" + "=" * 60)
print("NEW HOUSE PRICE PREDICTION")
print("=" * 60)

print("House Details:")
print("Area       : 10 Marla")
print("Bedrooms   : 4")
print("Bathrooms  : 4")
print("City       : Islamabad")
print("Type       : House")

print(
    f"\nEstimated Price: PKR {predicted_price:,.0f}"
)

print("\nProject completed successfully!")