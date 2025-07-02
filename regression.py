# model_comparison.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from utils import load_data

# 0. Functions to train and evaluate a models

def train_and_evaluate_ridge_model(X_train, X_test, y_train, y_test) : # For Ridge Regression
    model = Ridge()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse,r2

def train_and_evaluate_svr_model(X_train, X_test, y_train, y_test) : # For SVR Regression
    model = SVR()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse,r2

def train_and_evaluate_random_forest_model(X_train, X_test, y_train, y_test) : # For Random Forest Regression
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse,r2


# 1. Load data
df = load_data()

# 2. Basic preprocessing
df = df.dropna()

# 3. Define features and target
X = df.drop(columns=['MEDV'])  
y = df['MEDV']

# 4. Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42) # Keeping 35% data for testing

# 5. Scale features for SVR
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 6. Initialize models
models = {
    'Ridge Regression': lambda : train_and_evaluate_ridge_model(x_train_scaled, x_test_scaled, y_train, y_test),
    'SVR': lambda : train_and_evaluate_svr_model(x_train_scaled, x_test_scaled, y_train, y_test),
    'Random Forest': lambda : train_and_evaluate_random_forest_model(x_train, x_test, y_train, y_test)

}


print(f"{'Model':<20} {'MSE':<15} {'R² Score' :<10}")
print("-" * 50)

# 7. Train and evaluate models
for name, func in models.items():
    mse, r2 = func()

    print(f"{name:<20} {mse:<15.2f} {r2:.3f}")

print("\n\nScript completed successfully.\n")