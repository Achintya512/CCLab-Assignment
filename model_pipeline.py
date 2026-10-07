import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_features(df, target_col, lags=5):
    df_feat = df.copy()
    for i in range(1, lags + 1):
        df_feat[f'lag_{i}'] = df_feat[target_col].shift(i)
    
    # Adding time-based features
    # Ensure timestamp is datetime
    df_feat['timestamp'] = pd.to_datetime(df_feat['timestamp'])
    df_feat['hour'] = df_feat['timestamp'].dt.hour
    df_feat['dayofweek'] = df_feat['timestamp'].dt.dayofweek
    
    df_feat.dropna(inplace=True)
    return df_feat

def train_and_evaluate(df, target_col='cpu_usage'):
    df_feat = create_features(df, target_col)
    
    features = [col for col in df_feat.columns if col not in ['timestamp', 'cpu_usage']]
    X = df_feat[features]
    y = df_feat[target_col]

    # Time series split (80% train, 20% test)
    split_idx = int(len(df_feat) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    timestamps_test = df_feat['timestamp'].iloc[split_idx:]

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=42)
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results[name] = {'MAE': mae, 'RMSE': rmse}
        predictions[name] = preds

    return results, predictions, y_test, timestamps_test
