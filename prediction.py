import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

## 1. Synthetic Global Petrol Price Data Generation
def generate_global_petrol_data(years=20):
    """Generate synthetic global petrol price data with regional trends"""
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=365*years)
    dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
    
    # Base oil price simulation (Brent crude)
    base_price = np.cumsum(np.random.normal(0, 0.5, len(dates))) + 50
    base_price = np.abs(base_price)
    
    # Regional multipliers
    regions = {
        'North America': {'base': 1.1, 'volatility': 0.15},
        'Europe': {'base': 1.3, 'volatility': 0.12},
        'Asia': {'base': 0.9, 'volatility': 0.18},
        'Middle East': {'base': 0.7, 'volatility': 0.2},
        'Africa': {'base': 1.0, 'volatility': 0.25},
        'South America': {'base': 1.2, 'volatility': 0.22},
        'Oceania': {'base': 1.4, 'volatility': 0.1}
    }
    
    data = []
    for date, crude_price in zip(dates, base_price):
        for region, params in regions.items():
            # Regional price formula
            price = (crude_price * params['base'] * 
                    (1 + params['volatility'] * np.random.randn()))
            
            # Add seasonal effects
            seasonal = 0.1 * np.sin(2 * np.pi * date.dayofyear/365)
            price *= (1 + seasonal)
            
            data.append({
                'date': date,
                'region': region,
                'crude_oil_price': crude_price,
                'petrol_price': max(0.5, price)  # Ensure positive price
            })
    
    return pd.DataFrame(data)

# Generate and preview data
global_df = generate_global_petrol_data()
print(global_df.head())

## 2. Data Preprocessing
def preprocess_data(df):
    """Prepare data for time series forecasting"""
    df = df.copy()
    
    # Extract time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Lag features
    for lag in [1, 7, 30, 90]:
        df[f'price_lag_{lag}'] = df.groupby('region')['petrol_price'].shift(lag)
    
    # Rolling statistics
    df['price_rolling_avg_7'] = df.groupby('region')['petrol_price'].transform(
        lambda x: x.rolling(7).mean())
    df['price_rolling_std_7'] = df.groupby('region')['petrol_price'].transform(
        lambda x: x.rolling(7).std())
    
    # Oil price relationships
    df['price_oil_ratio'] = df['petrol_price'] / df['crude_oil_price']
    df['oil_price_change'] = df.groupby('region')['crude_oil_price'].pct_change()
    
    # Drop missing values
    df.dropna(inplace=True)
    
    return df

processed_df = preprocess_data(global_df)

## 3. Time Series Forecasting Models

### 3.1 ARIMA Model
def train_arima(df, region):
    """Train ARIMA model for a specific region"""
    region_df = df[df['region'] == region].set_index('date')
    train_size = int(len(region_df) * 0.8)
    train, test = region_df.iloc[:train_size], region_df.iloc[train_size:]
    
    model = ARIMA(train['petrol_price'], order=(1,1,1))
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.forecast(steps=len(test))
    
    # Evaluate
    mae = mean_absolute_error(test['petrol_price'], forecast)
    print(f"{region} ARIMA MAE: {mae:.2f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['petrol_price'], label='Train')
    plt.plot(test.index, test['petrol_price'], label='Actual')
    plt.plot(test.index, forecast, label='Forecast')
    plt.title(f'{region} Petrol Price Forecast (ARIMA)')
    plt.legend()
    plt.show()
    
    return model_fit

# Train ARIMA for each region
arima_models = {}
for region in processed_df['region'].unique():
    arima_models[region] = train_arima(processed_df, region)

### 3.2 Prophet Model
def train_prophet(df, region):
    """Train Facebook Prophet model for a region"""
    region_df = df[df['region'] == region][['date', 'petrol_price']].rename(
        columns={'date': 'ds', 'petrol_price': 'y'})
    
    train_size = int(len(region_df) * 0.8)
    train, test = region_df.iloc[:train_size], region_df.iloc[train_size:]
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.fit(train)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    
    # Evaluate
    preds = forecast.iloc[-len(test):]['yhat']
    mae = mean_absolute_error(test['y'], preds)
    print(f"{region} Prophet MAE: {mae:.2f}")
    
    # Plot
    fig = model.plot(forecast)
    plt.title(f'{region} Petrol Price Forecast (Prophet)')
    plt.show()
    
    return model

# Train Prophet for each region
prophet_models = {}
for region in processed_df['region'].unique()[:3]:  # Just first 3 for demo
    prophet_models[region] = train_prophet(processed_df, region)

### 3.3 XGBoost Model
def train_xgboost(df, region):
    """Train XGBoost model with temporal features"""
    region_df = df[df['region'] == region]
    
    # Features and target
    features = ['year', 'month', 'day', 'day_of_week', 'day_of_year',
               'crude_oil_price', 'price_lag_1', 'price_lag_7', 
               'price_lag_30', 'price_rolling_avg_7', 'price_rolling_std_7',
               'price_oil_ratio', 'oil_price_change']
    target = 'petrol_price'
    
    train_size = int(len(region_df) * 0.8)
    train, test = region_df.iloc[:train_size], region_df.iloc[train_size:]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])
    y_train, y_test = train[target], test[target]
    
    # Train model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"{region} XGBoost MAE: {mae:.2f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(features, model.feature_importances_)
    plt.title(f'{region} Feature Importance (XGBoost)')
    plt.show()
    
    return model, scaler

# Train XGBoost for each region
xgboost_models = {}
for region in processed_df['region'].unique():
    xgboost_models[region] = train_xgboost(processed_df, region)

## 4. Forecasting Future Prices
def forecast_prices(region, days_to_forecast=30, model_type='xgboost'):
    """Forecast future petrol prices for a region"""
    region_df = processed_df[processed_df['region'] == region].copy()
    last_date = region_df['date'].max()
    
    if model_type == 'xgboost':
        model, scaler = xgboost_models[region]
        
        # Prepare future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_to_forecast
        )
        
        # Create future features
        future_df = pd.DataFrame({'date': future_dates})
        future_df['year'] = future_df['date'].dt.year
        future_df['month'] = future_df['date'].dt.month
        future_df['day'] = future_df['date'].dt.day
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['day_of_year'] = future_df['date'].dt.dayofyear
        
        # Use last known values for lags (could be improved with recursive forecasting)
        last_row = region_df.iloc[-1]
        for col in ['crude_oil_price', 'price_lag_1', 'price_lag_7', 'price_lag_30',
                   'price_rolling_avg_7', 'price_rolling_std_7', 'price_oil_ratio',
                   'oil_price_change']:
            future_df[col] = last_row[col]
        
        # Scale and predict
        X_future = scaler.transform(future_df.drop('date', axis=1))
        future_prices = model.predict(X_future)
        
        # Create result dataframe
        result = pd.DataFrame({
            'date': future_dates,
            'region': region,
            'forecasted_price': future_prices
        })
        
    elif model_type == 'prophet':
        model = prophet_models.get(region)
        if not model:
            raise ValueError(f"No Prophet model for {region}")
            
        future = model.make_future_dataframe(periods=days_to_forecast)
        forecast = model.predict(future)
        result = forecast[['ds', 'yhat']].rename(
            columns={'ds': 'date', 'yhat': 'forecasted_price'})
        result['region'] = region
    
    elif model_type == 'arima':
        model = arima_models[region]
        forecast = model.forecast(steps=days_to_forecast)
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_to_forecast
        )
        result = pd.DataFrame({
            'date': future_dates,
            'region': region,
            'forecasted_price': forecast
        })
    
    else:
        raise ValueError("Invalid model_type. Use 'xgboost', 'prophet', or 'arima'")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(region_df['date'], region_df['petrol_price'], label='Historical')
    plt.plot(result['date'], result['forecasted_price'], label='Forecast', color='red')
    plt.title(f'{region} Petrol Price {days_to_forecast}-Day Forecast ({model_type.title()})')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()
    
    return result

# Example forecast
forecast_results = forecast_prices('Europe', days_to_forecast=90, model_type='xgboost')
print(forecast_results.head())

## 5. Comparative Analysis
def compare_models(df, region):
    """Compare performance of different models for a region"""
    results = []
    
    # ARIMA
    arima_model = arima_models[region]
    test_size = int(len(df[df['region'] == region]) * 0.2)
    test = df[df['region'] == region].iloc[-test_size:]
    arima_preds = arima_model.forecast(steps=len(test))
    results.append({
        'model': 'ARIMA',
        'mae': mean_absolute_error(test['petrol_price'], arima_preds),
        'rmse': np.sqrt(mean_squared_error(test['petrol_price'], arima_preds))
    })
    
    # Prophet (if available)
    if region in prophet_models:
        prophet_model = prophet_models[region]
        prophet_preds = prophet_model.predict(
            prophet_model.make_future_dataframe(periods=len(test)))['yhat'].iloc[-len(test):]
        results.append({
            'model': 'Prophet',
            'mae': mean_absolute_error(test['petrol_price'], prophet_preds),
            'rmse': np.sqrt(mean_squared_error(test['petrol_price'], prophet_preds))
        })
    
    # XGBoost
    xgb_model, scaler = xgboost_models[region]
    X_test = scaler.transform(test[[
        'year', 'month', 'day', 'day_of_week', 'day_of_year',
        'crude_oil_price', 'price_lag_1', 'price_lag_7', 'price_lag_30',
        'price_rolling_avg_7', 'price_rolling_std_7', 'price_oil_ratio',
        'oil_price_change'
    ]])
    xgb_preds = xgb_model.predict(X_test)
    results.append({
        'model': 'XGBoost',
        'mae': mean_absolute_error(test['petrol_price'], xgb_preds),
        'rmse': np.sqrt(mean_squared_error(test['petrol_price'], xgb_preds))
    })
    
    # Create comparison dataframe
    comparison = pd.DataFrame(results).set_index('model')
    
    # Plot
    comparison[['mae', 'rmse']].plot(kind='bar', figsize=(10, 6))
    plt.title(f'Model Comparison for {region}')
    plt.ylabel('Error')
    plt.xticks(rotation=0)
    plt.show()
    
    return comparison

# Example comparison
model_comparison = compare_models(processed_df, 'North America')
print(model_comparison)