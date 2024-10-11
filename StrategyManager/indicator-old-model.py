import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import ta
from logger.logger import logger

class IndicatorCalculator:
    """Calculates both traditional and price action/SMC indicators for Forex trading signals.""" 

    def calculate_traditional_indicators(self, df):
        # Add technical indicators to the DataFrame using the ta library
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()  # Forex may need smaller window sizes
        df['sma'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()  # Longer SMA for Forex
        df['ema'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()  # EMA for faster reaction
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()  # ADX for trend strength
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()  # ATR for volatility
        df.dropna(inplace=True)  # Drop rows with NaN values (created due to indicators)
        return df

    def detect_swing_highs(self, high, lookback=3):
        # Detect swing highs: adapted for Forex, lower lookback to capture quick market swings
        swing_highs = np.zeros(len(high))

        for i in range(lookback, len(high) - lookback):
            if (high.iloc[i] > high.iloc[i - lookback:i].max() and
                high.iloc[i] > high.iloc[i + 1:i + lookback + 1].max()):
                swing_highs[i] = 1

        return pd.Series(swing_highs, index=high.index)

    def detect_swing_lows(self, low, lookback=3):
        # Detect swing lows: adapted for Forex, lower lookback to capture quick market swings
        swing_lows = np.zeros(len(low))

        for i in range(lookback, len(low) - lookback):
            if (low.iloc[i] < low.iloc[i - lookback:i].min() and
                low.iloc[i] < low.iloc[i + 1:i + lookback + 1].min()):
                swing_lows[i] = 1

        return pd.Series(swing_lows, index=low.index)

    def detect_support_resistance(self, close_prices, window=20):
        # For Forex, use a slightly larger window to detect support and resistance zones
        support = pd.Series(close_prices).rolling(window).min()
        resistance = pd.Series(close_prices).rolling(window).max()
        return support, resistance

class MLIndicatorCalculator(IndicatorCalculator):
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.scaler = StandardScaler()
        self.is_model_trained = False

    def prepare_combined_features(self, df):
        if len(df) < 200:
            logger.error("Not enough data to calculate features.")
            raise ValueError("Not enough data to calculate features.")

        # Calculate traditional indicators
        df = self.calculate_traditional_indicators(df)
        
        # Detect price action/SMC indicators
        swing_highs = self.detect_swing_highs(df['high'])
        swing_lows = self.detect_swing_lows(df['low'])
        support, resistance = self.detect_support_resistance(df['close'])
        
        # Combine all features into a DataFrame
        features = pd.DataFrame({
            'swing_high': swing_highs,
            'swing_low': swing_lows,
            'support': support,
            'resistance': resistance,
            'rsi': df['rsi'],
            'sma': df['sma'],
            'ema': df['ema'],
            'adx': df['adx'],
            'macd': df['macd'],
            'atr': df['atr'],  # Include ATR for volatility measurement
        })

        csv_filename = "features_forex.csv"
        features.to_csv(csv_filename, index=False)
        logger.info(f"Features exported to {csv_filename}")

        # Drop NaN values
        features.dropna(inplace=True)

        logger.info(f"Features data length after dropping NaN: {len(features)}")

        if len(features) == 0:
            logger.error("No valid features after dropping NaNs.")
            raise ValueError("No valid features after dropping NaNs.")
    
        # Align the indices of df['close'] and features to create target labels
        target_series = df['close'].iloc[len(df) - len(features):]  # Get the last `len(features)` entries

        # Create target labels (price direction)
        features['target'] = (target_series.shift(-1) > target_series).astype(int)

        return features.drop('target', axis=1), features['target']

    def train_model(self, df):
        X, y = self.prepare_combined_features(df)
        X_scaled = self.scaler.fit_transform(X)

        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=2)
        grid_search.fit(X_scaled, y)

        self.model = grid_search.best_estimator_
        logger.info(f"Best Parameters: {grid_search.best_params_}")

        # Evaluate on a test split to confirm performance
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        logger.info(f"Model trained with accuracy: {accuracy:.2f}%")

        if accuracy >= 60:
            self.is_model_trained = True
            logger.info("Model accuracy is above 60%, trades are allowed.")
        else:
            self.is_model_trained = False
            logger.info("Model accuracy is below 60%, no trades will be allowed.")

    def predict_signal(self, df):
        if not self.is_model_trained:
            logger.info("Model not trained or accuracy below 60%, no trades allowed.")
            return "trade not allowed"

        try:
            X, _ = self.prepare_combined_features(df)
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)

            if len(probabilities) == 0:
                logger.info("No predictions made.")
                return "no signal"

            buy_probability = probabilities[-1, 1]
            sell_probability = probabilities[-1, 0]

            buy_threshold = 0.80
            sell_threshold = 0.80

            if buy_probability >= buy_threshold:
                signal = "buy"
            elif sell_probability >= sell_threshold:
                signal = "sell"
            else:
                signal = "no trade"

            logger.info(f"Predicted signal: {signal}")
            logger.info(f"Buy probability: {buy_probability:.4f}, Sell probability: {sell_probability:.4f}")
            logger.info(f"Buy threshold: {buy_threshold:.2f}, Sell threshold: {sell_threshold:.2f}")

            # Log additional feature information
            feature_importance = self.model.feature_importances_
            feature_names = X.columns
            for name, importance in zip(feature_names, feature_importance):
                logger.info(f"Feature {name}: importance = {importance:.4f}, value = {X.iloc[-1][name]}")

            return signal

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "no signal"
