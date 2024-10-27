import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import talib  # Import TA-Lib instead of ta
from logger.logger import logger
from typing import Tuple
import joblib  # Import joblib for model persistence
import os

class IndicatorCalculator:
    """Calculates both traditional and price action/SMC indicators for Forex trading signals."""

    def calculate_traditional_indicators(self, df):
        # Add technical indicators to the DataFrame using TA-Lib
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)  # Forex may need smaller window sizes
        df['sma'] = talib.SMA(df['close'], timeperiod=50)  # Longer SMA for Forex
        df['ema'] = talib.EMA(df['close'], timeperiod=21)  # EMA for faster reaction
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)  # ADX for trend strength
        macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)  # ATR for volatility
        df.dropna(inplace=True)  # Drop rows with NaN values (created due to indicators)
        return df

    @staticmethod
    def detect_swing_points(data: pd.Series, lookback: int = 3, high: bool = True) -> pd.Series:
        """Detect swing highs or lows based on price action."""
        swing_points = np.zeros(len(data))
        for i in range(lookback, len(data) - lookback):
            if high:
                condition = (data.iloc[i] > data.iloc[i - lookback:i].max() and
                             data.iloc[i] > data.iloc[i + 1:i + lookback + 1].max())
            else:
                condition = (data.iloc[i] < data.iloc[i - lookback:i].min() and
                             data.iloc[i] < data.iloc[i + 1:i + lookback + 1].min())
            swing_points[i] = 1 if condition else 0
        return pd.Series(swing_points, index=data.index)

    @staticmethod
    def detect_order_blocks(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
        """Detect potential order blocks based on price imbalances."""
        order_blocks = np.zeros(len(df))
        for i in range(lookback, len(df) - lookback):
            if (df['low'].iloc[i] < df['low'].iloc[i - lookback:i].min()) and (df['close'].iloc[i] > df['open'].iloc[i]):
                order_blocks[i] = 1  # Bullish order block
            elif (df['high'].iloc[i] > df['high'].iloc[i - lookback:i].max()) and (df['close'].iloc[i] < df['open'].iloc[i]):
                order_blocks[i] = -1  # Bearish order block
        return pd.Series(order_blocks, index=df.index)
    
    @staticmethod
    def detect_break_of_structure(df: pd.DataFrame) -> pd.Series:
        """Detect Break of Structure (BOS) for trend continuation or reversal."""
        bos = np.zeros(len(df))
        
        # Ensure both series have the same index after shifting
        close_shifted = df['close'].shift(1)
        high_shifted = df['high'].shift(1)
        low_shifted = df['low'].shift(1)
        
        # Break of structure conditions
        bos = np.where(df['close'] > high_shifted, 1,
                    np.where(df['close'] < low_shifted, -1, 0))
        
        return pd.Series(bos, index=df.index)

    @staticmethod
    def detect_liquidity_grabs(df: pd.DataFrame, threshold: float = 0.01) -> pd.Series:
        """Detect liquidity grabs based on price spikes or stop hunts."""
        range_high = df['high'] - df['low']
        wick_high = df['high'] - df[['close', 'open']].max(axis=1)
        wick_low = df[['close', 'open']].min(axis=1) - df['low']
        
        liquidity_grabs = np.zeros(len(df))
        liquidity_grabs[1:] = np.where(wick_high.iloc[1:] > threshold * range_high.iloc[1:], -1,
                                       np.where(wick_low.iloc[1:] > threshold * range_high.iloc[1:], 1, 0))
        return pd.Series(liquidity_grabs, index=df.index)

    @staticmethod
    def detect_support_resistance(close_prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Detect support and resistance levels."""
        support = close_prices.rolling(window).min()
        resistance = close_prices.rolling(window).max()
        return support, resistance

class MLIndicatorCalculator(IndicatorCalculator):
    def __init__(self, symbol_name):
        # Initialize file paths for the model and scaler
        self.model_path = f"model/{symbol_name}_model.pkl"
        self.scaler_path = f"model/{symbol_name}_scaler.pkl"
        self.symbol_name = symbol_name
        
        # Initialize XGBoost Classifier
        self.model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.scaler = StandardScaler()
        self.is_model_trained = False

        # Attempt to load existing model and scaler
        self.load_model()

    def save_model(self):
        """Saves the trained model, scaler, and symbol name to disk."""
        try:
            joblib.dump({'model': self.model, 'scaler': self.scaler}, self.model_path)
            logger.info(f"Model, scaler, and symbol '{self.symbol_name}' saved to {self.model_path}.")
        except Exception as e:
            logger.error(f"Failed to save model, scaler, and symbol: {e}")

    def load_model(self):
        """Loads the trained model, scaler, and symbol name from disk if they exist."""
        if os.path.exists(self.model_path):
            try:
                saved_objects = joblib.load(self.model_path)
                self.model = saved_objects['model']
                self.scaler = saved_objects['scaler']
                self.is_model_trained = True  # Assume model is trained if loaded successfully
                logger.info(f"Loaded model, scaler, and symbol '{self.symbol_name}' from {self.model_path}.")
            except Exception as e:
                logger.error(f"Failed to load model, scaler, and symbol: {e}")
                self.is_model_trained = False
        else:
            logger.info("No existing model found. A new model will be trained.")

    def prepare_combined_features(self, df):
        if len(df) < 200:
            logger.error("Not enough data to calculate features.")
            raise ValueError("Not enough data to calculate features.")

        # Calculate traditional indicators
        df = self.calculate_traditional_indicators(df)
        
        # Detect support, resistance
        support, resistance = self.detect_support_resistance(df['close'])
        
        # Combine all features into a DataFrame
        features = pd.DataFrame({
            'swing_high': self.detect_swing_points(df['high'], high=True),
            'swing_low': self.detect_swing_points(df['low'], high=False),
            'order_blocks': self.detect_order_blocks(df),
            'bos': self.detect_break_of_structure(df),
            'liquidity_grabs': self.detect_liquidity_grabs(df),
            'support': support,
            'resistance': resistance,
            'rsi': df['rsi'],
            'sma': df['sma'],
            'ema': df['ema'],
            'adx': df['adx'],
            'macd': df['macd'],
            'atr': df['atr'],  # Include ATR for volatility measurement
        })

        # features.to_csv("features_forex.csv", index=False)
        # logger.info(f"Features exported to features_forex.csv")

        # Drop NaN values
        features.dropna(inplace=True)

        logger.info(f"Symbol: {self.symbol_name}, Features data length after dropping NaN: {len(features)}")

        if len(features) == 0:
            logger.error("No valid features after dropping NaNs.")
            raise ValueError("No valid features after dropping NaNs.")
    
        # Align the indices of df['close'] and features to create target labels
        target_series = df['close'].iloc[len(df) - len(features):]  # Get the last `len(features)` entries

        # Create target labels (price direction)
        features['target'] = (target_series.shift(-1) > target_series).astype(int)

        return features.drop('target', axis=1), features['target']

    def train_model(self, df):
        if self.is_model_trained:
            logger.info("Model is already trained. Loading existing model.")
            return

        X, y = self.prepare_combined_features(df)
        X_scaled = self.scaler.fit_transform(X)

        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
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

        if accuracy >= 50:
            self.is_model_trained = True
            logger.info("Model accuracy is above 50%, trades are allowed.")
            self.save_model()  # Save the trained model and scaler
        else:
            self.is_model_trained = False
            logger.info("Model accuracy is below 50%, no trades will be allowed.")

    def predict_signal(self, df):
        if not self.is_model_trained:
            logger.info("Model not trained or accuracy below 50%, no trades allowed.")
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

            # Calculate ATR-based dynamic thresholds
            current_atr = df['atr'].iloc[-1]  # Get the latest ATR value
            baseline_atr = df['atr'].mean()   # Calculate the average ATR for normalization

            # Normalize the ATR value to calculate dynamic thresholds
            atr_ratio = current_atr / baseline_atr
            buy_threshold = max(0.5, min(0.8 * atr_ratio, 1.0))  # Set a range for the buy threshold
            sell_threshold = max(0.5, min(0.8 * atr_ratio, 1.0))  # Set a range for the sell threshold

            # Determine the signal based on dynamic thresholds
            if buy_probability >= buy_threshold:
                signal = "buy"
            elif sell_probability >= sell_threshold:
                signal = "sell"
            else:
                signal = "no signal"
            
            # Log the signal and probabilities
            logger.info(f"Symbol: {self.symbol_name}, Predicted signal: {signal}")
            logger.info(f"Symbol: {self.symbol_name}, Buy probability: {buy_probability:.4f}, Sell probability: {sell_probability:.4f}")
            logger.info(f"Symbol: {self.symbol_name}, Dynamic Buy threshold: {buy_threshold:.2f}, Dynamic Sell threshold: {sell_threshold:.2f}")

            # Log additional feature importance information
            feature_importance = self.model.feature_importances_
            feature_names = X.columns
            for name, importance in zip(feature_names, feature_importance):
                logger.info(f"Symbol: {self.symbol_name}, Feature {name}: importance = {importance:.4f}, value = {X.iloc[-1][name]}")

            return signal

        except Exception as e:
            logger.error(f"Error in generating signal: {e}")
            return "no signal"
