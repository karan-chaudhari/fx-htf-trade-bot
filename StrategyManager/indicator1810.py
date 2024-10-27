import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import ta
from logger.logger import logger
import joblib
import os
from typing import Tuple, Dict, Any

# class ForexSMCStrategy:
#     @staticmethod
#     def detect_swing_points(data: pd.Series, lookback: int = 3, high: bool = True) -> pd.Series:
#         """Detect swing highs or lows based on price action."""
#         swing_points = np.zeros(len(data))
#         for i in range(lookback, len(data) - lookback):
#             if high:
#                 condition = (data.iloc[i] > data.iloc[i - lookback:i].max() and
#                              data.iloc[i] > data.iloc[i + 1:i + lookback + 1].max())
#             else:
#                 condition = (data.iloc[i] < data.iloc[i - lookback:i].min() and
#                              data.iloc[i] < data.iloc[i + 1:i + lookback + 1].min())
#             swing_points[i] = 1 if condition else 0
#         return pd.Series(swing_points, index=data.index)

#     @staticmethod
#     def detect_order_blocks(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
#         """Detect potential order blocks based on price imbalances."""
#         order_blocks = np.zeros(len(df))
#         for i in range(lookback, len(df) - lookback):
#             if (df['low'].iloc[i] < df['low'].iloc[i - lookback:i].min()) and (df['close'].iloc[i] > df['open'].iloc[i]):
#                 order_blocks[i] = 1  # Bullish order block
#             elif (df['high'].iloc[i] > df['high'].iloc[i - lookback:i].max()) and (df['close'].iloc[i] < df['open'].iloc[i]):
#                 order_blocks[i] = -1  # Bearish order block
#         return pd.Series(order_blocks, index=df.index)
    
#     @staticmethod
#     def detect_break_of_structure(df: pd.DataFrame) -> pd.Series:
#         """Detect Break of Structure (BOS) for trend continuation or reversal."""
#         bos = np.zeros(len(df))
        
#         # Ensure both series have the same index after shifting
#         close_shifted = df['close'].shift(1)
#         high_shifted = df['high'].shift(1)
#         low_shifted = df['low'].shift(1)
        
#         # Break of structure conditions
#         bos = np.where(df['close'] > high_shifted, 1,
#                     np.where(df['close'] < low_shifted, -1, 0))
        
#         return pd.Series(bos, index=df.index)

#     @staticmethod
#     def detect_liquidity_grabs(df: pd.DataFrame, threshold: float = 0.01) -> pd.Series:
#         """Detect liquidity grabs based on price spikes or stop hunts."""
#         range_high = df['high'] - df['low']
#         wick_high = df['high'] - df[['close', 'open']].max(axis=1)
#         wick_low = df[['close', 'open']].min(axis=1) - df['low']
        
#         liquidity_grabs = np.zeros(len(df))
#         liquidity_grabs[1:] = np.where(wick_high.iloc[1:] > threshold * range_high.iloc[1:], -1,
#                                        np.where(wick_low.iloc[1:] > threshold * range_high.iloc[1:], 1, 0))
#         return pd.Series(liquidity_grabs, index=df.index)

class IndicatorCalculator:
    @staticmethod
    def calculate_traditional_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate traditional technical indicators."""
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['sma'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
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
    def __init__(self, symbol_name: str):
        self.symbol_name = symbol_name
        self.model_path = f"model/{symbol_name}_model.pkl"
        self.model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.scaler = StandardScaler()
        self.is_model_trained = False
        self.load_model()

    def save_model(self) -> None:
        """Save the trained model and scaler to disk."""
        try:
            joblib.dump({'model': self.model, 'scaler': self.scaler}, self.model_path)
            logger.info(f"Model and scaler saved for symbol '{self.symbol_name}' at {self.model_path}.")
        except Exception as e:
            logger.error(f"Failed to save model and scaler: {e}")

    def load_model(self) -> None:
        """Load the trained model and scaler from disk if they exist."""
        if os.path.exists(self.model_path):
            try:
                saved_objects = joblib.load(self.model_path)
                self.model = saved_objects['model']
                self.scaler = saved_objects['scaler']
                self.is_model_trained = True
                logger.info(f"Loaded model and scaler for symbol '{self.symbol_name}' from {self.model_path}.")
            except Exception as e:
                logger.error(f"Failed to load model and scaler: {e}")
                self.is_model_trained = False
        else:
            logger.info(f"No existing model found for symbol '{self.symbol_name}'. A new model will be trained.")

    def prepare_combined_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare combined features for model training or prediction."""
        if len(df) < 200:
            raise ValueError("Not enough data to calculate features.")
        
        # Ensure the required price columns are present
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            
        support, resistance = self.detect_support_resistance(df['close'])

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
        })

        features.to_csv("features_forex.csv", index=False)
        logger.info(f"Features exported to features_forex.csv")

        features.dropna(inplace=True)
        logger.info(f"Symbol: {self.symbol_name}, Features data length after dropping NaN: {len(features)}")

        if len(features) == 0:
            raise ValueError("No valid features after dropping NaNs.")
    
        target_series = df['close'].iloc[len(df) - len(features):]
        features['target'] = (target_series.shift(-1) > target_series).astype(int)

        return features.drop('target', axis=1), features['target']

    def train_model(self, df: pd.DataFrame) -> None:
        """Train the XGBoost model using GridSearchCV for hyperparameter tuning."""
        if self.is_model_trained:
            logger.info("Model is already trained. Loading existing model.")
            return

        X, y = self.prepare_combined_features(df)
        X_scaled = self.scaler.fit_transform(X)

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

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        logger.info(f"Model trained with accuracy: {accuracy:.2f}%")

        if accuracy >= 50:
            self.is_model_trained = True
            logger.info("Model accuracy is above 50%, trades are allowed.")
            self.save_model()
        else:
            self.is_model_trained = False
            logger.info("Model accuracy is below 50%, no trades will be allowed.")

    def predict_signal(self, df: pd.DataFrame) -> str:
        """Predict trading signal based on the trained model."""
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

            buy_probability, sell_probability = probabilities[-1]
            buy_threshold, sell_threshold = 0.80, 0.80

            if buy_probability >= buy_threshold:
                signal = "buy"
            elif sell_probability >= sell_threshold:
                signal = "sell"
            else:
                signal = "no trade"

            logger.info(f"Symbol: {self.symbol_name}, Predicted signal: {signal}")
            logger.info(f"Symbol: {self.symbol_name}, Buy probability: {buy_probability:.4f}, Sell probability: {sell_probability:.4f}")
            logger.info(f"Symbol: {self.symbol_name}, Buy threshold: {buy_threshold:.2f}, Sell threshold: {sell_threshold:.2f}")

            self._log_feature_importance(X)

            return signal

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "no signal"

    def _log_feature_importance(self, X: pd.DataFrame) -> None:
        """Log feature importance and values."""
        feature_importance = self.model.feature_importances_
        feature_names = X.columns
        for name, importance in zip(feature_names, feature_importance):
            logger.info(f"Symbol: {self.symbol_name}, Feature {name}: importance = {importance:.4f}, value = {X.iloc[-1][name]}")
