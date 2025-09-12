import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import talib  # Import TA-Lib instead of ta
from logger.logger import logger
from typing import Tuple
import joblib  # Import joblib for model persistence
import os
import yaml
import hashlib

def load_config():
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'), 'r') as f:
        return yaml.safe_load(f)

config = load_config()

def get_feature_hash(feature_names):
    """Hash the feature names list to detect changes."""
    return hashlib.md5(",".join(feature_names).encode()).hexdigest()

class IndicatorCalculator:
    """Calculates both traditional and price action/SMC indicators for Forex trading signals."""

    def calculate_traditional_indicators(self, df):
        # Add technical indicators to the DataFrame using TA-Lib
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['sma'] = talib.SMA(df['close'], timeperiod=50)
        df['ema'] = talib.EMA(df['close'], timeperiod=21)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macdsignal'] = macdsignal
        df['macdhist'] = macdhist
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        # Additional indicators for more features
        df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['roc'] = talib.ROC(df['close'], timeperiod=10)
        df['obv'] = talib.OBV(df['close'], df['volume']) if 'volume' in df.columns else 0
        # More indicators
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14) if 'volume' in df.columns else 0
        df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        # Time-based features
        if 'datetime' in df.columns:
            df['hour'] = pd.to_datetime(df['datetime']).dt.hour
            df['dayofweek'] = pd.to_datetime(df['datetime']).dt.dayofweek
        # Rolling window statistics
        df['close_mean_10'] = df['close'].rolling(window=10).mean()
        df['close_std_10'] = df['close'].rolling(window=10).std()
        df['close_min_10'] = df['close'].rolling(window=10).min()
        df['close_max_10'] = df['close'].rolling(window=10).max()
        # Lagged features (previous values)
        for col in ['close', 'rsi', 'sma', 'ema', 'adx', 'macd', 'atr']:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag2'] = df[col].shift(2)
        df.dropna(inplace=True)
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
        self.symbol_name = symbol_name
        self.model_dir = config.get('MODEL_DIR', 'model/')
        self.model_path = os.path.join(self.model_dir, f"{symbol_name}_model.pkl")
        self.scaler_path = os.path.join(self.model_dir, f"{symbol_name}_scaler.pkl")
        self.feature_hash_path = os.path.join(self.model_dir, f"{symbol_name}_featurehash.txt")
        self.xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lgbm_model = LGBMClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_model_trained = False
        self.feature_names = None
        self.selector = None  # Will hold SelectFromModel
        self.load_model()

    def save_model(self):
        """Saves the trained ensemble models, scaler, selector, and feature hash to disk."""
        try:
            joblib.dump({'xgb_model': self.xgb_model, 'rf_model': self.rf_model, 'lgbm_model': self.lgbm_model, 'scaler': self.scaler, 'selector': self.selector}, self.model_path)
            # Save feature hash for auto-cleanup
            if self.feature_names:
                with open(self.feature_hash_path, 'w') as f:
                    f.write(get_feature_hash(self.feature_names))
            logger.info(f"Ensemble models (XGB, RF, LGBM), scaler, selector, and symbol '{self.symbol_name}' saved to {self.model_path}.")
        except Exception as e:
            logger.error(f"Failed to save ensemble models, scaler, selector, and symbol: {e}")

    def load_model(self):
        """Loads the trained ensemble models, scaler, selector, and checks feature hash for auto-cleanup."""
        if os.path.exists(self.model_path):
            try:
                saved_objects = joblib.load(self.model_path)
                self.xgb_model = saved_objects['xgb_model']
                self.rf_model = saved_objects['rf_model']
                self.lgbm_model = saved_objects['lgbm_model']
                self.scaler = saved_objects['scaler']
                self.selector = saved_objects.get('selector', None)
                # Check feature hash
                if os.path.exists(self.feature_hash_path):
                    with open(self.feature_hash_path, 'r') as f:
                        saved_hash = f.read().strip()
                    # Will check actual hash after features are set in prepare_combined_features
                    self.saved_feature_hash = saved_hash
                else:
                    self.saved_feature_hash = None
                self.is_model_trained = True
                logger.info(f"Loaded ensemble models (XGB, RF, LGBM), scaler, selector, and symbol '{self.symbol_name}' from {self.model_path}.")
            except Exception as e:
                logger.error(f"Failed to load ensemble models, scaler, selector, and symbol: {e}")
                self.is_model_trained = False
        else:
            logger.info("No existing model found. A new model will be trained.")

    def prepare_combined_features(self, df):
        min_features = config.get('MIN_FEATURES', 200)
        if len(df) < min_features:
            logger.error(f"Not enough data to calculate features. Minimum required: {min_features}")
            raise ValueError("Not enough data to calculate features.")

        # Calculate traditional indicators (now with more features)
        df = self.calculate_traditional_indicators(df)

        # Detect support, resistance
        support, resistance = self.detect_support_resistance(df['close'])

        # Combine all features into a DataFrame (add new features)
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
            'macdsignal': df['macdsignal'],
            'macdhist': df['macdhist'],
            'atr': df['atr'],
            'willr': df['willr'],
            'cci': df['cci'],
            'stoch_k': df['stoch_k'],
            'stoch_d': df['stoch_d'],
            'roc': df['roc'],
            'obv': df['obv'],
            # Lagged features
            'close_lag1': df['close_lag1'],
            'close_lag2': df['close_lag2'],
            'rsi_lag1': df['rsi_lag1'],
            'rsi_lag2': df['rsi_lag2'],
            'sma_lag1': df['sma_lag1'],
            'sma_lag2': df['sma_lag2'],
            'ema_lag1': df['ema_lag1'],
            'ema_lag2': df['ema_lag2'],
            'adx_lag1': df['adx_lag1'],
            'adx_lag2': df['adx_lag2'],
            'macd_lag1': df['macd_lag1'],
            'macd_lag2': df['macd_lag2'],
            'atr_lag1': df['atr_lag1'],
            'atr_lag2': df['atr_lag2'],
        })

        # Drop NaN values
        features.dropna(inplace=True)

        # Save feature names for hash check
        self.feature_names = list(features.columns)

        # Auto-cleanup: if feature hash changed, delete old model/scaler
        current_hash = get_feature_hash(self.feature_names)
        if hasattr(self, 'saved_feature_hash') and self.saved_feature_hash and self.saved_feature_hash != current_hash:
            logger.info("Feature set changed, deleting old model and scaler to avoid mismatch.")
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            if os.path.exists(self.scaler_path):
                os.remove(self.scaler_path)
            if os.path.exists(self.feature_hash_path):
                os.remove(self.feature_hash_path)
            self.is_model_trained = False
            self.saved_feature_hash = current_hash

        logger.info(f"Symbol: {self.symbol_name}, Features data length after dropping NaN: {len(features)}")

        if len(features) == 0:
            logger.error("No valid features after dropping NaNs.")
            raise ValueError("No valid features after dropping NaNs.")

        # Align the indices of df['close'] and features to create target labels
        target_series = df['close'].iloc[len(df) - len(features):]

        # Create target labels (price direction)
        features['target'] = (target_series.shift(-1) > target_series).astype(int)

        return features.drop('target', axis=1), features['target']

    def train_model(self, df):
        if self.is_model_trained:
            logger.info("Model is already trained. Loading existing model.")
            return

        from sklearn.feature_selection import SelectFromModel

        X, y = self.prepare_combined_features(df)
        X_scaled = self.scaler.fit_transform(X)

        # XGBoost grid search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(self.xgb_model, param_grid, cv=tscv, scoring='accuracy', n_jobs=1, verbose=2)
        grid_search.fit(X_scaled, y)

        # Feature selection: keep only important features
        self.selector = SelectFromModel(grid_search.best_estimator_, prefit=True, threshold='median')
        X_selected = self.selector.transform(X_scaled)
        selected_features = X.columns[self.selector.get_support(indices=True)]
        logger.info(f"Selected features: {list(selected_features)}")

        # Retrain XGBoost on selected features
        grid_search_selected = GridSearchCV(self.xgb_model, param_grid, cv=tscv, scoring='accuracy', n_jobs=1, verbose=2)
        grid_search_selected.fit(X_selected, y)
        self.xgb_model = grid_search_selected.best_estimator_
        logger.info(f"Best XGBoost Parameters: {grid_search_selected.best_params_}")

        # Train RandomForest and LightGBM on selected features
        self.rf_model.fit(X_selected, y)
        self.lgbm_model.fit(X_selected, y)

        # Evaluate ensemble on the last split
        for train_idx, test_idx in tscv.split(X_selected):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        xgb_pred = self.xgb_model.predict(X_test)
        rf_pred = self.rf_model.predict(X_test)
        lgbm_pred = self.lgbm_model.predict(X_test)
        # Majority vote
        import scipy.stats
        ensemble_pred = scipy.stats.mode([xgb_pred, rf_pred, lgbm_pred], axis=0)[0][0]
        accuracy = accuracy_score(y_test, ensemble_pred) * 100
        logger.info(f"Ensemble model (XGB, RF, LGBM) trained with accuracy (last split): {accuracy:.2f}%")

        if accuracy >= 50:
            self.is_model_trained = True
            logger.info("Ensemble model accuracy is above 50%, trades are allowed.")
            self.save_model()
        else:
            self.is_model_trained = False
            logger.info("Ensemble model accuracy is below 50%, no trades will be allowed.")

    def predict_signal(self, df):
        if not self.is_model_trained:
            logger.info("Model not trained or accuracy below 50%, no trades allowed.")
            return "trade not allowed"

        try:
            X, _ = self.prepare_combined_features(df)
            X_scaled = self.scaler.transform(X)
            # Use the same selector as during training
            if self.selector is not None:
                X_selected = self.selector.transform(X_scaled)
            else:
                logger.error("Feature selector not found. Cannot proceed with prediction.")
                return "no signal"

            # Ensemble prediction: average probabilities and majority vote
            xgb_probs = self.xgb_model.predict_proba(X_selected)
            rf_probs = self.rf_model.predict_proba(X_selected)
            lgbm_probs = self.lgbm_model.predict_proba(X_selected)
            avg_probs = (xgb_probs + rf_probs + lgbm_probs) / 3

            if len(avg_probs) == 0:
                logger.info("No predictions made.")
                return "no signal"

            buy_probability = avg_probs[-1, 1]
            sell_probability = avg_probs[-1, 0]

            # Filter out low-confidence signals (only trade if prob >0.6 or <0.4)
            confidence_threshold = 0.6
            if buy_probability >= confidence_threshold:
                signal = "buy"
            elif sell_probability >= confidence_threshold:
                signal = "sell"
            else:
                signal = "no signal"

            # Log the signal and probabilities
            logger.info(f"Symbol: {self.symbol_name}, Predicted signal: {signal}")
            logger.info(f"Symbol: {self.symbol_name}, Buy probability: {buy_probability:.4f}, Sell probability: {sell_probability:.4f}")
            logger.info(f"Symbol: {self.symbol_name}, Confidence threshold: {confidence_threshold:.2f}")

            # Log additional feature importance information (from XGBoost)
            feature_importance = self.xgb_model.feature_importances_
            feature_names = X.columns
            for name, importance in zip(feature_names, feature_importance):
                logger.info(f"Symbol: {self.symbol_name}, Feature {name}: importance = {importance:.4f}, value = {X.iloc[-1][name]}")

            return signal

        except Exception as e:
            logger.error(f"Error in generating signal: {e}")
            return "no signal"
