from talib import RSI, SMA, MACD, ATR, STOCH
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from logger.logger import logger

class IndicatorCalculator:
    """Calculates indicators for trading signals."""

    @staticmethod
    def calculate_sma(data, period):
        return SMA(data, timeperiod=period)

    @staticmethod
    def calculate_rsi(data, period=14):
        return RSI(data, timeperiod=period)

    @staticmethod
    def calculate_macd(data):
        macd, macdsignal, _ = MACD(data)
        return macd, macdsignal

    @staticmethod
    def calculate_atr(high, low, close, period=14):
        return ATR(high, low, close, timeperiod=period)

    @staticmethod
    def calculate_stoch(high, low, close):
        slowk, slowd = STOCH(high, low, close)
        return slowk, slowd

    def check_signals(self, symbol, close_prices):
        signals = {}
        signals['sma'] = self.check_sma_signal(symbol, close_prices)
        signals['rsi'] = self.check_rsi_signal(symbol, close_prices)
        signals['macd'] = self.check_macd_signal(symbol, close_prices)
        return signals

    def check_sma_signal(self, symbol, close_prices):
        sma_50 = self.calculate_sma(close_prices, 50)
        sma_200 = self.calculate_sma(close_prices, 200)
        if sma_50[-1] > sma_200[-1]:
            logger.info(f"SMA Buy Signal detected on {symbol}")
            return "buy"
        elif sma_50[-1] < sma_200[-1]:
            logger.info(f"SMA Sell Signal detected on {symbol}")
            return "sell"
        return None

    def check_rsi_signal(self, symbol, close_prices):
        rsi = self.calculate_rsi(close_prices)
        if rsi[-1] < 30:
            logger.info(f"RSI Buy Signal detected on {symbol} (RSI: {rsi[-1]})")
            return "buy"
        elif rsi[-1] > 70:
            logger.info(f"RSI Sell Signal detected on {symbol} (RSI: {rsi[-1]})")
            return "sell"
        return None

    def check_macd_signal(self, symbol, close_prices):
        macd, macdsignal = self.calculate_macd(close_prices)
        if macd[-1] > macdsignal[-1]:
            logger.info(f"MACD Buy Signal detected on {symbol}")
            return "buy"
        elif macd[-1] < macdsignal[-1]:
            logger.info(f"MACD Sell Signal detected on {symbol}")
            return "sell"
        return None

class MLIndicatorCalculator(IndicatorCalculator):
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.scaler = StandardScaler()
        self.is_model_trained = False

    def prepare_features(self, high, low, close_prices):
        if len(close_prices) < 200:
            logger.error("Not enough close prices to calculate indicators.")
            raise ValueError("Not enough close prices to calculate indicators.")

        # Create features from traditional indicators
        sma_50 = self.calculate_sma(close_prices, 50)
        sma_200 = self.calculate_sma(close_prices, 200)
        rsi = self.calculate_rsi(close_prices)
        macd, macdsignal = self.calculate_macd(close_prices)
        atr = self.calculate_atr(high, low, close_prices)
        stoch_k, stoch_d = self.calculate_stoch(high, low, close_prices)

        # Combine features into a DataFrame
        data = pd.DataFrame({
            'sma_50': sma_50,
            'sma_200': sma_200,
            'rsi': rsi,
            'macd': macd,
            'macdsignal': macdsignal,
            'atr': atr,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d
        })

        # Drop NaN values (if any)
        data.dropna(inplace=True)

        logger.info(f"Data length after dropping NaN: {len(data)}")

        if len(data) == 0:
            logger.error("No valid data after dropping NaNs.")
            raise ValueError("No valid data after dropping NaNs.")

        # Create target
        target_start_index = len(close_prices) - len(data) - 1
        if target_start_index + 1 + len(data) <= len(close_prices):
            data['target'] = (
                close_prices[target_start_index + 1:target_start_index + 1 + len(data)]
                > close_prices[target_start_index:target_start_index + len(data)]
            ).astype(int)
        else:
            logger.error("Target creation out of bounds.")
            raise ValueError("Target creation out of bounds.")

        if 'target' not in data.columns:
            logger.error("Target column not created.")
            raise KeyError("Target column not created.")

        return data.drop('target', axis=1), data['target']

    def train_model(self, high, low, close_prices):
        X, y = self.prepare_features(high, low, close_prices)
        X_scaled = self.scaler.fit_transform(X)

        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # Use n_jobs=1 to run GridSearchCV in the main thread
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=2)
        grid_search.fit(X_scaled, y)

        # Use the best estimator from the Grid Search
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

    def predict_signal(self, high, low, close_prices):
        if not self.is_model_trained:
            logger.info("Model not trained or accuracy below 60%, no trades allowed.")
            return "trade not allowed"

        try:
            X, _ = self.prepare_features(high, low, close_prices)
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)

            if len(predictions) == 0:
                logger.info("No trades available. Trade not found.")
                return "trade not found"

            if predictions[-1] == 1:
                logger.info("ML Buy Signal detected")
                return "buy"
            elif predictions[-1] == 0:
                logger.info("ML Sell Signal detected")
                return "sell"
            else:
                logger.info("No valid trade signal found.")
                return "trade not found"
        except Exception as e:
            logger.error(f"Error predicting trade signal: {str(e)}")
            return "error"
