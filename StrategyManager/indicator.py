from talib import RSI, SMA, MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
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
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def prepare_features(self, close_prices):
        if len(close_prices) < 200:  # Ensure we have enough data for the longest SMA
            logger.error("Not enough close prices to calculate indicators.")
            raise ValueError("Not enough close prices to calculate indicators.")

        # Create features from traditional indicators
        sma_50 = self.calculate_sma(close_prices, 50)
        sma_200 = self.calculate_sma(close_prices, 200)
        rsi = self.calculate_rsi(close_prices)
        macd, macdsignal = self.calculate_macd(close_prices)

        # Log lengths
        logger.info(f"Lengths - SMA 50: {len(sma_50)}, SMA 200: {len(sma_200)}, RSI: {len(rsi)}, MACD: {len(macd)}, MACD Signal: {len(macdsignal)}")

        # Combine features into a DataFrame
        data = pd.DataFrame({
            'sma_50': sma_50,
            'sma_200': sma_200,
            'rsi': rsi,
            'macd': macd,
            'macdsignal': macdsignal
        })

        # Drop NaN values (if any)
        data.dropna(inplace=True)

        # Log the length of the DataFrame after dropping NaN values
        logger.info(f"Data length after dropping NaN: {len(data)}")

        # Check if the data is empty after dropping NaNs
        if len(data) == 0:
            logger.error("No valid data after dropping NaNs. Unable to create target column.")
            raise ValueError("No valid data after dropping NaNs.")

        # Create target
        target_start_index = len(close_prices) - len(data) - 1  # Adjust index based on dropped rows
        if target_start_index + 1 + len(data) <= len(close_prices):
            data['target'] = (close_prices[target_start_index + 1:target_start_index + 1 + len(data)] > close_prices[target_start_index:target_start_index + len(data)]).astype(int)
        else:
            logger.error("Target creation out of bounds.")
            raise ValueError("Target creation out of bounds.")

        # Ensure the target column was created
        if 'target' not in data.columns:
            logger.error("Target column not created.")
            raise KeyError("Target column not created.")

        return data.drop('target', axis=1), data['target']

    def train_model(self, close_prices):  # This method expects only one argument
        X, y = self.prepare_features(close_prices)
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        logger.info(f"Model trained with accuracy: {self.model.score(X_test, y_test)}")

    def predict_signal(self, close_prices):
        X, _ = self.prepare_features(close_prices)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Decision logic based on the latest prediction
        if predictions[-1] == 1:
            logger.info("ML Buy Signal detected")
            return "buy"
        else:
            logger.info("ML Sell Signal detected")
            return "sell"
