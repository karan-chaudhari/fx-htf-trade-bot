from talib import RSI, SMA, MACD
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
