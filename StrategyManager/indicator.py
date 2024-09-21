from talib import RSI, SMA, MACD

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

    def check_signals(self, close_prices):
        signals = {}
        signals['sma'] = self.check_sma_signal(close_prices)
        signals['rsi'] = self.check_rsi_signal(close_prices)
        signals['macd'] = self.check_macd_signal(close_prices)
        return signals

    def check_sma_signal(self, close_prices):
        sma_50 = self.calculate_sma(close_prices, 50)
        sma_200 = self.calculate_sma(close_prices, 200)
        if sma_50[-1] > sma_200[-1]:
            return "buy"
        elif sma_50[-1] < sma_200[-1]:
            return "sell"
        return None

    def check_rsi_signal(self, close_prices):
        rsi = self.calculate_rsi(close_prices)
        if rsi[-1] < 30:
            return "buy"
        elif rsi[-1] > 70:
            return "sell"
        return None

    def check_macd_signal(self, close_prices):
        macd, macdsignal = self.calculate_macd(close_prices)
        if macd[-1] > macdsignal[-1]:
            return "buy"
        elif macd[-1] < macdsignal[-1]:
            return "sell"
        return None
