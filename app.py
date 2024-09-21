import time
import os
import MetaTrader5 as mt5
import numpy as np
from dotenv import load_dotenv 
from MT5Connector.mt5_connector import MT5Connector
from StrategyManager.indicator import IndicatorCalculator
from TradeManager.trade_manager import TradeManager
from logger.logger import logger

# Load environment variables from .env file
load_dotenv()

# Main execution
if __name__ == "__main__":
    logger.info("Starting the application...")
    account_number = int(os.getenv('MT5_ACCOUNT'))
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')

    # Get symbols from the environment variable
    symbols = os.getenv('MT5_SYMBOLS').split(',')

    connector = MT5Connector(account_number, password, server)
    if not connector.initialize():
        exit(1)
        
    volume = float(os.getenv('TRADE_VOL'))
    trade_manager = TradeManager(volume)

    try:
        while True:
            # Check for existing open positions for the symbol
            trade_manager.monitor_trade()

            for symbol in symbols:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
                close_prices = np.array([rate['close'] for rate in rates])

                indicator_calculator = IndicatorCalculator()
                signals = indicator_calculator.check_signals(symbol, close_prices)

                # Place orders based on signals
                for strategy, signal in signals.items():
                    if signal == "buy":
                        trade_manager.place_order(symbol, "buy")
                    elif signal == "sell":
                        trade_manager.place_order(symbol, "sell")

            time.sleep(60)

    except KeyboardInterrupt:
        logger.error("Trading bot stopped by user.")

    finally:
        connector.shutdown()
