import time
import os
import MetaTrader5 as mt5
import numpy as np
from dotenv import load_dotenv
from MT5Connector.mt5_connector import MT5Connector
from StrategyManager.indicator import MLIndicatorCalculator
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
        logger.error("Failed to initialize MT5 Connector.")
        exit(1)

    volume = float(os.getenv('TRADE_VOL'))
    trade_manager = TradeManager(volume)

    # Initialize the ML-based Indicator Calculator
    indicator_calculator = MLIndicatorCalculator()

    # Train the model on historical data
    historical_data = []
    for symbol in symbols:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1000)
        if rates is None or len(rates) == 0:
            logger.error(f"No data returned for symbol {symbol}.")
            continue
        historical_close_prices = np.array([rate['close'] for rate in rates])
        if len(historical_close_prices) < 50:  # Minimum length for indicators
            logger.error(f"Not enough data for symbol {symbol}. Found: {len(historical_close_prices)}")
            continue
        historical_data.append(historical_close_prices)

    if not historical_data:
        logger.error("No valid historical data collected. Exiting.")
        exit(1)

    # Combine data from all symbols for training
    combined_close_prices = np.concatenate(historical_data)

    try:
        if len(combined_close_prices) == 0:
            logger.error("Combined close prices array is empty. Cannot train the model.")
            exit(1)

        logger.info(f"Training model with {len(combined_close_prices)} data points.")
        logger.debug(f"Combined close prices: {combined_close_prices[:10]}...")

        # Train the model using combined data
        indicator_calculator.train_model(combined_close_prices)  # Adjusted to match method signature
        logger.info("Model training completed successfully.")

    except ValueError as e:
        logger.error(f"ValueError during model training: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        exit(1)

    try:
        while True:
            trade_manager.monitor_trade()
            for symbol in symbols:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1000)
                if rates is None or len(rates) == 0:
                    logger.error(f"No data returned for symbol {symbol}. Skipping.")
                    continue
                close_prices = np.array([rate['close'] for rate in rates])
                if len(close_prices) < 50:  # Check for sufficient data for prediction
                    logger.error(f"Not enough close prices for symbol {symbol}. Skipping.")
                    continue

                signal = indicator_calculator.predict_signal(close_prices)
                if signal == "buy":
                    trade_manager.place_order(symbol, "buy")
                elif signal == "sell":
                    trade_manager.place_order(symbol, "sell")

            time.sleep(60)

    except KeyboardInterrupt:
        logger.error("Trading bot stopped by user.")
    finally:
        connector.shutdown()
