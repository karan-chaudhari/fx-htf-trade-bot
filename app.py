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

    # Collect historical close, low, and high prices for training the model
    historical_close_prices = []
    historical_low_prices = []
    historical_high_prices = []

    for symbol in symbols:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1000)
        if rates is None or len(rates) == 0:
            logger.error(f"No data returned for symbol {symbol}.")
            continue
        
        close_prices = np.array([rate['close'] for rate in rates])
        low_prices = np.array([rate['low'] for rate in rates])
        high_prices = np.array([rate['high'] for rate in rates])

        if len(close_prices) < 50 or len(low_prices) < 50 or len(high_prices) < 50:  # Minimum length for indicators
            logger.error(f"Not enough data for symbol {symbol}. Found close: {len(close_prices)}, low: {len(low_prices)}, high: {len(high_prices)}")
            continue

        historical_close_prices.append(close_prices)
        historical_low_prices.append(low_prices)
        historical_high_prices.append(high_prices)

    # Combine data from all symbols for training
    combined_close_prices = np.concatenate(historical_close_prices)
    combined_low_prices = np.concatenate(historical_low_prices)
    combined_high_prices = np.concatenate(historical_high_prices)

    try:
        if len(combined_close_prices) == 0 or len(combined_low_prices) == 0 or len(combined_high_prices) == 0:
            logger.error("Combined prices arrays are empty. Cannot train the model.")
            exit(1)

        # Log the shapes of the combined price arrays
        logger.debug(f"Low prices shape: {combined_low_prices.shape}")
        logger.debug(f"High prices shape: {combined_high_prices.shape}")
        logger.debug(f"Close prices shape: {combined_close_prices.shape}")

        # Limit data size for training if too large
        if len(combined_close_prices) > 1000:
            combined_close_prices = combined_close_prices[-1000:]
            combined_low_prices = combined_low_prices[-1000:]
            combined_high_prices = combined_high_prices[-1000:]

        logger.info(f"Training model with {len(combined_close_prices)} data points.")
        
        # Train the model using combined data
        try:
            indicator_calculator.train_model(combined_low_prices, combined_high_prices, combined_close_prices)
            logger.info("Model training completed successfully.")
        except MemoryError:
            logger.error("MemoryError: Not enough memory available for model training.")
        except Exception as e:
            logger.error(f"Unexpected error during model training: {e}")

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
                low_prices = np.array([rate['low'] for rate in rates])  # Collect low prices for prediction
                if len(close_prices) < 50 or len(low_prices) < 50:  # Check for sufficient data for prediction
                    logger.error(f"Not enough close prices for symbol {symbol}. Skipping.")
                    continue

                # Make sure we have the correct number of arguments
                signal = indicator_calculator.predict_signal(high_prices, low_prices, close_prices)
                if signal == "buy":
                    trade_manager.place_order(symbol, "buy")
                elif signal == "sell":
                    trade_manager.place_order(symbol, "sell")

            time.sleep(60)
            logger.info("Checking for next trade")

    except KeyboardInterrupt:
        logger.error("Trading bot stopped by user.")
    finally:
        connector.shutdown()
