import time
import os
import MetaTrader5 as mt5
import numpy as np
import pandas as pd  # Make sure to import pandas
import traceback
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

    # Collect historical close, low, high, and open prices for each symbol
    historical_data = []  # Create a list to hold historical data for all symbols

    for symbol in symbols:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1000)
        if rates is None or len(rates) == 0:
            logger.error(f"No data returned for symbol {symbol}.")
            continue

        close_prices = np.array([rate['close'] for rate in rates])
        low_prices = np.array([rate['low'] for rate in rates])
        high_prices = np.array([rate['high'] for rate in rates])
        open_prices = np.array([rate['open'] for rate in rates])  # Retrieve open prices

        if len(close_prices) < 50 or len(low_prices) < 50 or len(high_prices) < 50 or len(open_prices) < 50:  # Minimum length for indicators
            logger.error(f"Not enough data for symbol {symbol}. Found close: {len(close_prices)}, low: {len(low_prices)}, high: {len(high_prices)}, open: {len(open_prices)}")
            continue

        # Append the data to the historical_data list as a DataFrame
        symbol_data = pd.DataFrame({
            'open': open_prices,   # Add open prices
            'close': close_prices,
            'low': low_prices,
            'high': high_prices,
        })

        historical_data.append(symbol_data)

    # Combine the data from all symbols into one DataFrame
    if len(historical_data) == 0:
        logger.error("No valid historical data for any symbols. Exiting.")
        exit(1)

    combined_df = pd.concat(historical_data, ignore_index=True)

    try:
        if combined_df.empty:
            logger.error("Combined DataFrame is empty. Cannot train the model.")
            exit(1)

        # Check for minimum row count
        if len(combined_df) < 200:
            logger.error("Not enough data to train the model. Minimum required is 200 rows.")
            exit(1)

        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close']  # Add 'open' to the required columns
        if not all(col in combined_df.columns for col in required_columns):
            logger.error("Combined DataFrame is missing required columns: open, high, low, close.")
            exit(1)

        logger.info(f"Training model with {len(combined_df)} data points.")
        logger.info(f"Combined DataFrame shape: {combined_df.shape}")
        logger.info(combined_df.head())  # Debugging: show the first few rows of the DataFrame
        
        # Train the model using the combined dataframe
        try:
            indicator_calculator.train_model(combined_df)
            logger.info("Model training completed successfully.")
        except MemoryError:
            logger.error("MemoryError: Not enough memory available for model training.")
        except Exception as e:
            logger.error(f"Unexpected error during model training: {e}")
            logger.error(traceback.format_exc())  # Capture and log the full traceback

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        exit(1)

    try:
        while True:
            trade_manager.monitor_trade()
            for symbol in symbols:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1000)
                if rates is None or len(rates) == 0:
                    logger.error(f"No data returned for symbol {symbol}. Skipping.")
                    continue

                close_prices = np.array([rate['close'] for rate in rates])
                low_prices = np.array([rate['low'] for rate in rates])  # Collect low prices for prediction
                high_prices = np.array([rate['high'] for rate in rates])
                open_prices = np.array([rate['open'] for rate in rates])  # Collect open prices for prediction

                if len(close_prices) < 50 or len(low_prices) < 50:  # Check for sufficient data for prediction
                    logger.error(f"Not enough close prices for symbol {symbol}. Skipping.")
                    continue

                # Create a DataFrame for prediction
                prediction_df = pd.DataFrame({
                    'open': open_prices,   # Add open prices
                    'close': close_prices,
                    'low': low_prices,
                    'high': high_prices
                })

                # Make predictions
                try:
                    # Ensure prediction_df indices are reset for alignment
                    prediction_df.reset_index(drop=True, inplace=True)

                    signal = indicator_calculator.predict_signal(prediction_df)
                    if signal == "buy":
                        trade_manager.place_order(symbol, "buy")
                        logger.info(f"Placed buy order for {symbol}.")
                    elif signal == "sell":
                        trade_manager.place_order(symbol, "sell")
                        logger.info(f"Placed sell order for {symbol}.")
                except Exception as e:
                    logger.error(f"Error during prediction for {symbol}: {e}")

            time.sleep(10)
            logger.info("Checking for next trade")

    except KeyboardInterrupt:
        logger.error("Trading bot stopped by user.")
    finally:
        connector.shutdown()
