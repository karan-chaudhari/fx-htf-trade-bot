import MetaTrader5 as mt5
import os
from logger.logger import logger


class TradeManager:
    """Manages trading operations like placing and closing orders."""

    def __init__(self, volume):
        self.volume = volume

    def can_open_position(self, symbol):
        """Check if there are fewer than 2 open positions for the symbol."""
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return True  # No positions open
        return len(positions) < int(os.getenv('NO_OF_POS'))

    def place_order(self, symbol, action):
        if not self.can_open_position(symbol):
            logger.info(f"Cannot open more than {int(os.getenv('NO_OF_POS'))} positions for {symbol}.")
            return

        order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if action == "buy" else mt5.symbol_info_tick(symbol).bid

        order_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.volume,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": "Auto-trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(order_request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to place order for {symbol}: {result.retcode}")
        else:
            logger.info(f"Order placed successfully for {symbol}")

    # Function to close an open position
    def close_order(self, position_id, symbol, volume):
        # Retrieve the current position based on the position ID
        position = mt5.positions_get(ticket=position_id)
        if position is None or len(position) == 0:
            logger.error(f"Failed to find position with ID {position_id} for {symbol}. Error: {mt5.last_error()}")
            return None

        # Determine the order type (opposite of the current position type)
        order_type = mt5.ORDER_TYPE_SELL if position[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        # Get the current price
        price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask

        # Retrieve the symbol info to determine supported filling modes
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Failed to retrieve symbol info for {symbol}. Error: {mt5.last_error()}")
            return None

        # Select a supported filling mode; adjust if ORDER_FILLING_RETURN is not supported
        # Try different modes based on the available options (ORDER_FILLING_FOK or ORDER_FILLING_IOC)
        filling_mode = mt5.ORDER_FILLING_RETURN  # Default mode; change based on the availability

        # Create a request to close the position
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,  # opposite of the current position
            "position": position_id,  # position ID to close
            "price": price,
            "deviation": 20,  # allowed price deviation
            "magic": 123456,  # identifier for the trade
            "comment": "Close trade",
            "type_filling": filling_mode,  # Use the appropriate filling mode
        }

        # Send the close request
        result = mt5.order_send(close_request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Successfully closed position {position_id} on {symbol}")
            return result
        else:
            # Try another filling mode if supported
            if filling_mode == mt5.ORDER_FILLING_RETURN:
                close_request["type_filling"] = mt5.ORDER_FILLING_IOC  # Immediate or Cancel as an alternative
                result = mt5.order_send(close_request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Successfully closed position {position_id} on {symbol} using IOC filling mode")
                    return result
                else:
                    logger.error(f"Retry failed. Error: {result.comment}")
            return None

    def monitor_trade(self):
        """Monitor open positions and perform actions if necessary."""
        positions = mt5.positions_get()
        if positions:
            for position in positions:
                logger.info(f"Position {position.ticket}: {position.type} - Volume: {position.volume} - Profit: {position.profit}")
                # Close the trade if profit target is met
                # if current_profit >= profit_target:
                # Close the trade if profit target is met (>= 10) or loss threshold is hit (<= -100)
                if position.profit >= 0.2:
                    logger.info(f"Profit target reached on {position.symbol}! Closing trade.")
                    self.close_order(position.ticket, position.symbol, position.volume)
                # elif position.profit <= -1:
                #     logger.info(f"Loss threshold reached on {position.symbol}! Closing trade.")
                #     self.close_order(position.ticket, position.symbol, position.volume)
        else:
            logger.info(f"No open positions.")

