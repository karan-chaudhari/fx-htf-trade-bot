import MetaTrader5 as mt5
import os
from logger.logger import logger
import yaml


def load_config():
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'), 'r') as f:
        return yaml.safe_load(f)

config = load_config()

class TradeManager:
    """Manages trading operations like placing and closing orders, with ATR-based risk management."""

    def __init__(self, volume):
        self.volume = volume
        self.atr_stoploss_mult = config.get('ATR_STOPLOSS_MULTIPLIER', 1.5)
        self.atr_takeprofit_mult = config.get('ATR_TAKEPROFIT_MULTIPLIER', 2.0)
        self.trade_start_hour = config.get('TRADE_START_HOUR', 6)
        self.trade_end_hour = config.get('TRADE_END_HOUR', 20)

    def can_open_position(self, symbol):
        """Check if there are fewer than allowed open positions for the symbol."""
        positions = mt5.positions_get(symbol=symbol)
        max_pos = int(os.getenv('NO_OF_POS', 2))
        if positions is None:
            return True  # No positions open
        return len(positions) < max_pos

    def place_order(self, symbol, action, atr=None):
        import datetime
        now_utc = datetime.datetime.utcnow().hour
        if not (self.trade_start_hour <= now_utc < self.trade_end_hour):
            logger.info(f"Trading not allowed at this hour: {now_utc} UTC. Allowed: {self.trade_start_hour}-{self.trade_end_hour}")
            return

        if not self.can_open_position(symbol):
            logger.info(f"Cannot open more than allowed positions for {symbol}.")
            return

        order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if action == "buy" else mt5.symbol_info_tick(symbol).bid

        # ATR-based stop-loss/take-profit
        if atr is not None:
            stop_loss = price - self.atr_stoploss_mult * atr if action == "buy" else price + self.atr_stoploss_mult * atr
            take_profit = price + self.atr_takeprofit_mult * atr if action == "buy" else price - self.atr_takeprofit_mult * atr
        else:
            stop_loss = 0
            take_profit = 0

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
            "sl": stop_loss,
            "tp": take_profit,
        }

        try:
            result = mt5.order_send(order_request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to place order for {symbol}: {result.retcode} - {result.comment}")
            else:
                logger.info(f"Order placed successfully for {symbol} at {price} with SL {stop_loss} and TP {take_profit}")
        except Exception as e:
            logger.error(f"Exception in placing order: {e}")

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
                # ATR-based dynamic close logic (optional)
                # Example: close if profit > 2*ATR or loss < -1.5*ATR
                # You can fetch ATR from your indicator logic and pass it here
                if position.profit >= self.atr_takeprofit_mult:
                    logger.info(f"Profit target reached on {position.symbol}! Closing trade.")
                    self.close_order(position.ticket, position.symbol, position.volume)
                elif position.profit <= -self.atr_stoploss_mult:
                    logger.info(f"Loss threshold reached on {position.symbol}! Closing trade.")
                    self.close_order(position.ticket, position.symbol, position.volume)
        else:
            logger.info(f"No open positions.")

