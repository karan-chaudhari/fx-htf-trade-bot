import MetaTrader5 as mt5
from logger.logger import logger


class MT5Connector:
    """Handles the connection to MetaTrader 5."""

    def __init__(self, account_number, password, server):
        self.account_number = account_number
        self.password = password
        self.server = server

    def initialize(self):
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return False
        if mt5.login(self.account_number, password=self.password, server=self.server):
            logger.info(f"Successfully logged in to account {self.account_number}")
            return True
        else:
            logger.error(f"Failed to log in. Error: {mt5.last_error()}")
            mt5.shutdown()
            return False

    def shutdown(self):
        mt5.shutdown()
        logger.info("MT5 connection closed.")
