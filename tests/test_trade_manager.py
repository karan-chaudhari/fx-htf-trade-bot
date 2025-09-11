import unittest
from unittest.mock import patch, MagicMock
from TradeManager.trade_manager import TradeManager

class TestTradeManager(unittest.TestCase):
    def setUp(self):
        self.trade_manager = TradeManager(volume=0.1)

    @patch('MetaTrader5.positions_get')
    def test_can_open_position_no_positions(self, mock_positions_get):
        mock_positions_get.return_value = None
        self.assertTrue(self.trade_manager.can_open_position('EURUSDm'))

    @patch('MetaTrader5.positions_get')
    def test_can_open_position_with_positions(self, mock_positions_get):
        mock_positions_get.return_value = [MagicMock(), MagicMock()]
        with patch('os.getenv', return_value='2'):
            self.assertFalse(self.trade_manager.can_open_position('EURUSDm'))

    @patch('MetaTrader5.order_send')
    @patch('MetaTrader5.symbol_info_tick')
    @patch('MetaTrader5.positions_get')
    def test_place_order(self, mock_positions_get, mock_symbol_info_tick, mock_order_send):
        mock_positions_get.return_value = None
        mock_symbol_info_tick.return_value = MagicMock(ask=1.2, bid=1.1)
        mock_order_send.return_value = MagicMock(retcode=0, comment='OK')
        self.trade_manager.place_order('EURUSDm', 'buy', atr=0.01)
        mock_order_send.assert_called()

if __name__ == '__main__':
    unittest.main()
