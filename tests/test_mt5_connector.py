import unittest
from unittest.mock import patch, MagicMock
from MT5Connector.mt5_connector import MT5Connector

class TestMT5Connector(unittest.TestCase):
    def setUp(self):
        self.connector = MT5Connector(123456, 'password', 'server')

    @patch('MetaTrader5.initialize')
    @patch('MetaTrader5.login')
    def test_initialize_success(self, mock_login, mock_initialize):
        mock_initialize.return_value = True
        mock_login.return_value = True
        self.assertTrue(self.connector.initialize())

    @patch('MetaTrader5.initialize')
    def test_initialize_fail(self, mock_initialize):
        mock_initialize.return_value = False
        self.assertFalse(self.connector.initialize())

if __name__ == '__main__':
    unittest.main()
