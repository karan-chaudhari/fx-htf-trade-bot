import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from StrategyManager.indicator import MLIndicatorCalculator

class TestMLIndicatorCalculator(unittest.TestCase):
    def setUp(self):
        self.symbol = 'EURUSDm'
        self.calculator = MLIndicatorCalculator(self.symbol)
        # Minimal DataFrame for indicator calculation
        self.df = pd.DataFrame({
            'open': [1.1]*250,
            'high': [1.2]*250,
            'low': [1.0]*250,
            'close': [1.15]*250,
            'volume': [1000]*250
        })

    def test_prepare_combined_features(self):
        X, y = self.calculator.prepare_combined_features(self.df)
        self.assertTrue(len(X) > 0)
        self.assertTrue(len(y) > 0)

    @patch('StrategyManager.indicator.joblib.dump')
    def test_save_model(self, mock_dump):
        self.calculator.model = MagicMock()
        self.calculator.scaler = MagicMock()
        self.calculator.feature_names = ['a', 'b']
        self.calculator.save_model()
        mock_dump.assert_called()

    @patch('StrategyManager.indicator.joblib.load')
    def test_load_model(self, mock_load):
        mock_load.return_value = {'model': MagicMock(), 'scaler': MagicMock()}
        self.calculator.load_model()
        self.assertTrue(self.calculator.model is not None)
        self.assertTrue(self.calculator.scaler is not None)

if __name__ == '__main__':
    unittest.main()
