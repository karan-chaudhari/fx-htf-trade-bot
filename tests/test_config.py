import unittest
import os
import yaml

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

    def test_config_exists(self):
        self.assertTrue(os.path.exists(self.config_path))

    def test_config_load(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.assertIn('MODEL_DIR', config)
        self.assertIn('CONFIDENCE_THRESHOLD', config)

if __name__ == '__main__':
    unittest.main()
