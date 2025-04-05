import os
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from integrations import TestAPI
from units import TestDataIngestion, TestPredictSeries

if __name__ == "__main__":
    loader = unittest.TestLoader()

    suite = unittest.TestSuite()

    # Ajouter les tests à la suite
    suite.addTests(loader.loadTestsFromTestCase(TestDataIngestion))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictSeries))
    suite.addTests(loader.loadTestsFromTestCase(TestAPI))

    runner = unittest.TextTestRunner(verbosity=2)

    result = runner.run(suite)

    sys.exit(not result.wasSuccessful())
