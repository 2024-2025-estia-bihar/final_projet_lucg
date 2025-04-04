import unittest
import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au chemin Python
sys.path.append(str(Path(__file__).parent.parent.parent))

# Importer les tests
from units import TestDataIngestion, TestPredictSeries
from integrations import TestAPI

if __name__ == "__main__":
    # Créer un TestLoader
    loader = unittest.TestLoader()
    
    # Créer une suite de tests
    suite = unittest.TestSuite()
    
    # Ajouter les tests à la suite
    suite.addTests(loader.loadTestsFromTestCase(TestDataIngestion))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictSeries))
    suite.addTests(loader.loadTestsFromTestCase(TestAPI))
    
    # Créer un TextTestRunner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Exécuter les tests
    result = runner.run(suite)
    
    # Sortir avec un code d'erreur si des tests ont échoué
    sys.exit(not result.wasSuccessful())