# Common project paths
from pathlib import Path

# Project root directory (parent of the PyISV package)
PROJECT_ROOT = Path(__name__).resolve().parent

# Directories relative to project root
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
NORMS_DIR = PROJECT_ROOT / 'norm_vals'
STATS_DIR = PROJECT_ROOT / 'stats'
LOGS_DIR = PROJECT_ROOT / 'logs'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
TESTS_DIR = PROJECT_ROOT / 'tests'

