import pathlib

PROJECT_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = PROJECT_PATH / "data"
MODELS_PATH = PROJECT_PATH / "models"
REPORTS_PATH = PROJECT_PATH / "reports"
CONFIG_PATH = PROJECT_PATH / "resources" / "configs"

LOGGING_FORMAT = "[%(asctime)s] {%(filename)s:%(levelno)s} %(levelname)s - %(message)s"
