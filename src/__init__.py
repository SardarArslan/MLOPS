import logging
import sys
import os

# 1. Create logs directory if it doesn't exist
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "running_logs.log")

# 2. Define the format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 3. Configure the Root Logger
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),     # Dumps to logs/running_logs.log
        logging.StreamHandler(sys.stdout)  # Prints to terminal
    ]
)

logger = logging.getLogger(__name__)