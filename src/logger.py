import logging
import os
from datetime import datetime

# Create log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create logs directory inside current working directory
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Full path of the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# Test block
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Logging file is working")
    print(f"Log file created at: {LOG_FILE_PATH}")