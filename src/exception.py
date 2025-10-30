import sys
from src.logger import logging

logger = logging.getLogger(__name__)

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number [{exc_tb.tb_lineno}] "
        f"with error message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        logger.error(self.error_message)  # Log the exception automatically

    def __str__(self):
        return self.error_message

# Test block
if __name__ == "__main__":
    try:
        a = 10 / 0  # Intentional error
    except Exception as e:
        raise CustomException(e, sys)
