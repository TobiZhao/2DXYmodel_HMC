#===============================================================================
# Logging Settings
#===============================================================================

import logging

# Define a custom log level for real-time logging
REALTIME = 60
logging.addLevelName(REALTIME, "REALTIME")

def setup_logging(log_file):
    """
    Set up a custom logging configuration.

    This function creates a logger with two handlers:
    1. A file handler that logs all messages to a file.
    2. A console handler that logs all messages except REALTIME level to the console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger object.
    """
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                       datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter('%(message)s')

    # Create and configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(lambda record: record.levelno != REALTIME)

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Add custom realtime method to Logger class
    def realtime(self, message, *args, **kwargs):
        """Log a message with REALTIME level."""
        if self.isEnabledFor(REALTIME):
            self._log(REALTIME, message, args, **kwargs)

    logging.Logger.realtime = realtime

    return logger

def get_logger(name):
    """
    Get a named logger.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Named logger object.
    """
    return logging.getLogger(name)
