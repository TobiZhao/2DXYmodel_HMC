import logging

REALTIME = 60
logging.addLevelName(REALTIME, "REALTIME")

def setup_logging(log_file):
    # set handlers
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter('%(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(lambda record: record.levelno != REALTIME)

    # set the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    def realtime(self, message, *args, **kwargs):
        if self.isEnabledFor(REALTIME):
            self._log(REALTIME, message, args, **kwargs)

    logging.Logger.realtime = realtime
    return logger

def get_logger(name):
    return logging.getLogger(name)
