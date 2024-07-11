import logging  # Import the logging module from Python's standard library

def get_logger() -> logging.Logger:
    """Returns a logger

    Returns:
        logging.Logger: _description_
    """
    # Create or get a logger instance named 'dataflow'
    logger = logging.getLogger('dataflow')
    
    # Set the logging level to INFO, meaning the logger will handle all messages
    # at this level or higher (e.g., INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.INFO)
    
    # Return the configured logger instance
    return logger
