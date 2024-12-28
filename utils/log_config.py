# log_config.py
import logging

def get_custom_logger(name):
    logger = logging.getLogger(name)
    
    # Prevent log propagation to the root logger
    logger.propagate = False
    
    # Check if the logger already has handlers to avoid adding them multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
