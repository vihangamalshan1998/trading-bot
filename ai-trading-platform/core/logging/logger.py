import logging
import json
import sys
from datetime import datetime
import os
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON strings after parsing the LogRecord."""

    def __init__(self, fmt_dict: Dict[str, str] = None, time_format: str = "%Y-%m-%dT%H:%M:%S", msec_format: str = "%s.%03dZ"):
        self.fmt_dict = fmt_dict if fmt_dict is not None else {"message": "message"}
        self.default_time_format = time_format
        self.default_msec_format = msec_format
        self.datefmt = None

    def usesTime(self) -> bool:
        return "asctime" in self.fmt_dict.values()

    def formatMessage(self, record: logging.LogRecord) -> Dict[str, Any]:
        return {fmt_key: record.__dict__.get(fmt_val, fmt_val) for fmt_key, fmt_val in self.fmt_dict.items()}

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        message_dict = self.formatMessage(record)
        
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            message_dict["exc_info"] = record.exc_text

        if record.stack_info:
            message_dict["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(message_dict, default=str)


def setup_logger(name: str = "ai_trading", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        json_format = {
            "timestamp": "asctime",
            "level": "levelname",
            "name": "name",
            "message": "message",
            "module": "module",
            "funcName": "funcName",
            "line": "lineno"
        }
        
        formatter = JSONFormatter(fmt_dict=json_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Default logger instance
logger = setup_logger()

def set_log_file(filepath: str):
    """
    Attaches a rotating file handler to the global logger.
    Rotates at midnight, keeping 1 day of history.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Check if a file handler already exists to prevent duplicate logs
    for handler in logger.handlers:
        if isinstance(handler, TimedRotatingFileHandler):
            return
            
    file_handler = TimedRotatingFileHandler(
        filepath, 
        when="midnight", 
        interval=1, 
        backupCount=1
    )
    
    json_format = {
        "timestamp": "asctime",
        "level": "levelname",
        "name": "name",
        "message": "message",
        "module": "module",
        "funcName": "funcName",
        "line": "lineno"
    }
    formatter = JSONFormatter(fmt_dict=json_format)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)

