import logging
import json
import sys
from datetime import datetime
import os
import time
import glob
from logging.handlers import RotatingFileHandler
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

def clean_old_logs(log_dir: str = "logs", max_size_mb: int = 20, max_age_days: int = 3):
    """
    Prunes stale or oversized log files to prevent VPS disk exhaustion.
    Truncates files > max_size_mb down to last 2,000 lines and removes backups older than max_age_days.
    """
    try:
        if not os.path.exists(log_dir):
            return
        now_ts = time.time()
        max_bytes = max_size_mb * 1024 * 1024
        max_age_sec = max_age_days * 86400

        for fpath in glob.glob(os.path.join(log_dir, "*")):
            if not os.path.isfile(fpath):
                continue
            mtime = os.path.getmtime(fpath)
            fsize = os.path.getsize(fpath)
            
            # Remove stale backup logs older than max_age_days (e.g. .log.1, .log.2026-...)
            if (now_ts - mtime) > max_age_sec and (".log." in fpath or fpath.endswith(".old")):
                try:
                    os.remove(fpath)
                except Exception:
                    pass
            # If any active log file exceeds max_size_mb, truncate down to last 2,000 lines immediately
            elif fsize > max_bytes:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.writelines(lines[-2000:])
                except Exception:
                    pass
    except Exception:
        pass

def set_log_file(filepath: str, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 2):
    """
    Attaches a size-bounded rotating file handler to the global logger.
    Strictly caps log files at max_bytes (default 10MB) with max 2 backups (20MB total max disk space),
    permanently preventing VPS hard drives from filling up.
    """
    log_dir = os.path.dirname(filepath)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        clean_old_logs(log_dir)
    
    # Check if a file handler already exists to prevent duplicate logs
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            return
            
    file_handler = RotatingFileHandler(
        filepath, 
        maxBytes=max_bytes, 
        backupCount=backup_count,
        encoding="utf-8"
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

