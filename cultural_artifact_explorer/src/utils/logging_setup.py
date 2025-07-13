# src/utils/logging_setup.py
# Utility for setting up standardized logging across the project.
# Renamed from 'logging.py' to avoid conflict with Python's built-in logging module.

import logging
import sys
# import os # If logging to files with dynamic paths

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Store configured loggers to prevent duplicate handlers if called multiple times
_configured_loggers = {}

def setup_logger(name='cultural_artifact_explorer', level=logging.INFO,
                 log_format=DEFAULT_LOG_FORMAT, date_format=DEFAULT_DATE_FORMAT,
                 log_to_console=True, log_file_path=None, file_log_level=logging.DEBUG):
    """
    Sets up a logger with specified configuration.

    Args:
        name (str): Name of the logger. Typically __name__ of the calling module or a project-wide name.
        level (int): Logging level for console output (e.g., logging.INFO, logging.DEBUG).
        log_format (str): Format string for log messages.
        date_format (str): Format string for dates in log messages.
        log_to_console (bool): Whether to output logs to the console (stderr).
        log_file_path (str, optional): Path to a file for logging. If None, no file logging.
        file_log_level (int): Logging level for file output (if log_file_path is provided).

    Returns:
        logging.Logger: Configured logger instance.
    """
    global _configured_loggers

    if name in _configured_loggers and _configured_loggers[name].get('handlers_set', False):
        # print(f"Logger '{name}' already configured. Returning existing instance.")
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(min(level, file_log_level) if log_file_path else level) # Set logger to the lower of console/file
    logger.propagate = False # Prevent root logger from handling messages again if it's configured

    formatter = logging.Formatter(log_format, datefmt=date_format)

    handlers_added = False

    # Console Handler
    if log_to_console:
        # Check if a similar console handler already exists for this logger
        has_console_handler = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stderr for h in logger.handlers)
        if not has_console_handler:
            console_handler = logging.StreamHandler(sys.stderr) # Use stderr for logs, stdout for actual program output
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            handlers_added = True
            # print(f"  Added Console Handler to logger '{name}'.")

    # File Handler
    if log_file_path:
        # Check if a similar file handler already exists
        has_file_handler = any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path for h in logger.handlers)
        if not has_file_handler:
            # try:
            #     # os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # Ensure log directory exists
            #     file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8') # Append mode
            #     file_handler.setLevel(file_log_level)
            #     file_handler.setFormatter(formatter)
            #     logger.addHandler(file_handler)
            #     handlers_added = True
            #     print(f"  Added File Handler ({log_file_path}) to logger '{name}'.")
            # except Exception as e:
            #     logger.error(f"Failed to create file handler for {log_file_path}: {e}", exc_info=False)
            #     # Fallback to console if file logging fails but console is enabled
            #     if not log_to_console: # If console wasn't primary, add it as fallback
            #         print(f"Fallback: Adding console handler due to file handler error for logger '{name}'.")
            #         console_handler = logging.StreamHandler(sys.stderr)
            #         console_handler.setLevel(level) # Use the general level
            #         console_handler.setFormatter(formatter)
            #         logger.addHandler(console_handler)
            #         handlers_added = True
            # Placeholder for file handling:
            print(f"  Placeholder: File Handler for '{log_file_path}' would be added to logger '{name}'.")
            # To simulate it being added for the _configured_loggers check:
            if not has_console_handler and not log_to_console: # If no console handler was intended/added
                 logger.addHandler(logging.NullHandler()) # Add a null handler to mark it as configured
            handlers_added = True


    if handlers_added or logger.hasHandlers(): # Check if any handlers were actually added or already existed
        _configured_loggers[name] = {'handlers_set': True}

    if not handlers_added and not logger.hasHandlers():
        # If no handlers were configured (e.g. console=False, file_path=None or error)
        # add a NullHandler to prevent "No handlers found" warnings for this logger.
        logger.addHandler(logging.NullHandler())
        # print(f"  Added Null Handler to logger '{name}' as no other handlers were specified/added.")
        _configured_loggers[name] = {'handlers_set': True} # Mark as "configured" to avoid re-entry


    return logger

if __name__ == '__main__':
    print("Testing logging_setup utility (placeholders)...")

    # --- Test 1: Basic console logger ---
    print("\n--- Test 1: Basic Console Logger (INFO) ---")
    logger1 = setup_logger("my_app_console", level=logging.INFO)
    logger1.debug("This is a DEBUG message (console - should not appear).")
    logger1.info("This is an INFO message (console - should appear).")
    logger1.warning("This is a WARNING message (console - should appear).")

    # --- Test 2: Logger with different name and level (DEBUG) ---
    print("\n--- Test 2: Another Console Logger (DEBUG) ---")
    logger2 = setup_logger("module_xyz", level=logging.DEBUG)
    logger2.debug("This is a DEBUG message from module_xyz (should appear).")
    logger2.info("This is an INFO message from module_xyz (should appear).")

    # --- Test 3: File logger (placeholder for file creation) ---
    print("\n--- Test 3: File Logger (DEBUG to file, INFO to console - placeholder file) ---")
    # dummy_log_file = "temp_app_log.log"
    # logger3 = setup_logger(
    #     "file_app",
    #     level=logging.INFO, # Console level
    #     log_file_path=dummy_log_file,
    #     file_log_level=logging.DEBUG # File level
    # )
    # logger3.debug(f"This DEBUG message should go to '{dummy_log_file}' (placeholder).")
    # logger3.info(f"This INFO message should go to console and '{dummy_log_file}' (placeholder).")
    # logger3.error("This ERROR message also to both (placeholder).")

    # # Placeholder check for file content (in real scenario, read the file)
    # print(f"  (Placeholder check: In a real test, verify content of '{dummy_log_file}')")
    # # if os.path.exists(dummy_log_file): os.remove(dummy_log_file)

    # --- Test 4: Re-getting a configured logger ---
    print("\n--- Test 4: Re-getting a Configured Logger ---")
    logger1_again = setup_logger("my_app_console") # Get existing logger1
    # logger1_again.info("This INFO message is from logger1_again (should use existing handlers).")
    # Check if it's the same object and doesn't add duplicate handlers (visual check of console output)
    # assert logger1 is logger1_again, "Re-getting logger did not return the same object."
    # initial_handler_count = len(logger1.handlers)
    # logger1_again_setup = setup_logger("my_app_console")
    # assert len(logger1_again_setup.handlers) == initial_handler_count, "Duplicate handlers added."


    # --- Test 5: Logger with only file output (console=False - placeholder file) ---
    print("\n--- Test 5: File-Only Logger (placeholder file) ---")
    # dummy_file_only_log = "temp_file_only.log"
    # logger_file_only = setup_logger(
    #     "file_only_logger",
    #     log_to_console=False,
    #     log_file_path=dummy_file_only_log,
    #     file_log_level=logging.INFO
    # )
    # logger_file_only.info(f"This INFO message should only be in '{dummy_file_only_log}' (placeholder).")
    # logger_file_only.warning("This WARNING too (placeholder).")
    # print(f"  (Placeholder check: Verify console for no output, and '{dummy_file_only_log}' for content)")
    # # if os.path.exists(dummy_file_only_log): os.remove(dummy_file_only_log)

    # --- Test 6: Logger with no handlers specified (should get NullHandler) ---
    print("\n--- Test 6: Logger with No Handlers Specified ---")
    # logger_no_handlers = setup_logger("null_handler_logger", log_to_console=False, log_file_path=None)
    # logger_no_handlers.info("This message should not appear anywhere (goes to NullHandler).")
    # assert isinstance(logger_no_handlers.handlers[0], logging.NullHandler), "NullHandler not added when no other handlers."

    print("\nLogging setup tests complete (placeholders). Check console output for expected messages.")
