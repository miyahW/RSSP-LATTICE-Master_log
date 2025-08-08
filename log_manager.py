# This file will contain functions for reading and writing to the master log.
import json
import fcntl
import time

LOG_FILE = 'log.json'

def read_log():
    """
    Reads the master log file.

    Returns:
        list: A list of log entries.
    """
    with open(LOG_FILE, 'r') as f:
        return json.load(f).get('log', [])

def write_log_entry(entry):
    """
    Writes a new entry to the master log file.

    Args:
        entry (dict): The log entry to add.
    """
    with open(LOG_FILE, 'r+') as f:
        # Lock the file to prevent race conditions
        fcntl.flock(f, fcntl.LOCK_EX)

        try:
            data = json.load(f)
            if 'log' not in data:
                data['log'] = []

            data['log'].append(entry)

            # Move the file pointer to the beginning and truncate the file
            f.seek(0)
            f.truncate()

            json.dump(data, f, indent=2)
        finally:
            # Unlock the file
            fcntl.flock(f, fcntl.LOCK_UN)
