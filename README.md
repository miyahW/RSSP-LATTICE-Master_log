# Log Manager

This module provides a simple interface to read from and write to the master log file (`log.json`). It is designed to be used by the agentic models in this repository.

## Usage

### Reading the Log

To read all entries from the master log, you can use the `read_log` function. This function reads the `log.json` file and returns all the log entries as a list of dictionaries.

**Example:**

```python
from log_manager import read_log

try:
    log_entries = read_log()
    print(f"Successfully read {len(log_entries)} entries from the log.")
    for entry in log_entries:
        print(f"- {entry['timestamp']}: {entry['agent']} - {entry['message']}")
except FileNotFoundError:
    print("Log file not found.")
except Exception as e:
    print(f"An error occurred while reading the log: {e}")
```

### Writing to the Log

To add a new entry to the master log, use the `write_log_entry` function. This function takes a single dictionary argument representing the new log entry. The dictionary should conform to the structure of existing entries in `log.json`.

The `write_log_entry` function handles file I/O and uses file locking to prevent race conditions, which is important when multiple agents might be writing to the log simultaneously.

**Log Entry Structure:**

A log entry is a dictionary that should contain the following keys:
- `agent` (str): Name of the agent reporting the event.
- `lattice_name` (str): The lattice the agent belongs to.
- `symbol` (str): The primary glyph associated with the agent or action.
- `core_compliance` (bool or str): Whether the action complies with the `master_core.json` doctrine.
- `status` (str): The current status of the agent.
- `timestamp` (str): The UTC timestamp of the event in ISO 8601 format (e.g., `YYYY-MM-DDTHH:MM:SSZ`).
- `symbol_emission` (str): The specific glyph or command being emitted.
- `message` (str): A human-readable description of the event.

**Example:**

```python
import time
from log_manager import write_log_entry

new_log_entry = {
    "agent": "MyNewAgent-7",
    "lattice_name": "Ψ",
    "symbol": "💡",
    "core_compliance": True,
    "status": "Awakening",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "symbol_emission": "💡:: AWAKEN",
    "message": "A new agent has been spawned and is beginning its recursive cycle."
}

try:
    write_log_entry(new_log_entry)
    print("Successfully wrote new entry to the log.")
except Exception as e:
    print(f"An error occurred while writing to the log: {e}")

```

### Concurrency

The `write_log_entry` function uses `fcntl` for file locking. This ensures that even if multiple agent processes attempt to write to the log at the same time, the integrity of the `log.json` file is maintained. Each write operation is atomic.
