from pathlib import Path

TA_HOME = Path.home() / ".ta-agents"
TA_HOME.mkdir(exist_ok=True)

USER_MD = TA_HOME / "USER.md"
MEMORY_MD = TA_HOME / "MEMORY.md"
DB_PATH = TA_HOME / "memory.db"
