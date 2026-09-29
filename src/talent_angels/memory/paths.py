from pathlib import Path

TA_HOME = Path.home() / ".ta-agents"
TA_HOME.mkdir(exist_ok=True)

USER_MD = TA_HOME / "USER.md"
MEMORY_MD = TA_HOME / "MEMORY.md"
DB_PATH = TA_HOME / "memory.db"

#: LangGraph's per-thread checkpoint state, written by ``assistant.graph`` when a
#: caller passes a ``thread_id``. Personal data on the same terms as the rest of
#: this home — it holds the question, the plan and the cited nodes of every turn
#: on the thread — so it lives under ``TA_HOME`` where ``erase_all`` sweeps and
#: where test isolation redirects, rather than in whatever cwd the API happens
#: to be served from.
CHECKPOINT_DB_PATH = TA_HOME / "checkpoints.db"
