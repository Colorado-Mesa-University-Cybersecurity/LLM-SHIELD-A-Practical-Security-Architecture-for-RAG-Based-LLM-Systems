import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.prompt_guard import PromptGuard  # noqa: F401, E402
