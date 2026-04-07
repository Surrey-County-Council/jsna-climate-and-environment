import os
import re
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv


def get_env_path(env_var: str, default: Path) -> Path:
    load_dotenv()
    try:
        path = Path(os.environ[env_var])
    except KeyError:
        path = default
    if not path.exists():
        logger.info(f"creating directory for '{env_var}': {path}")
        path.mkdir(parents=True)
    return path


PROJECT_NAME = "jsna-climate-and-environment"
# note the convention of hyphens for project and underscores for modules helps get a consistent root

bad_chars = re.sub(PROJECT_NAME, r"[a-zA-Z\-]+", "")
if bad_chars:
    raise ValueError(
        f"Bad project configuration! {PROJECT_NAME} contains forbidden characters: '{bad_chars}'"
    )

# find the root from anywhere the code is run
ROOT_DIR = next(p for p in Path(__file__).parents if p.name == PROJECT_NAME)

# ensure the project layout is as expected
MODULE_DIR = ROOT_DIR / "src" / PROJECT_NAME.replace("-", "_")

if not MODULE_DIR.exists():
    raise FileNotFoundError(f"Bad project configuration! {MODULE_DIR} does not exist")

logger.info(f"ROOT_DIR: {ROOT_DIR}")

# make a home for all data inputs associated with the project
DATA_DIR = get_env_path("DATA_DIR", ROOT_DIR / "data" / "input")

# make a home for all data outputs associated with the project
OUTPUT_DIR = get_env_path("OUTPUT_PATH", ROOT_DIR / "data" / "output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
