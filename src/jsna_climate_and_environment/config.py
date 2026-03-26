import re
from pathlib import Path
from loguru import logger

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

# make a home for all data outputs associated with the project
DATA_DIR = ROOT_DIR / "data"
if not DATA_DIR.exists():
    logger.info(f"data not found. creating datastore: {DATA_DIR}")
    DATA_DIR.mkdir()
