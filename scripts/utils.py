"""
Utilities to run processing steps in loops with error handling.
"""

import logging
import traceback
from functools import wraps
from pathlib import Path

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("pyxu").setLevel(logging.WARNING)

def loop(subjects):
    """Allows to loop processing step through a list of subjects."""
    def decorator(func):
        """Decorator for processing function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper for looping"""
            for sub in subjects:
                try:
                    func(sub, *args, **kwargs)
                    logging.info("SUCCESS: sub-%s", sub)

                except Exception as _:
                    logging.error("FAILED: sub-%s", sub)
                    logging.error(traceback.format_exc())
        return wrapper
    return decorator


def check_files(files: list[Path]):
    """Check if files exist. Returns 1 at least 1 file does."""

    for f in files:
        # exception (stc files)
        if f.name.endswith("stc"):
            f = f.parent/f"{f.name}-lh.stc"
        if f.exists():
            logging.info("Following file already exists: %s", str(f))
            return True
        
        # create parent directory if doesn't exist
        f.parent.mkdir(parents=True, exist_ok=True)
    
    return False
