import os
import sys

# `main` is defined in run.py, not here; make Hydra resolve config_path
# as if it were, so it uses a filesystem path instead of a module import.
os.environ.setdefault("HYDRA_MAIN_MODULE", "__main__")

from tracking.curate.run import main  # noqa: E402

sys.exit(main())
