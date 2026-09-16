#!/usr/bin/env python
"""Validate a training root and name every violation.

    python scripts/check_training_root.py /path/to/root [--masks /path/to/root_masks]

Exit code 0 when there are no errors, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys

from tracking.curate.validate import format_findings, validate_root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--masks", default=None)
    args = ap.parse_args()
    findings = validate_root(args.root, masks_root=args.masks)
    print(format_findings(findings))
    return 1 if any(f.level == "error" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
