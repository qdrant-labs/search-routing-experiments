"""Thin caller for the v3 composition build — the class owns everything.

    poetry run python src/scripts/compose_v3.py [--force]
"""

from composition.composer import main

if __name__ == "__main__":
    main()
