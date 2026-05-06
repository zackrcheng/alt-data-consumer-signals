"""
wrds_utils.py — shared WRDS connection helpers.

Extracted from the three pull scripts (pull_wrds_compustat, pull_wrds_ibes,
pull_crsp) that all need an authenticated WRDS connection. Tries the password
cached in ~/.pgpass first (non-interactive); falls back to wrds.Connection's
interactive prompt, which requires a TTY (run with `! python -m src.pull_…`
in Claude Code to get one).
"""

import os

WRDS_HOSTNAME = "wrds-pgdata.wharton.upenn.edu"


def read_pgpass_password(hostname: str = WRDS_HOSTNAME,
                          username: str | None = None) -> str | None:
    """Parse ~/.pgpass for a matching entry and return the password.

    The .pgpass format is `host:port:database:user:password` per row, where
    each field can also be `*` (wildcard). Returns None if no match.
    """
    pgpass = os.path.expanduser("~/.pgpass")
    if not os.path.exists(pgpass):
        return None
    if username is None:
        username = os.getenv("WRDS_USERNAME")
    with open(pgpass) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 5:
                continue
            h, _port, _db, u, pw = parts[0], parts[1], parts[2], parts[3], ":".join(parts[4:])
            if (h in (hostname, "*")) and (u in (username, "*")):
                return pw
    return None


def get_wrds_connection():
    """Establish an authenticated WRDS connection.

    Order of attempts:
      1. ~/.pgpass cached password (non-interactive)
      2. wrds.Connection's interactive prompt (TTY required)

    Raises:
      ImportError if `wrds` not installed.
      EnvironmentError if WRDS_USERNAME not set in .env.
    """
    try:
        import wrds
    except ImportError:
        raise ImportError("wrds not installed. Run: pip install wrds")

    username = os.getenv("WRDS_USERNAME")
    if not username:
        raise EnvironmentError(
            "WRDS_USERNAME not set. Copy .env.template to .env and add WRDS_USERNAME."
        )

    password = read_pgpass_password(WRDS_HOSTNAME, username)
    if password:
        try:
            return wrds.Connection(wrds_username=username, wrds_password=password)
        except Exception as e:
            print(f"  pgpass credentials failed ({e}). Falling back to interactive login...")

    return wrds.Connection(wrds_username=username)
