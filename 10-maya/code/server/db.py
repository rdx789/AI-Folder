"""Read-only operational lookups using Maya's own data/database files."""

import functools
import sqlite3
from ..paths import DATA_PATH

DB_DIR = DATA_PATH / "database"

# The graph host must authorize employee-scoped operational reads.
EMPLOYEE_FIELDS = (
    "employee_id, full_name, email, role, department_id, "
    "manager_id, location, employment_type, status"
)


@functools.lru_cache(maxsize=1)
def _conn() -> sqlite3.Connection:
    """Build the read-only tools' in-memory DB from local schema and seed SQL."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript((DB_DIR / "schema.sql").read_text(encoding="utf-8"))
    conn.executescript((DB_DIR / "seed.sql").read_text(encoding="utf-8"))
    return conn


def get_employee(query: str) -> list[dict]:
    """Look up NovaOps employees by id, email, or (partial) name.

    Returns EVERY match — the seed data intentionally has duplicate first names
    (two Daniels, two Mayas, two Saras), so a name search can return more than one
    person, which the agent then has to disambiguate.
    """
    like = f"%{query}%"
    rows = _conn().execute(
        f"SELECT {EMPLOYEE_FIELDS} FROM employees "
        "WHERE employee_id = ? OR lower(email) = lower(?) OR lower(full_name) LIKE lower(?) "
        "ORDER BY full_name",
        (query, query, like),
    ).fetchall()
    return [dict(row) for row in rows]


def check_software_subscription(software: str) -> list[dict]:
    """Look up a SaaS subscription's seat usage by software or vendor name.

    Resolves the name against both the system catalog and the vendor list, so
    'Webex', 'webex', or a partial match all work. Returns seat_limit against the
    seats currently in use, plus a computed `over_limit` flag — the signal an
    access workflow reads to decide whether a new seat needs approval.

    `annual_cost` is included because approval thresholds are stated in money, not
    seats. A tool that returns only what its immediate caller needs quietly forces
    the model to guess at the rest, and it will guess confidently.
    """
    like = f"%{software}%"
    rows = _conn().execute(
        "SELECT sys.system_name, v.vendor_name, sub.seat_limit, sub.active_seats, "
        "       sub.status, sub.renewal_date, sub.annual_cost "
        "FROM software_subscriptions sub "
        "JOIN systems sys ON sub.system_id = sys.system_id "
        "JOIN vendors  v  ON sub.vendor_id = v.vendor_id "
        "WHERE lower(sys.system_name) LIKE lower(?) OR lower(v.vendor_name) LIKE lower(?) "
        "ORDER BY sys.system_name",
        (like, like),
    ).fetchall()
    results = []
    for row in rows:
        sub = dict(row)
        sub["seats_available"] = sub["seat_limit"] - sub["active_seats"]
        sub["over_limit"] = sub["active_seats"] >= sub["seat_limit"]
        results.append(sub)
    return results


def list_onboarding_tasks(employee_id: str) -> list[dict]:
    """Return one employee's onboarding checklist, earliest due date first.

    The checklist is the system of record for onboarding: a task is done only when
    this table says so. Statuses that matter to the workflow are `blocked` (something
    is in the way) and `planned`/`pending` (still open).
    """
    rows = _conn().execute(
        "SELECT task_id, task_type, description, owner_group, status, due_date "
        "FROM onboarding_tasks WHERE employee_id = ? ORDER BY due_date, task_id",
        (employee_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def check_asset_inventory(asset_type: str = "", location: str = "") -> list[dict]:
    """Return hardware assets, optionally narrowed to a type and/or a location.

    Both filters are optional substring matches, so 'Laptop' / 'israel' / '' all work.
    `status` is the operative field: 'available' means it can be assigned now.
    """
    rows = _conn().execute(
        "SELECT asset_id, asset_type, model, status, location, employee_id, notes "
        "FROM assets "
        "WHERE lower(asset_type) LIKE lower(?) AND lower(location) LIKE lower(?) "
        "ORDER BY asset_type, asset_id",
        (f"%{asset_type}%", f"%{location}%"),
    ).fetchall()
    return [dict(row) for row in rows]


def list_employee_tickets(employee_id: str, status: str = "") -> list[dict]:
    """Return an employee's IT tickets, newest first; optionally filter by status.

    Used to answer 'what's already been reported?' — and, in this lesson, to show
    how a *resolved* earlier issue keeps polluting the model's context if nothing
    ever drops it.
    """
    rows = _conn().execute(
        "SELECT ticket_id, category, subcategory, priority, status, subject, "
        "       created_at, assigned_team "
        "FROM tickets WHERE employee_id = ? AND lower(status) LIKE lower(?) "
        "ORDER BY created_at DESC",
        (employee_id, f"%{status}%"),
    ).fetchall()
    return [dict(row) for row in rows]

