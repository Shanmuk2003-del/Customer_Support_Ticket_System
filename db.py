"""
db.py

SQLite database storage for the Financial Customer Complaint Classification
and Routing System.

This module uses Python's built-in `sqlite3` library - no additional packages
are required.

It provides three functions:
  - init_db()            Create the complaints table if it does not exist.
  - insert_complaint()   Insert one processed complaint and return its ticket ID.
  - get_recent_complaints()  Return the most recent complaints, newest first.

The database is stored at: data/complaints.db
The data directory is created automatically if it does not exist.
"""

import os
import sqlite3
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Database path (relative to the project root)
# ---------------------------------------------------------------------------
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "complaints.db")

# SQL used to create the complaints table.
#   id                  : auto-increment integer primary key (the ticket ID)
#   complaint_text      : the raw complaint description
#   predicted_category  : the ML-predicted product category
#   confidence_estimate : the model's confidence score (0-1)
#   department          : the department the complaint was routed to
#   priority            : High / Medium / Low
#   priority_reason     : the explanation of why the priority was assigned
#   created_at          : UTC timestamp of when the complaint was processed
#   status              : current ticket status (default 'Open')
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS complaints (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_text      TEXT NOT NULL,
    predicted_category  TEXT NOT NULL,
    confidence_estimate REAL,
    department          TEXT NOT NULL,
    priority            TEXT NOT NULL,
    priority_reason     TEXT,
    created_at          TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'Open'
)
"""


def init_db():
    """
    Create the complaints table if it does not exist.

    This function is idempotent: calling it multiple times is safe and will
    not duplicate the table or lose existing data.

    Returns
    -------
    None
    """
    # Create the data directory automatically if it does not exist.
    os.makedirs(DB_DIR, exist_ok=True)

    # Connect to the database (creates the file if it does not exist).
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def insert_complaint(result):
    """
    Insert one processed complaint into the database.

    Accepts the dictionary returned by process_complaint() and stores its
    fields in the complaints table. The created_at timestamp is set to the
    current UTC time, and the status is set to 'Open'.

    Parameters
    ----------
    result : dict
        A dictionary with at least these keys (as returned by
        process_complaint()):
          - complaint_text
          - predicted_category
          - confidence_estimate
          - department
          - priority
          - priority_reason

    Returns
    -------
    int
        The auto-generated ticket ID (primary key) of the newly inserted
        complaint.

    Raises
    ------
    ValueError
        If the result dictionary is missing required keys or contains
        invalid values.
    sqlite3.Error
        If a database error occurs during the insert.
    """
    # Validate that the result dictionary contains the required keys.
    required_keys = [
        "complaint_text",
        "predicted_category",
        "confidence_estimate",
        "department",
        "priority",
        "priority_reason",
    ]
    for key in required_keys:
        if key not in result:
            raise ValueError(
                f"Missing required key '{key}' in result dictionary."
            )

    # Ensure the table exists before inserting.
    init_db()

    # Current UTC timestamp in ISO-8601 format.
    created_at = datetime.now(timezone.utc).isoformat()

    # Parameterized SQL statement to prevent SQL injection.
    insert_sql = """
    INSERT INTO complaints
        (complaint_text, predicted_category, confidence_estimate,
         department, priority, priority_reason, created_at, status)
    VALUES
        (?, ?, ?, ?, ?, ?, ?, 'Open')
    """

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            insert_sql,
            (
                result["complaint_text"],
                result["predicted_category"],
                result["confidence_estimate"],
                result["department"],
                result["priority"],
                result["priority_reason"],
                created_at,
            ),
        )
        conn.commit()
        # cursor.lastrowid is the auto-generated primary key of the new row.
        ticket_id = cursor.lastrowid
    finally:
        conn.close()

    return ticket_id


def get_recent_complaints(limit=10):
    """
    Return the most recent complaints, newest first.

    Parameters
    ----------
    limit : int, optional
        Maximum number of complaints to return (default is 10).

    Returns
    -------
    list of dict
        A list of dictionaries, one per complaint row. Each dictionary has
        the keys: id, complaint_text, predicted_category, confidence_estimate,
        department, priority, priority_reason, created_at, status.
        The list is ordered by created_at descending (newest first).
    """
    # Ensure the table exists before querying.
    init_db()

    select_sql = """
    SELECT
        id,
        complaint_text,
        predicted_category,
        confidence_estimate,
        department,
        priority,
        priority_reason,
        created_at,
        status
    FROM complaints
    ORDER BY id DESC
    LIMIT ?
    """

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows accessing columns by name
    try:
        cursor = conn.cursor()
        cursor.execute(select_sql, (limit,))
        rows = cursor.fetchall()
        # Convert each sqlite3.Row into a plain dict.
        results = [dict(row) for row in rows]
    finally:
        conn.close()

    return results


# ---------------------------------------------------------------------------
# Allowed ticket statuses (business rule)
# ---------------------------------------------------------------------------
VALID_STATUSES = ("Open", "In Progress", "Resolved")


def update_ticket_status(ticket_id, new_status):
    """
    Update the status of an existing ticket.

    Only the three permitted statuses are allowed:
      - "Open"
      - "In Progress"
      - "Resolved"

    No other columns are modified and no records are deleted.

    Parameters
    ----------
    ticket_id : int
        The primary key (ticket ID) of the complaint to update.
    new_status : str
        The new status value. Must be one of VALID_STATUSES.

    Returns
    -------
    bool
        True if a ticket with the supplied ID was found and updated,
        False if no ticket matched the ID (or the status was invalid).

    Raises
    ------
    ValueError
        If new_status is not one of the permitted values.
    sqlite3.Error
        If a database error occurs during the update.
    """
    # Validate the status before doing anything else.
    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{new_status}'. "
            f"Allowed values: {', '.join(VALID_STATUSES)}"
        )

    # Validate the ticket ID type.
    if not isinstance(ticket_id, int) or ticket_id <= 0:
        raise ValueError(
            f"Invalid ticket ID: {ticket_id!r}. Must be a positive integer."
        )

    # Ensure the table exists before updating.
    init_db()

    # Parameterized UPDATE statement. Only the status column is touched.
    update_sql = """
    UPDATE complaints
    SET status = ?
    WHERE id = ?
    """

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(update_sql, (new_status, ticket_id))
        conn.commit()
        # cursor.rowcount is the number of rows modified.
        # 0 means no ticket matched the supplied ID.
        updated = cursor.rowcount > 0
    finally:
        conn.close()

    return updated


# ---------------------------------------------------------------------------
# Filtered complaint search
# ---------------------------------------------------------------------------
def get_filtered_complaints(
    ticket_id=None,
    complaint_text=None,
    predicted_category=None,
    department=None,
    priority=None,
    status=None,
    limit=100,
):
    """
    Return complaints matching the supplied filters, newest first.

    All filters are optional. When a filter is None (or an empty string) it is
    ignored, so multiple filters can be combined. Partial text matching is
    used for the complaint description (case-insensitive LIKE).

    Parameters
    ----------
    ticket_id : int or None
        Exact match on the ticket primary key.
    complaint_text : str or None
        Partial, case-insensitive match on the complaint description.
    predicted_category : str or None
        Exact match on the predicted product category.
    department : str or None
        Exact match on the department.
    priority : str or None
        Exact match on the priority (High / Medium / Low).
    status : str or None
        Exact match on the status (Open / In Progress / Resolved).
    limit : int
        Maximum number of rows to return (default 100).

    Returns
    -------
    list of dict
        A list of dictionaries, one per matching complaint row. Each
        dictionary has the keys: id, complaint_text, predicted_category,
        confidence_estimate, department, priority, priority_reason,
        created_at, status. Ordered by id DESC (newest first).
    """
    # Ensure the table exists before querying.
    init_db()

    # Build the WHERE clause dynamically. Each clause uses a parameter
    # placeholder (?) so the values are bound safely.
    where_clauses = []
    params = []

    if ticket_id is not None and ticket_id != "":
        where_clauses.append("id = ?")
        params.append(int(ticket_id))

    if complaint_text is not None and complaint_text.strip() != "":
        # LIKE is case-insensitive for SQLite text columns.
        where_clauses.append("complaint_text LIKE ?")
        params.append(f"%{complaint_text.strip()}%")

    if predicted_category is not None and predicted_category.strip() != "":
        where_clauses.append("predicted_category = ?")
        params.append(predicted_category.strip())

    if department is not None and department.strip() != "":
        where_clauses.append("department = ?")
        params.append(department.strip())

    if priority is not None and priority.strip() != "":
        where_clauses.append("priority = ?")
        params.append(priority.strip())

    if status is not None and status.strip() != "":
        where_clauses.append("status = ?")
        params.append(status.strip())

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    select_sql = f"""
    SELECT
        id,
        complaint_text,
        predicted_category,
        confidence_estimate,
        department,
        priority,
        priority_reason,
        created_at,
        status
    FROM complaints
    {where_sql}
    ORDER BY id DESC
    LIMIT ?
    """
    params.append(int(limit))

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute(select_sql, params)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
    finally:
        conn.close()

    return results


# ---------------------------------------------------------------------------
# Dashboard summary (SQL aggregation over the full table)
# ---------------------------------------------------------------------------
def get_dashboard_summary():
    """
    Return aggregate statistics for the dashboard.

    All counts are computed directly in SQLite using COUNT / GROUP BY over
    the ENTIRE complaints table - not just the latest 50 or 100 rows. The
    function handles an empty database safely by returning zero counts.

    Returns
    -------
    dict
        A dictionary with the following keys:
          - total_tickets : int
          - open_tickets  : int
          - in_progress_tickets : int
          - resolved_tickets   : int
          - by_category   : list of (category, count) tuples, sorted by count desc
          - by_department : list of (department, count) tuples, sorted by count desc
          - by_priority   : list of (priority, count) tuples, sorted by count desc
          - daily_trend   : list of (date, count) tuples for the last 30 days,
                            sorted by date ascending
    """
    # Ensure the table exists before querying.
    init_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()

        # --- Total tickets ---
        cursor.execute("SELECT COUNT(*) AS total FROM complaints")
        total_tickets = int(cursor.fetchone()["total"])

        # --- Status counts (always include all three statuses, even if zero) ---
        open_tickets = 0
        in_progress_tickets = 0
        resolved_tickets = 0
        cursor.execute(
            "SELECT status, COUNT(*) AS count FROM complaints GROUP BY status"
        )
        for row in cursor.fetchall():
            status = row["status"]
            count = int(row["count"])
            if status == "Open":
                open_tickets = count
            elif status == "In Progress":
                in_progress_tickets = count
            elif status == "Resolved":
                resolved_tickets = count

        # --- Group by predicted category ---
        cursor.execute(
            "SELECT predicted_category, COUNT(*) AS count "
            "FROM complaints GROUP BY predicted_category "
            "ORDER BY count DESC, predicted_category ASC"
        )
        by_category = [(row["predicted_category"], int(row["count"]))
                       for row in cursor.fetchall()]

        # --- Group by department ---
        cursor.execute(
            "SELECT department, COUNT(*) AS count "
            "FROM complaints GROUP BY department "
            "ORDER BY count DESC, department ASC"
        )
        by_department = [(row["department"], int(row["count"]))
                         for row in cursor.fetchall()]

        # --- Group by priority ---
        cursor.execute(
            "SELECT priority, COUNT(*) AS count "
            "FROM complaints GROUP BY priority "
            "ORDER BY count DESC, priority ASC"
        )
        by_priority = [(row["priority"], int(row["count"]))
                       for row in cursor.fetchall()]

        # --- Daily creation trend for the last 30 days ---
        # created_at is stored as an ISO-8601 timestamp string.
        # SQLite date() extracts the date portion; we group by that date.
        cursor.execute(
            "SELECT date(created_at) AS day, COUNT(*) AS count "
            "FROM complaints "
            "WHERE date(created_at) >= date('now', '-30 days') "
            "GROUP BY day "
            "ORDER BY day ASC"
        )
        daily_trend = [(row["day"], int(row["count"]))
                       for row in cursor.fetchall()]

    finally:
        conn.close()

    return {
        "total_tickets": total_tickets,
        "open_tickets": open_tickets,
        "in_progress_tickets": in_progress_tickets,
        "resolved_tickets": resolved_tickets,
        "by_category": by_category,
        "by_department": by_department,
        "by_priority": by_priority,
        "daily_trend": daily_trend,
    }


# ---------------------------------------------------------------------------
# Standalone test (run this file directly to verify it works)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing db.py...")

    # 1. Initialize the database (creates data/complaints.db and the table).
    init_db()
    print(f"Database initialized at: {os.path.abspath(DB_PATH)}")

    # 2. Insert a sample complaint using the same dict structure as
    #    process_complaint() returns.
    sample_result = {
        "complaint_text": "I noticed an unauthorized transaction on my credit card.",
        "predicted_category": "Credit card",
        "confidence_estimate": 0.6771,
        "department": "Cards and Payments",
        "priority": "High",
        "priority_reason": "Matched high-priority keyword: unauthorized transaction",
    }

    ticket_id = insert_complaint(sample_result)
    print(f"Inserted complaint with ticket ID: {ticket_id}")

    # 3. Retrieve the most recent complaints.
    recent = get_recent_complaints(limit=5)
    print(f"\nMost recent complaints ({len(recent)}):")
    for row in recent:
        print(
            f"  ID {row['id']:>4} | {row['created_at']} | "
            f"{row['priority']:<6} | {row['department']:<22} | "
            f"{row['predicted_category']}"
        )

    print("\ndb.py test completed successfully.")