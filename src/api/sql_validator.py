"""
ATLAS-X – Deterministic SQL Pre-Validation (Feature 2)
────────────────────────────────────────────────────────
Validates SQL strings before execution to prevent injection attacks.

Rules enforced:
  1. SELECT-only  – no DML/DDL (DROP, DELETE, UPDATE, INSERT, CREATE, ALTER, TRUNCATE)
  2. Whitelisted tables only – predictions, fraud_alerts, transaction_events
  3. No injection patterns – semicolons, inline comments, UNION, stacked queries
  4. No dangerous functions – pg_sleep, pg_read_file, COPY, etc.

Usage:
    from src.api.sql_validator import validate_query, ValidationError

    try:
        validate_query(sql)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
"""
import re
from typing import Optional


# ── Allowed tables ────────────────────────────────────────────────────────────

ALLOWED_TABLES: frozenset[str] = frozenset(
    {
        "predictions",
        "fraud_alerts",
        "transaction_events",
        "transaction_audit",
        "transaction_embeddings",   # pgvector similarity search
    }
)

# ── Forbidden patterns ────────────────────────────────────────────────────────

# DML / DDL keywords that must not appear as statement starters or after semicolons
_FORBIDDEN_STATEMENTS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER|TRUNCATE|REPLACE|MERGE|EXEC|EXECUTE|CALL)\b",
    re.IGNORECASE,
)

# SQL injection patterns
_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("semicolon (statement terminator / stacked queries)",
     re.compile(r";")),
    ("inline comment (--)",
     re.compile(r"--")),
    ("block comment (/* */)",
     re.compile(r"/\*|\*/")),
    ("UNION-based injection",
     re.compile(r"\bUNION\b", re.IGNORECASE)),
    ("dangerous function: pg_sleep",
     re.compile(r"\bpg_sleep\b", re.IGNORECASE)),
    ("dangerous function: pg_read_file",
     re.compile(r"\bpg_read_file\b", re.IGNORECASE)),
    ("dangerous function: COPY",
     re.compile(r"\bCOPY\b", re.IGNORECASE)),
    ("dangerous function: lo_import / lo_export",
     re.compile(r"\blo_(import|export)\b", re.IGNORECASE)),
    ("null-byte injection",
     re.compile(r"\x00")),
    ("hex-encoded injection attempt",
     re.compile(r"0x[0-9a-fA-F]{4,}", re.IGNORECASE)),
]

# Extract table names referenced in a query (FROM / JOIN clauses)
_TABLE_REF = re.compile(
    r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)


# ── Public exception ──────────────────────────────────────────────────────────

class ValidationError(ValueError):
    """Raised when a SQL string fails pre-validation."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"SQL validation failed: {reason}")


# ── Core validator ────────────────────────────────────────────────────────────

def validate_query(sql: str, *, allow_tables: Optional[frozenset[str]] = None) -> bool:
    """
    Validate *sql* against the ATLAS-X security policy.

    Parameters
    ----------
    sql:
        The raw SQL string to validate.
    allow_tables:
        Override the default whitelist.  Pass ``None`` to use
        ``ALLOWED_TABLES``.

    Returns
    -------
    True
        Always returns ``True`` on success (so callers can write
        ``assert validate_query(sql)`` in tests).

    Raises
    ------
    ValidationError
        With a human-readable ``reason`` attribute describing the
        specific policy violation.
    """
    if not isinstance(sql, str):
        raise ValidationError("SQL must be a string.")

    stripped = sql.strip()
    if not stripped:
        raise ValidationError("Empty SQL string.")

    # ── Rule 1: SELECT-only ───────────────────────────────────────────────────
    first_token = stripped.split()[0].upper()
    if first_token != "SELECT":
        raise ValidationError(
            f"Only SELECT statements are allowed; got '{first_token}'."
        )

    # ── Rule 2: No forbidden DML/DDL anywhere in the string ──────────────────
    match = _FORBIDDEN_STATEMENTS.search(stripped)
    if match:
        raise ValidationError(
            f"Forbidden keyword '{match.group().upper()}' detected in query."
        )

    # ── Rule 3: No injection patterns ────────────────────────────────────────
    for description, pattern in _INJECTION_PATTERNS:
        if pattern.search(stripped):
            raise ValidationError(
                f"Forbidden pattern detected: {description}."
            )

    # ── Rule 4: Whitelisted tables only ──────────────────────────────────────
    whitelist = allow_tables if allow_tables is not None else ALLOWED_TABLES
    referenced = {m.group(1).lower() for m in _TABLE_REF.finditer(stripped)}
    disallowed = referenced - {t.lower() for t in whitelist}
    if disallowed:
        raise ValidationError(
            f"Query references non-whitelisted table(s): "
            f"{', '.join(sorted(disallowed))}. "
            f"Allowed: {', '.join(sorted(whitelist))}."
        )

    return True


def safe_table_name(name: str) -> str:
    """
    Return *name* if it is in the whitelist, otherwise raise ValidationError.
    Use this when building dynamic queries with table names from user input.
    """
    if name.lower() not in {t.lower() for t in ALLOWED_TABLES}:
        raise ValidationError(
            f"Table '{name}' is not in the allowed list: "
            f"{', '.join(sorted(ALLOWED_TABLES))}."
        )
    return name
