from __future__ import annotations

from typing import Any, Dict, List, Optional


DOCUMENTATION_PATH = ".agents/skills/static-scene-harness/references/tool-schemas.md"


class ContractViolation(ValueError):
    """All independently detectable contract violations for one request."""

    def __init__(self, issues: List[Dict[str, Any]]):
        super().__init__("{} contract issue(s)".format(len(issues)))
        self.issues = issues


class StructuredToolError(RuntimeError):
    def __init__(self, code: str, message: str, field_path: str = "$",
                 expected: Optional[str] = None, received: Any = None,
                 expected_example: Any = None, retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.field_path = field_path
        self.expected = expected
        self.received = received
        self.expected_example = expected_example
        self.retryable = retryable


class FragmentChangedAfterFreeze(StructuredToolError):
    def __init__(self, received: str, expected: str):
        super().__init__(
            "FRAGMENT_CHANGED_AFTER_FREEZE",
            "The registered geometry changed after its first successful render.",
            "$.fragment_id", "Frozen fragment hash {}.".format(expected), received,
            expected, False,
        )


class EvidenceUnavailable(StructuredToolError):
    def __init__(self, message: str):
        super().__init__("EVIDENCE_UNAVAILABLE", message, retryable=True)


def issue(error_code: str, message: str, field_path: str, expected: str,
          received: Any, expected_example: Any, retryable: bool = True) -> Dict[str, Any]:
    return {
        "error_code": error_code,
        "message": message,
        "field_path": field_path,
        "expected": expected,
        "received": received,
        "expected_example": expected_example,
        "retryable": retryable,
        "documentation_path": DOCUMENTATION_PATH,
    }


def tool_error(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, ContractViolation):
        first = exc.issues[0]
        return {
            "ok": False,
            "error_code": "CONTRACT_VIOLATION",
            "message": str(exc),
            "issues": exc.issues,
            "documentation_path": DOCUMENTATION_PATH,
            "retryable": all(row.get("retryable", True) for row in exc.issues),
            "field_path": first.get("field_path", "$"),
        }
    if isinstance(exc, StructuredToolError):
        return {
            "ok": False,
            "error_code": exc.code,
            "message": str(exc),
            "field_path": exc.field_path,
            "expected": exc.expected,
            "received": exc.received,
            "expected_example": exc.expected_example,
            "retryable": exc.retryable,
            "documentation_path": DOCUMENTATION_PATH,
        }
    return {
        "ok": False,
        "error_code": "INTERNAL_ERROR",
        "message": "The harness failed internally; inspect server stderr.",
        "field_path": "$",
        "expected": "A valid request handled successfully.",
        "received": None,
        "expected_example": None,
        "retryable": False,
        "documentation_path": DOCUMENTATION_PATH,
    }
