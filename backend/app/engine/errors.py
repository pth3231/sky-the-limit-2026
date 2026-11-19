"""Errors raised for draft requests the engine cannot serve."""


class DraftStateError(ValueError):
    """The submitted draft or series state breaks the draft rules or references unknown data."""


class UnknownPatchError(DraftStateError):
    """The requested patch has no aggregates."""
