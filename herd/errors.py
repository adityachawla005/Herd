"""One error type for everything a user can fix, carrying the fix."""


class HerdError(RuntimeError):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint
