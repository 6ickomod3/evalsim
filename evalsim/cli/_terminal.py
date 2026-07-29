"""Descriptor-level terminal isolation for privacy-safe local commands.

The status descriptors are duplicated before stdout and stderr are redirected.  A
command can therefore inspect and preserve unsafe diagnostics locally, restore both
terminal streams, and emit only an allowlisted final status.
"""
from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar


_T = TypeVar("_T")
_TERMINAL_CODES = frozenset(
    {
        "terminal_capture_failed",
        "terminal_output_detected",
    }
)


class TerminalBoundaryError(RuntimeError):
    """A terminal-isolation failure carrying only a stable reason code."""

    def __init__(self, code: str) -> None:
        if code not in _TERMINAL_CODES:
            raise ValueError("unregistered terminal boundary error code")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class TerminalStatus:
    """Non-inheritable copies of the command's original status streams."""

    stdout_fd: int
    stderr_fd: int

    def close_best_effort(self) -> None:
        for descriptor in (self.stdout_fd, self.stderr_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass


@dataclass(frozen=True, slots=True)
class CapturedResult(Generic[_T]):
    """A successful callback result plus safe status descriptors."""

    value: _T
    terminal_status: TerminalStatus


@dataclass(frozen=True, slots=True)
class TerminalizedFailure(RuntimeError):
    """A callback/capture failure whose details must remain off-terminal."""

    primary: BaseException = field(repr=False)
    transcript: bytes = field(repr=False)
    terminal_status: TerminalStatus = field(repr=False)

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, "terminalized command failure")


def _flush_process_streams() -> None:
    # A failed flush means we cannot prove whether process-global buffered output
    # crossed the capture boundary.  Fail closed instead of silently treating the
    # transcript as complete.
    sys.stdout.flush()
    sys.stderr.flush()
    libc = ctypes.CDLL(None, use_errno=True)
    fflush = libc.fflush
    fflush.argtypes = [ctypes.c_void_p]
    fflush.restype = ctypes.c_int
    if fflush(None) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "native stdio flush failed")


def _safe_dup(descriptor: int) -> int:
    duplicate = os.dup(descriptor)
    try:
        os.set_inheritable(duplicate, False)
    except BaseException:
        os.close(duplicate)
        raise
    return duplicate


def _safe_dup_pair(
    first: int,
    second: int,
) -> tuple[int, int]:
    first_duplicate = _safe_dup(first)
    try:
        second_duplicate = _safe_dup(second)
    except BaseException:
        os.close(first_duplicate)
        raise
    return first_duplicate, second_duplicate


def _restore_descriptor(
    source: int,
    target: int,
    *,
    inheritable: bool,
) -> None:
    os.dup2(source, target, inheritable=inheritable)


def _read_transcript(handle: object) -> bytes:
    handle.seek(0)  # type: ignore[attr-defined]
    payload = handle.read()  # type: ignore[attr-defined]
    if type(payload) is not bytes:
        raise OSError("terminal capture did not produce bytes")
    return payload


def capture_terminal(callback: Callable[[], _T]) -> CapturedResult[_T]:
    """Run ``callback`` with fd 1/2 isolated and require a silent success.

    Any callback exception or captured output becomes ``TerminalizedFailure``.
    Exception details and captured bytes are carried only in-memory so the caller can
    persist them under its ignored failure directory after terminal restoration.
    """

    if not callable(callback):
        raise TypeError("callback must be callable")

    status: TerminalStatus | None = None
    restore_stdout: int | None = None
    restore_stderr: int | None = None
    capture = None
    stdout_inheritable = os.get_inheritable(1)
    stderr_inheritable = os.get_inheritable(2)
    stdout_redirected = False
    stderr_redirected = False
    primary: BaseException | None = None
    capture_failure: BaseException | None = None
    value: _T | None = None
    transcript = b""

    try:
        status_stdout, status_stderr = _safe_dup_pair(1, 2)
        status = TerminalStatus(
            stdout_fd=status_stdout,
            stderr_fd=status_stderr,
        )
        restore_stdout, restore_stderr = _safe_dup_pair(1, 2)
        capture = tempfile.TemporaryFile(mode="w+b")
        os.set_inheritable(capture.fileno(), False)
        _flush_process_streams()
        os.dup2(
            capture.fileno(),
            1,
            inheritable=stdout_inheritable,
        )
        stdout_redirected = True
        os.dup2(
            capture.fileno(),
            2,
            inheritable=stderr_inheritable,
        )
        stderr_redirected = True
        try:
            value = callback()
        except BaseException as exc:
            primary = exc
        _flush_process_streams()
    except BaseException as exc:
        capture_failure = exc
    finally:
        try:
            if stdout_redirected and restore_stdout is not None:
                _restore_descriptor(
                    restore_stdout,
                    1,
                    inheritable=stdout_inheritable,
                )
            if stderr_redirected and restore_stderr is not None:
                _restore_descriptor(
                    restore_stderr,
                    2,
                    inheritable=stderr_inheritable,
                )
        except BaseException as exc:
            if capture_failure is None:
                capture_failure = exc
        finally:
            for descriptor in (restore_stdout, restore_stderr):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
        if capture is not None:
            try:
                transcript = _read_transcript(capture)
            except BaseException as exc:
                if capture_failure is None:
                    capture_failure = exc
            finally:
                capture.close()

    if status is None:
        raise TerminalBoundaryError("terminal_capture_failed") from None
    if primary is not None:
        raise TerminalizedFailure(
            primary=primary,
            transcript=transcript,
            terminal_status=status,
        ) from None
    if capture_failure is not None:
        raise TerminalizedFailure(
            primary=TerminalBoundaryError("terminal_capture_failed"),
            transcript=transcript,
            terminal_status=status,
        ) from None
    if transcript:
        raise TerminalizedFailure(
            primary=TerminalBoundaryError("terminal_output_detected"),
            transcript=transcript,
            terminal_status=status,
        ) from None
    return CapturedResult(
        value=value,  # type: ignore[arg-type]
        terminal_status=status,
    )


def write_all(descriptor: int, payload: bytes) -> None:
    """Write exact bytes to one restored status descriptor."""

    if type(payload) is not bytes:
        raise TypeError("terminal payload must be exact bytes")
    remaining = memoryview(payload)
    while remaining:
        try:
            written = os.write(descriptor, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("status descriptor accepted no bytes")
        remaining = remaining[written:]


__all__ = [
    "CapturedResult",
    "TerminalBoundaryError",
    "TerminalStatus",
    "TerminalizedFailure",
    "capture_terminal",
    "write_all",
]
