"""Descriptor-level terminal isolation for privacy-safe local commands.

The status descriptors are duplicated before stdout and stderr are redirected.  A
command can therefore inspect and preserve unsafe diagnostics locally, restore both
terminal streams, and emit only an allowlisted final status.
"""
from __future__ import annotations

import ctypes
import os
import selectors
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar


_T = TypeVar("_T")
MAX_TERMINAL_TRANSCRIPT_BYTES = 2 * 1024 * 1024
_TRANSCRIPT_CHUNK_BYTES = 64 * 1024
_TRANSCRIPT_TRUNCATED = b"\n...[terminal transcript truncated]...\n"
_TRANSCRIPT_POLL_SECONDS = 0.05
_TRANSCRIPT_EOF_GRACE_SECONDS = 0.5
_TRANSCRIPT_STOP_GRACE_SECONDS = 0.5
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


class TerminalizedFailure(RuntimeError):
    """A callback/capture failure whose details must remain off-terminal.

    Exception subclasses deliberately remain ordinary mutable objects.  CPython
    and context managers assign ``__traceback__`` while propagating exceptions;
    frozen/slotted dataclass exceptions reject that assignment and can mask the
    original privacy-safe failure with a ``TypeError``.
    """

    def __init__(
        self,
        *,
        primary: BaseException,
        transcript: bytes,
        terminal_status: TerminalStatus,
    ) -> None:
        self.primary = primary
        self.transcript = transcript
        self.terminal_status = terminal_status
        super().__init__("terminalized command failure")


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


@dataclass(slots=True)
class _BoundedTranscript:
    """Drain one shared stdout/stderr pipe while retaining bounded evidence."""

    write_fd: int
    thread: threading.Thread
    stop: threading.Event = field(repr=False)
    chunks: list[bytes] = field(repr=False)
    retained_bytes: int = 0
    truncated: bool = False
    failure: BaseException | None = field(default=None, repr=False)

    def retained_payload(self) -> bytes:
        payload = b"".join(self.chunks)
        if self.truncated:
            payload += _TRANSCRIPT_TRUNCATED
        if len(payload) > MAX_TERMINAL_TRANSCRIPT_BYTES:
            raise OSError("terminal transcript exceeded its retention bound")
        return payload

    def finish(self) -> bytes:
        try:
            os.close(self.write_fd)
        except OSError as exc:
            if self.failure is None:
                self.failure = exc
        self.thread.join(timeout=_TRANSCRIPT_EOF_GRACE_SECONDS)
        if self.thread.is_alive():
            self.stop.set()
            self.thread.join(timeout=_TRANSCRIPT_STOP_GRACE_SECONDS)
            if self.failure is None:
                self.failure = OSError(
                    "terminal transcript retained an inherited writer"
                )
        if self.thread.is_alive() and self.failure is None:
            self.failure = OSError("terminal transcript drain did not stop")
        payload = self.retained_payload()
        if self.failure is not None:
            raise OSError("terminal transcript drain failed") from self.failure
        return payload


def _bounded_transcript_pipe() -> _BoundedTranscript:
    read_fd, write_fd = os.pipe()
    try:
        os.set_inheritable(read_fd, False)
        os.set_inheritable(write_fd, False)
        os.set_blocking(read_fd, False)
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise

    chunks: list[bytes] = []
    stop = threading.Event()
    capture: _BoundedTranscript

    def drain() -> None:
        selector: selectors.BaseSelector | None = None
        try:
            selector = selectors.DefaultSelector()
            selector.register(read_fd, selectors.EVENT_READ)
            while True:
                if stop.is_set():
                    if capture.failure is None:
                        capture.failure = OSError(
                            "terminal transcript retained an inherited writer"
                        )
                    break
                try:
                    ready = selector.select(_TRANSCRIPT_POLL_SECONDS)
                except InterruptedError:
                    continue
                if not ready:
                    continue
                try:
                    chunk = os.read(read_fd, _TRANSCRIPT_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                except InterruptedError:
                    continue
                if not chunk:
                    break
                retained_limit = (
                    MAX_TERMINAL_TRANSCRIPT_BYTES
                    - len(_TRANSCRIPT_TRUNCATED)
                )
                available = max(
                    0,
                    retained_limit - capture.retained_bytes,
                )
                if available:
                    retained = chunk[:available]
                    chunks.append(retained)
                    capture.retained_bytes += len(retained)
                if len(chunk) > available:
                    capture.truncated = True
        except BaseException as exc:
            capture.failure = exc
        finally:
            if selector is not None:
                try:
                    selector.close()
                except BaseException as exc:
                    if capture.failure is None:
                        capture.failure = exc
            try:
                os.close(read_fd)
            except OSError as exc:
                if capture.failure is None:
                    capture.failure = exc

    thread = threading.Thread(
        target=drain,
        name="evalsim-terminal-drain",
        daemon=True,
    )
    capture = _BoundedTranscript(
        write_fd=write_fd,
        thread=thread,
        stop=stop,
        chunks=chunks,
    )
    try:
        thread.start()
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise
    return capture


def capture_terminal(
    callback: Callable[[], _T],
    *,
    terminal_commit: Callable[[_T], None] | None = None,
    seal_terminal: bool = False,
) -> CapturedResult[_T]:
    """Run ``callback`` with fd 1/2 isolated and require a silent success.

    Any callback exception or captured output becomes ``TerminalizedFailure``.
    Exception details and captured bytes are carried only in-memory so the caller
    can persist them under its ignored failure directory.  With
    ``seal_terminal=True``, process fd 1/2 remain bound to a sink after the callback;
    the caller must emit one allowlisted status through ``terminal_status`` and then
    exit the process.
    """

    if not callable(callback):
        raise TypeError("callback must be callable")
    if terminal_commit is not None and not callable(terminal_commit):
        raise TypeError("terminal_commit must be callable or None")
    if type(seal_terminal) is not bool:
        raise TypeError("seal_terminal must be a boolean")

    status: TerminalStatus | None = None
    restore_stdout: int | None = None
    restore_stderr: int | None = None
    capture: _BoundedTranscript | None = None
    stdout_inheritable = os.get_inheritable(1)
    stderr_inheritable = os.get_inheritable(2)
    stdout_redirected = False
    stderr_redirected = False
    primary: BaseException | None = None
    capture_failure: BaseException | None = None
    value: _T | None = None
    transcript = b""
    terminal_commit_succeeded = False
    callback_started = False
    sink_fd: int | None = None

    try:
        status_stdout, status_stderr = _safe_dup_pair(1, 2)
        status = TerminalStatus(
            stdout_fd=status_stdout,
            stderr_fd=status_stderr,
        )
        restore_stdout, restore_stderr = _safe_dup_pair(1, 2)
        capture = _bounded_transcript_pipe()
        _flush_process_streams()
        os.dup2(
            capture.write_fd,
            1,
            inheritable=stdout_inheritable,
        )
        stdout_redirected = True
        os.dup2(
            capture.write_fd,
            2,
            inheritable=stderr_inheritable,
        )
        stderr_redirected = True
        callback_started = True
        try:
            value = callback()
        except BaseException as exc:
            primary = exc
        _flush_process_streams()
    except BaseException as exc:
        capture_failure = exc
    finally:
        # Replace the process-global redirected descriptors with a write-only
        # sink before draining.  That both releases this process's pipe writers
        # and prevents registered Python threads, raw ``_thread`` workers, and
        # native runtime threads from reaching the real terminal during the
        # drain/commit interval.  A production caller can leave the sink sealed
        # until process exit; allowlisted status uses the saved descriptors.
        restorations = (
            (
                stdout_redirected,
                restore_stdout,
                1,
                stdout_inheritable,
            ),
            (
                stderr_redirected,
                restore_stderr,
                2,
                stderr_inheritable,
            ),
        )
        if callback_started:
            try:
                sink_fd = os.open(
                    os.devnull,
                    os.O_WRONLY
                    | (os.O_CLOEXEC if hasattr(os, "O_CLOEXEC") else 0),
                )
            except BaseException as exc:
                if capture_failure is None:
                    capture_failure = exc
            if sink_fd is not None:
                for redirected, _, target, inheritable in restorations:
                    if not redirected:
                        continue
                    try:
                        os.dup2(
                            sink_fd,
                            target,
                            inheritable=inheritable,
                        )
                    except BaseException as exc:
                        if capture_failure is None:
                            capture_failure = exc
        else:
            for redirected, _, target, _ in restorations:
                if not redirected:
                    continue
                try:
                    os.close(target)
                except BaseException as exc:
                    if capture_failure is None:
                        capture_failure = exc
        if capture is not None:
            try:
                transcript = capture.finish()
            except BaseException as exc:
                try:
                    transcript = capture.retained_payload()
                except BaseException:
                    transcript = b""
                if capture_failure is None:
                    capture_failure = exc
        # The optional irreversible commit runs only after the callback,
        # transcript, and capture machinery are proven clean, but before the real
        # terminal descriptors are restored.  Therefore no async/native output can
        # cross a gap between transcript acceptance and the terminal commit.
        if (
            terminal_commit is not None
            and primary is None
            and capture_failure is None
            and not transcript
        ):
            try:
                terminal_commit(value)  # type: ignore[arg-type]
                terminal_commit_succeeded = True
            except BaseException as exc:
                primary = exc
        if not (seal_terminal and callback_started):
            for redirected, source, target, inheritable in restorations:
                if not redirected or source is None:
                    continue
                try:
                    _restore_descriptor(
                        source,
                        target,
                        inheritable=inheritable,
                    )
                except BaseException as exc:
                    if capture_failure is None:
                        capture_failure = exc
        if sink_fd is not None and sink_fd not in {1, 2}:
            try:
                os.close(sink_fd)
            except OSError as exc:
                if capture_failure is None:
                    capture_failure = exc
        for descriptor in (restore_stdout, restore_stderr):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    if capture_failure is None:
                        capture_failure = exc

    if status is None:
        raise TerminalBoundaryError("terminal_capture_failed") from None
    if primary is not None:
        raise TerminalizedFailure(
            primary=primary,
            transcript=transcript,
            terminal_status=status,
        ) from None
    if capture_failure is not None and not terminal_commit_succeeded:
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
    "MAX_TERMINAL_TRANSCRIPT_BYTES",
    "TerminalBoundaryError",
    "TerminalStatus",
    "TerminalizedFailure",
    "capture_terminal",
    "write_all",
]
