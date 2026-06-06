"""
Lightweight KataGo GTP client for terminal play.

This module launches KataGo as a subprocess and communicates using GTP.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import queue
import subprocess
import threading
from typing import Optional, Tuple


class KataGoError(RuntimeError):
    """Raised when KataGo process or GTP communication fails."""


@dataclass
class KataGoConfig:
    """Configuration needed to start KataGo in GTP mode."""

    executable: str
    model_path: str
    config_path: Optional[str] = None
    board_size: int = 7
    komi: float = 9.5


class KataGoGTP:
    """Minimal GTP wrapper for KataGo."""

    _COLS = "ABCDEFGHJKLMNOPQRST"

    def __init__(self, cfg: KataGoConfig) -> None:
        self.cfg = cfg
        self._process: Optional[subprocess.Popen[str]] = None
        self._stdout_queue: queue.Queue[str] = queue.Queue()
        self._stderr_queue: queue.Queue[str] = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start KataGo and initialize board settings."""
        if self._process is not None:
            return

        executable = os.path.expanduser(self.cfg.executable)
        model = os.path.expanduser(self.cfg.model_path)
        config = os.path.expanduser(self.cfg.config_path) if self.cfg.config_path else None

        cmd = [executable, "gtp", "-model", model]
        if config:
            cmd.extend(["-config", config])

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise KataGoError(f"KataGo executable not found: {executable}") from exc

        if self._process.stdout is None or self._process.stdin is None:
            raise KataGoError("Failed to open KataGo stdio streams")

        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

        self._send_and_expect_ok(f"boardsize {self.cfg.board_size}")
        self._send_and_expect_ok(f"komi {self.cfg.komi}")
        self._send_and_expect_ok("clear_board")

    def close(self) -> None:
        """Terminate KataGo process."""
        if self._process is None:
            return

        try:
            self._send_command("quit")
        except Exception:
            pass

        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()

        self._process = None

    def play(self, color: int, row: Optional[int], col: Optional[int]) -> None:
        """Send a played move to KataGo to keep state in sync."""
        gtp_color = self._color_to_gtp(color)
        move = "pass" if row is None or col is None else self.to_gtp_vertex(row, col, self.cfg.board_size)
        self._send_and_expect_ok(f"play {gtp_color} {move}")

    def genmove(self, color: int) -> Tuple[Optional[int], Optional[int], str]:
        """Ask KataGo for a move and return both parsed and raw forms."""
        gtp_color = self._color_to_gtp(color)
        raw = self._send_and_read_response(f"genmove {gtp_color}")

        lower = raw.strip().lower()
        if lower in {"pass", "resign"}:
            return None, None, lower

        row, col = self.from_gtp_vertex(raw, self.cfg.board_size)
        return row, col, raw

    @classmethod
    def to_gtp_vertex(cls, row: int, col: int, size: int) -> str:
        """Convert internal row/col (0-based, top-left origin) to GTP coordinates."""
        if col < 0 or col >= size or row < 0 or row >= size:
            raise ValueError("Move is out of board bounds")
        letter = cls._COLS[col]
        number = size - row
        return f"{letter}{number}"

    @classmethod
    def from_gtp_vertex(cls, vertex: str, size: int) -> Tuple[int, int]:
        """Convert GTP coordinates to internal row/col (0-based, top-left origin)."""
        vertex = vertex.strip().upper()
        if len(vertex) < 2:
            raise ValueError(f"Invalid GTP move: {vertex}")

        col_char = vertex[0]
        if col_char not in cls._COLS[:size]:
            raise ValueError(f"Invalid GTP column: {vertex}")
        col = cls._COLS.index(col_char)

        try:
            number = int(vertex[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid GTP row: {vertex}") from exc

        if number < 1 or number > size:
            raise ValueError(f"GTP row out of bounds: {vertex}")

        row = size - number
        return row, col

    def _color_to_gtp(self, color: int) -> str:
        return "B" if color == 1 else "W"

    def _read_stdout(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._stdout_queue.put(line.rstrip("\n"))

    def _read_stderr(self) -> None:
        assert self._process is not None
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr_queue.put(line.rstrip("\n"))

    def _send_command(self, command: str) -> None:
        if self._process is None or self._process.stdin is None:
            raise KataGoError("KataGo process is not running")
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

    def _send_and_expect_ok(self, command: str) -> None:
        self._send_and_read_response(command)

    def _send_and_read_response(self, command: str) -> str:
        self._send_command(command)
        return self._read_gtp_response(command)

    def _read_gtp_response(self, command: str) -> str:
        lines = []
        while True:
            try:
                line = self._stdout_queue.get(timeout=10)
            except queue.Empty as exc:
                if self._process is not None and self._process.poll() is not None:
                    stderr_text = self._collect_stderr()
                    detail = f" Process exited with code {self._process.returncode}."
                    if stderr_text:
                        detail += f" Stderr: {stderr_text}"
                    raise KataGoError(f"KataGo failed while waiting for response to: {command}.{detail}") from exc
                raise KataGoError(f"Timed out waiting for KataGo response to: {command}") from exc

            if line == "":
                break
            lines.append(line)

        if not lines:
            raise KataGoError(f"Empty response from KataGo for command: {command}")

        first = lines[0]
        if first.startswith("?"):
            msg = first[1:].strip() or "Unknown GTP error"
            raise KataGoError(f"KataGo rejected '{command}': {msg}")

        if not first.startswith("="):
            raise KataGoError(f"Malformed GTP response for '{command}': {first}")

        payload = first[1:].strip()
        if payload:
            return payload

        if len(lines) > 1:
            return "\n".join(lines[1:]).strip()

        return ""

    def _collect_stderr(self) -> str:
        lines = []
        while True:
            try:
                lines.append(self._stderr_queue.get_nowait())
            except queue.Empty:
                break
        return " | ".join(line for line in lines if line)