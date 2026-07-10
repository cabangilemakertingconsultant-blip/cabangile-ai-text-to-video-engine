import csv
import io
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


class FileManagerException(Exception):
    """Custom exception raised by the FileManager class for file operation errors."""
    pass


class FileManager:
    def __init__(self, base_dir: str | Path | None = None):
        """Initialize the FileManager with an optional base directory and a thread lock."""
        self.base_dir = Path(base_dir) if base_dir else None
        self._lock = threading.Lock()

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve the given path, optionally anchoring it to the base directory."""
        p = Path(path)
        if self.base_dir and not p.is_absolute():
            return self.base_dir / p
        return p

    def read_text(self, path: str | Path, encoding: str = "utf-8") -> str:
        """Read the entire contents of a file as a string.

        Args:
            path: The path to the file.
            encoding: The text encoding to use.

        Returns:
            The string contents of the file.

        Raises:
            FileManagerException: If the file cannot be read.
        """
        resolved = self.resolve_path(path)
        with self._lock:
            try:
                return resolved.read_text(encoding=encoding)
            except Exception as e:
                raise FileManagerException(f"Failed to read text from {resolved}: {e}") from e

    def write_text(self, path: str | Path, data: str, encoding: str = "utf-8") -> Path:
        """Write string data to a file atomically.

        Args:
            path: The target path.
            data: The string data to write.
            encoding: The text encoding to use.

        Returns:
            The resolved path to the written file.

        Raises:
            FileManagerException: If the file cannot be written.
        """
        resolved = self.resolve_path(path)
        with self._lock:
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                self._write_atomic(resolved, data.encode(encoding))
                return resolved
            except Exception as e:
                raise FileManagerException(f"Failed to write text to {resolved}: {e}") from e

    def append_text(self, path: str | Path, data: str, encoding: str = "utf-8") -> Path:
        """Append string data to a file.

        Args:
            path: The target path.
            data: The string data to append.
            encoding: The text encoding to use.

        Returns:
            The resolved path to the modified file.

        Raises:
            FileManagerException: If the data cannot be appended.
        """
        resolved = self.resolve_path(path)
        with self._lock:
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                with resolved.open(mode="a", encoding=encoding) as f:
                    f.write(data)
                return resolved
            except Exception as e:
                raise FileManagerException(f"Failed to append text to {resolved}: {e}") from e

    def read_bytes(self, path: str | Path) -> bytes:
        """Read the entire contents of a file as bytes.

        Args:
            path: The path to the file.

        Returns:
            The binary contents of the file.

        Raises:
            FileManagerException: If the file cannot be read.
        """
        resolved = self.resolve_path(path)
        with self._lock:
            try:
                return resolved.read_bytes()
            except Exception as e:
                raise FileManagerException(f"Failed to read bytes from {resolved}: {e}") from e

    def write_bytes(self, path: str | Path, data: bytes) -> Path:
        """Write binary data to a file atomically.

        Args:
            path: The target path.
            data: The bytes to write.

        Returns:
            The resolved path to the written file.

        Raises:
            FileManagerException: If the file cannot be written.
        """
        resolved = self.resolve_path(path)
        with self._lock:
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                self._write_atomic(resolved, data)
                return resolved
            except Exception as e:
                raise FileManagerException(f"Failed to write bytes to {resolved}: {e}") from e

    def append_bytes(self, path: str | Path, data: bytes) -> Path:
        """Append binary data to a file.

        Args:
            path: The target path.
            data: The bytes to append.

        Returns:
            The resolved path to the modified file.

        Raises:
            FileManagerException: If the data cannot be appended.
        """
        resolved = self.resolve_path(path)
        with self._lock:
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                with resolved.open(mode="ab") as f:
                    f.write(data)
                return resolved
            except Exception as e:
                raise FileManagerException(f"Failed to append bytes to {resolved}: {e}") from e

    def read_json(self, path: str | Path, encoding: str = "utf-8") -> Any:
        """Read and parse a JSON file.

        Args:
            path: The path to the JSON file.
            encoding: The text encoding to use.

        Returns:
            The parsed Python data structures.

        Raises:
            FileManagerException: If the JSON cannot be read or decoded.
        """
        resolved = self.resolve_path(path)
        with self._lock:
            try:
                with resolved.open(mode="r", encoding=encoding) as f:
                    return json.load(f)
            except Exception as e:
                raise FileManagerException(f"Failed to read JSON from {resolved}: {e}") from e

    def write_json(self, path: str | Path, data: Any, encoding: str = "utf-8") -> Path:
        """Write data to a file as formatted JSON atomically.

        Args:
            path: The target path.
            data: The JSON-serializable data.
            encoding: The text encoding to use.

        Returns:
            The resolved path to the written file.

        Raises:
            FileManagerException: If the data cannot be written or encoded.
        """
        resolved = self.resolve_path(path)
        with self._lock:
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                serialized = json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True)
                self._write_atomic(resolved, serialized.encode(encoding))
                return resolved
            except Exception as e:
                raise FileManagerException(f"Failed to write JSON to {resolved}: {e}") from e

    def read_csv(self, path: str | Path, encoding: str = "utf-8") -> list[dict[str, str]]:
        """Read a CSV file into a list of dictionaries.

        Args:
            path: The path to the CSV file.
            encoding: The text encoding to use.

        Returns:
            A list of dictionaries containing row data.

        Raises:
            FileManagerException: If the CSV cannot be read or parsed.
        """
        resolved = self.resolve_path(path)
        with self._lock:
            try:
                with resolved.open(mode="r", encoding=encoding, newline="") as f:
                    reader = csv.DictReader(f)
                    return list(reader)
            except Exception as e:
                raise FileManagerException(f"Failed to read CSV from {resolved}: {e}") from e

    def write_csv(self, path: str | Path, data: list[dict[str, str]], encoding: str = "utf-8") -> Path:
        """Write a list of dictionaries to a CSV file atomically with sorted headers.

        Args:
            path: The target path.
            data: The row data to write.
            encoding: The text encoding to use.

        Returns:
            The resolved path to the written file.

        Raises:
            FileManagerException: If the CSV cannot be written.
        """
        resolved = self.resolve_path(path)
        if not data:
            raise FileManagerException("Cannot write empty data list to CSV; fieldnames cannot be determined.")

        with self._lock:
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                fieldnames = sorted(list(data[0].keys()))

                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                    writer.writerow(row)

                self._write_atomic(resolved, output.getvalue().encode(encoding))
                return resolved
            except Exception as e:
                raise FileManagerException(f"Failed to write CSV to {resolved}: {e}") from e

    def _write_atomic(self, path: Path, data: bytes) -> None:
        """Atomically write binary data using a temporary file in the same directory.

        Args:
            path: The final resolved destination path.
            data: The binary payload to write.
        """
        temp_file = None
        try:
            fd, temp_path_str = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
            temp_file = Path(temp_path_str)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            temp_file.replace(path)
        except Exception as e:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise e
