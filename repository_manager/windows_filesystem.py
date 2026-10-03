"""Dormant Windows handle primitives; no boundary or lease integration.

Only explicitly opened, owned capabilities are accepted. Bootstrap is limited
to an exact drive root plus an independently trusted identity. UNC/device paths,
existing-file mutation, owner-only ACLs, atomic complete publication and durable
directory-entry commits are unsupported. Creation inherits the directory ACL;
it is NOT an owner-only lease publication primitive. File flush is not evidence
of namespace crash durability. Actual Windows hostile-race/ABI qualification is
required before any separately reviewed adapter may use this module.

Directory handles deny write/delete sharing; newly created files deny all
sharing. All operations and closes on this backend serialize. Raw handles must
never escape to code that could close them. Python object privacy is not a
security boundary against hostile code inside this process.

Native contracts:
https://learn.microsoft.com/windows/win32/api/winternl/nf-winternl-ntcreatefile
https://learn.microsoft.com/windows/win32/api/ntdef/ns-ntdef-_object_attributes
https://learn.microsoft.com/windows/win32/api/winbase/ns-winbase-file_id_info
https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-setfileinformationbyhandle
"""

from __future__ import annotations

import ctypes as c
import re
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_DWORD = c.c_uint32
_HANDLE = c.c_void_p
_INVALID_HANDLE = c.c_void_p(-1).value
_FILE_OPEN = 1
_FILE_CREATE = 2
_READ_ATTRIBUTES = 0x80
_SYNCHRONIZE = 0x100000
_GENERIC_WRITE = 0x40000000
_DELETE = 0x10000
_SHARE_READ = 1
_DONT_REPARSE = 0x1000
_CASE_INSENSITIVE = 0x40
_REPARSE_POINT = 0x400
_DIRECTORY = 0x10
_OWNED = object()


class WindowsFilesystemError(OSError):
    """A native operation failed or could not prove the required invariant."""


class UnsupportedGuarantee(WindowsFilesystemError):
    """The dormant backend cannot supply a requested guarantee."""


@dataclass(frozen=True)
class FileIdentity:
    """Volume serial and the entire 128-bit FILE_ID_INFO identity."""

    volume: int
    file_id: bytes

    def __post_init__(self) -> None:
        if type(self.volume) is not int or not 0 <= self.volume < 2**64:
            raise WindowsFilesystemError("invalid volume identity")
        if type(self.file_id) is not bytes or len(self.file_id) != 16:
            raise WindowsFilesystemError("file identity must contain 128 bits")


@dataclass(frozen=True)
class _Snapshot:
    identity: FileIdentity
    directory: bool
    attributes: int
    reparse_tag: int
    links: int
    delete_pending: bool
    disk: bool


class _UnicodeString(c.Structure):
    _fields_ = [
        ("Length", c.c_uint16),
        ("MaximumLength", c.c_uint16),
        ("Buffer", c.c_void_p),
    ]


class _ObjectAttributes(c.Structure):
    _fields_ = [
        ("Length", _DWORD),
        ("RootDirectory", _HANDLE),
        ("ObjectName", c.POINTER(_UnicodeString)),
        ("Attributes", _DWORD),
        ("SecurityDescriptor", c.c_void_p),
        ("SecurityQualityOfService", c.c_void_p),
    ]


class _StatusUnion(c.Union):
    _fields_ = [("Status", c.c_int32), ("Pointer", c.c_void_p)]


class _IoStatus(c.Structure):
    _anonymous_ = ("result",)
    _fields_ = [("result", _StatusUnion), ("Information", c.c_size_t)]


class _FileIdInfo(c.Structure):
    _fields_ = [("VolumeSerialNumber", c.c_uint64), ("FileId", c.c_ubyte * 16)]


class _StandardInfo(c.Structure):
    _fields_ = [
        ("AllocationSize", c.c_int64),
        ("EndOfFile", c.c_int64),
        ("NumberOfLinks", _DWORD),
        ("DeletePending", c.c_ubyte),
        ("Directory", c.c_ubyte),
    ]


class _AttributeInfo(c.Structure):
    _fields_ = [("FileAttributes", _DWORD), ("ReparseTag", _DWORD)]


class _DispositionInfo(c.Structure):
    _fields_ = [("DeleteFile", c.c_ubyte)]


def _component(name: str) -> str:
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise WindowsFilesystemError("expected one literal path component")
    if any(ord(char) < 32 or char in '\\/:*?"<>|' for char in name):
        raise WindowsFilesystemError("path separators, streams and controls refused")
    if name.endswith((".", " ")):
        raise WindowsFilesystemError("ambiguous trailing dot or space refused")
    _check_component_encoding(name)
    return name


def _check_component_encoding(name: str) -> None:
    stem = name.split(".", 1)[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} or re.fullmatch(
        r"(?:COM|LPT)[0-9¹²³]", stem
    ):
        raise WindowsFilesystemError("DOS device alias refused")
    try:
        encoded = name.encode("utf-16-le")
    except UnicodeEncodeError as exc:
        raise WindowsFilesystemError("invalid Unicode component") from exc
    if len(encoded) > 510:
        raise WindowsFilesystemError("component exceeds supported length")


class _NativeAPI:
    """Thin, synchronous ctypes binding; injected DLLs are a private test seam."""

    def __init__(
        self,
        *,
        kernel32: Any = None,
        ntdll: Any = None,
        last_error: Callable[[], int] | None = None,
    ) -> None:
        if kernel32 is None or ntdll is None:
            if sys.platform != "win32":
                raise UnsupportedGuarantee("native Windows runtime required")
            kernel32 = c.WinDLL("kernel32", use_last_error=True)
            ntdll = c.WinDLL("ntdll", use_last_error=True)
            last_error = c.get_last_error
        self._error = last_error or (lambda: 0)
        self._kernel = kernel32
        self._nt = ntdll
        signatures = (
            (
                kernel32,
                "CreateFileW",
                _HANDLE,
                [c.c_wchar_p, _DWORD, _DWORD, c.c_void_p, _DWORD, _DWORD, _HANDLE],
            ),
            (
                kernel32,
                "GetFileInformationByHandleEx",
                c.c_int32,
                [_HANDLE, c.c_int32, c.c_void_p, _DWORD],
            ),
            (kernel32, "GetFileType", _DWORD, [_HANDLE]),
            (
                kernel32,
                "WriteFile",
                c.c_int32,
                [_HANDLE, c.c_void_p, _DWORD, c.POINTER(_DWORD), c.c_void_p],
            ),
            (kernel32, "FlushFileBuffers", c.c_int32, [_HANDLE]),
            (
                kernel32,
                "SetFileInformationByHandle",
                c.c_int32,
                [_HANDLE, c.c_int32, c.c_void_p, _DWORD],
            ),
            (kernel32, "CloseHandle", c.c_int32, [_HANDLE]),
            (
                ntdll,
                "NtCreateFile",
                c.c_int32,
                [
                    c.POINTER(_HANDLE),
                    _DWORD,
                    c.POINTER(_ObjectAttributes),
                    c.POINTER(_IoStatus),
                    c.c_void_p,
                    _DWORD,
                    _DWORD,
                    _DWORD,
                    _DWORD,
                    c.c_void_p,
                    _DWORD,
                ],
            ),
        )
        try:
            for dll, name, result, arguments in signatures:
                function = getattr(dll, name)
                function.restype = result
                function.argtypes = arguments
        except AttributeError as exc:
            raise UnsupportedGuarantee(
                "required native entry point unavailable"
            ) from exc

    def _check(self, result: int, operation: str) -> None:
        if not result:
            code = self._error()
            raise WindowsFilesystemError(code, f"{operation} failed (Win32 {code})")

    def open_root(self, root: str) -> int:
        handle = self._kernel.CreateFileW(
            root,
            _READ_ATTRIBUTES | _SYNCHRONIZE,
            _SHARE_READ,
            None,
            3,
            0x02000000 | 0x00200000,
            None,
        )
        if handle in (None, 0, _INVALID_HANDLE):
            raise WindowsFilesystemError(self._error(), "drive root open failed")
        return int(handle)

    def open_relative(
        self, parent: int, name: str, directory: bool, create: bool
    ) -> int:
        raw = name.encode("utf-16-le")
        buffer = c.create_string_buffer(raw + b"\0\0")
        string = _UnicodeString(len(raw), len(raw) + 2, c.cast(buffer, c.c_void_p))
        attributes = _ObjectAttributes(
            c.sizeof(_ObjectAttributes),
            parent,
            c.pointer(string),
            _CASE_INSENSITIVE | _DONT_REPARSE,
            None,
            None,
        )
        handle, status = _HANDLE(), _IoStatus()
        access = _READ_ATTRIBUTES | _SYNCHRONIZE
        if create:
            access |= _GENERIC_WRITE | _DELETE
        options = 0x00200000 | 0x20 | (1 if directory else 0x40)
        result = self._nt.NtCreateFile(
            c.byref(handle),
            access,
            c.byref(attributes),
            c.byref(status),
            None,
            0x80,
            0 if create else _SHARE_READ,
            _FILE_CREATE if create else _FILE_OPEN,
            options,
            None,
            0,
        )
        try:
            return self._completed_open(result, status, handle.value, create)
        except BaseException as exc:
            self._close_failed_open(handle.value, exc)
            raise

    @staticmethod
    def _completed_open(
        result: int, status: _IoStatus, raw: int | None, create: bool
    ) -> int:
        # STATUS_PENDING/informational results are not completed admissions.
        if result != 0 or status.Status != 0:
            raise WindowsFilesystemError(
                f"NtCreateFile failed: 0x{result & 0xFFFFFFFF:08x}; "
                f"IO status 0x{status.Status & 0xFFFFFFFF:08x}"
            )
        if raw is None or raw in (0, _INVALID_HANDLE):
            raise WindowsFilesystemError("NtCreateFile returned an invalid handle")
        if status.Information != (2 if create else 1):
            raise WindowsFilesystemError("unexpected create/open disposition")
        return raw

    def _close_failed_open(self, raw: int | None, error: BaseException) -> None:
        if raw is None or raw in (0, _INVALID_HANDLE):
            return
        try:
            self.close(raw)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "open and close failed", [error, cleanup]
            ) from None

    def _query(self, handle: int, kind: int, value: Any) -> Any:
        self._check(
            self._kernel.GetFileInformationByHandleEx(
                handle, kind, c.byref(value), c.sizeof(value)
            ),
            "GetFileInformationByHandleEx",
        )
        return value

    def _identity(self, handle: int) -> FileIdentity:
        value = self._query(handle, 18, _FileIdInfo())
        return FileIdentity(value.VolumeSerialNumber, bytes(value.FileId))

    def query(self, handle: int) -> _Snapshot:
        identity = self._identity(handle)
        standard = self._query(handle, 1, _StandardInfo())
        attributes = self._query(handle, 9, _AttributeInfo())
        disk = self._kernel.GetFileType(handle) == 1
        if self._identity(handle) != identity:
            raise WindowsFilesystemError("identity changed during handle inspection")
        return _Snapshot(
            identity,
            bool(standard.Directory),
            attributes.FileAttributes,
            attributes.ReparseTag,
            standard.NumberOfLinks,
            bool(standard.DeletePending),
            disk,
        )

    def write(self, handle: int, value: bytes) -> int:
        written = _DWORD()
        buffer = c.create_string_buffer(value)
        self._check(
            self._kernel.WriteFile(handle, buffer, len(value), c.byref(written), None),
            "WriteFile",
        )
        return written.value

    def flush(self, handle: int) -> None:
        self._check(self._kernel.FlushFileBuffers(handle), "FlushFileBuffers")

    def discard(self, handle: int) -> None:
        value = _DispositionInfo(1)
        self._check(
            self._kernel.SetFileInformationByHandle(
                handle, 4, c.byref(value), c.sizeof(value)
            ),
            "FileDispositionInfo",
        )

    def close(self, handle: int) -> None:
        try:
            self._check(self._kernel.CloseHandle(handle), "CloseHandle")
        except WindowsFilesystemError as exc:
            exc.add_note(
                "Native closure is uncertain: the resource may remain open. "
                "Do not retry the numeric handle; its slot may have been reused."
            )
            raise


class OwnedHandle:
    """One owned capability. Close once, including after uncertain close failure.

    Obtain instances from WindowsFilesystem, never by adopting a numeric handle.
    Close is explicit or via a context manager; there is no unsafe retry/finalizer.
    A failed close may leak a kernel resource and is always reported.
    """

    def __init__(
        self,
        owner: WindowsFilesystem,
        raw: int,
        snapshot: _Snapshot,
        *,
        _key: object,
        created: bool = False,
    ) -> None:
        if _key is not _OWNED:
            raise WindowsFilesystemError("numeric handle adoption is unsupported")
        self._owner = owner
        self._raw: int | None = raw
        self._identity = snapshot.identity
        self._directory = snapshot.directory
        self._created = created
        self._disposed = False

    @property
    def identity(self) -> FileIdentity:
        """The immutable identity proved when this capability was admitted."""
        return self._identity

    def __copy__(self) -> OwnedHandle:
        raise WindowsFilesystemError("owned handles cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> OwnedHandle:
        raise WindowsFilesystemError("owned handles cannot be copied")

    def close(self) -> None:
        """Invalidate before native close; a failed close must never be retried."""
        with self._owner._lock:
            raw, self._raw = self._raw, None
            if raw is not None:
                self._owner._api.close(raw)

    def __enter__(self) -> OwnedHandle:
        with self._owner._lock:
            self._owner._live(self)
        return self

    def __exit__(self, _kind: object, error: BaseException | None, _tb: object) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if error is not None:
                raise BaseExceptionGroup(
                    "operation and close failed", [error, cleanup]
                ) from None
            raise


class WindowsFilesystem:
    """Explicit experimental primitives, never an automatically selected backend."""

    def __init__(self, *, _api: _NativeAPI | None = None) -> None:
        self._api = _api if _api is not None else _NativeAPI()
        self._lock = threading.RLock()

    def _live(self, handle: OwnedHandle) -> int:
        if handle._owner is not self or handle._raw is None or handle._disposed:
            raise WindowsFilesystemError("foreign, closed or disposed capability")
        return handle._raw

    def _verify(
        self, raw: int, directory: bool, expected: FileIdentity | None = None
    ) -> _Snapshot:
        snapshot = self._api.query(raw)
        self._verify_type(snapshot, directory)
        if snapshot.links < 1 or (not directory and snapshot.links != 1):
            raise WindowsFilesystemError("multiple or missing file links refused")
        if expected is not None and snapshot.identity != expected:
            raise WindowsFilesystemError("file identity mismatch")
        return snapshot

    @staticmethod
    def _verify_type(snapshot: _Snapshot, directory: bool) -> None:
        if not snapshot.disk or snapshot.delete_pending:
            raise WindowsFilesystemError("non-disk or delete-pending object refused")
        if snapshot.attributes & _REPARSE_POINT or snapshot.reparse_tag:
            raise WindowsFilesystemError("all reparse objects refused")
        if (
            snapshot.directory != directory
            or bool(snapshot.attributes & _DIRECTORY) != directory
        ):
            raise WindowsFilesystemError("unexpected file type")

    def _admit(
        self,
        raw: int,
        directory: bool,
        *,
        expected: FileIdentity | None = None,
        volume: int | None = None,
        created: bool = False,
    ) -> OwnedHandle:
        try:
            snapshot = self._verify(raw, directory, expected)
            if volume is not None and snapshot.identity.volume != volume:
                raise WindowsFilesystemError("volume transition refused")
            return OwnedHandle(self, raw, snapshot, _key=_OWNED, created=created)
        except BaseException as exc:
            # Unverified objects are only closed, never written or deleted.
            try:
                self._api.close(raw)
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "admission and close failed", [exc, cleanup]
                ) from None
            raise

    def open_drive_root(self, root: str, *, expected: FileIdentity) -> OwnedHandle:
        """Pin an exact drive root to an independently admitted full identity.

        No path normalization or caller-provided device namespace is accepted.
        UNC and root identity discovery/admission policy are separate work.
        """
        if not isinstance(root, str) or re.fullmatch(r"[A-Za-z]:\\", root) is None:
            raise UnsupportedGuarantee("only an exact drive root is supported")
        if not isinstance(expected, FileIdentity):
            raise WindowsFilesystemError("trusted root identity is required")
        with self._lock:
            return self._admit(self._api.open_root(root), True, expected=expected)

    def open_child(
        self,
        parent: OwnedHandle,
        name: str,
        *,
        directory: bool,
        expected: FileIdentity | None = None,
    ) -> OwnedHandle:
        """Open a single component without mutation, through its pinned parent."""
        return self._child(parent, name, directory=directory, expected=expected)

    def create_exclusive(self, parent: OwnedHandle, name: str) -> OwnedHandle:
        """Create one file, refusing collisions; NOT atomic complete publication.

        The caller owns cleanup on later failure. ACLs are inherited without any
        owner-only guarantee. No lease lifetime or delete-on-close is implied.
        """
        return self._child(parent, name, directory=False, created=True)

    def _child(
        self,
        parent: OwnedHandle,
        name: str,
        *,
        directory: bool,
        expected: FileIdentity | None = None,
        created: bool = False,
    ) -> OwnedHandle:
        _component(name)
        if type(directory) is not bool:
            raise TypeError("directory must be a bool")
        if expected is not None and not isinstance(expected, FileIdentity):
            raise TypeError("expected must be a FileIdentity")
        with self._lock:
            raw = self._live(parent)
            if not parent._directory:
                raise WindowsFilesystemError("parent must be a directory capability")
            self._verify(raw, True, parent.identity)
            child = self._api.open_relative(raw, name, directory, created)
            return self._admit(
                child,
                directory,
                expected=expected,
                volume=parent.identity.volume,
                created=created,
            )

    def _mutable(self, handle: OwnedHandle) -> int:
        raw = self._live(handle)
        if not handle._created or handle._directory:
            raise UnsupportedGuarantee("mutation requires our exclusive new file")
        self._verify(raw, False, handle.identity)
        return raw

    def write_all(self, handle: OwnedHandle, value: bytes) -> None:
        """Write every byte or raise; only newly created files may be written."""
        if type(value) is not bytes:
            raise TypeError("value must be immutable bytes")
        with self._lock:
            raw = self._mutable(handle)
            offset = 0
            while offset < len(value):
                self._verify(raw, False, handle.identity)
                chunk = value[offset : offset + 1024 * 1024]
                count = self._api.write(raw, chunk)
                if type(count) is not int or not 0 < count <= len(chunk):
                    raise WindowsFilesystemError("invalid or zero write progress")
                offset += count
            self._verify(raw, False, handle.identity)

    def flush_file(self, handle: OwnedHandle) -> None:
        """Request file-buffer flush; does not promise directory-entry durability."""
        with self._lock:
            self._api.flush(self._mutable(handle))

    def discard_created(self, handle: OwnedHandle) -> None:
        """Mark only the retained created object for deletion, never its pathname."""
        with self._lock:
            self._api.discard(self._mutable(handle))
            handle._disposed = True

    def require_durable_publication(self) -> None:
        """Refuse namespace/receipt durability before any publication side effect."""
        raise UnsupportedGuarantee("atomic durable publication is not qualified")
