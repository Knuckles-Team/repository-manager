"""Windows handle primitives behind the operation boundary and lease adapters.

Two layers live here. ``WindowsFilesystem`` hands out owned capabilities for
exclusive creation: bootstrap is limited to an exact drive root plus an
independently trusted identity, and UNC/device paths, existing-file mutation,
atomic complete publication and durable directory-entry commits are refused.
The ``descriptor_*`` and ``path_*`` functions at the end are the Windows
mechanism for ``operation_boundary`` and the lease publishers: they emulate
POSIX ``openat``/``O_NOFOLLOW`` semantics with handle-relative ``NtCreateFile``
opens that never traverse a reparse point.

Late hardlinks. A same-principal process can add a name for any file it can
open with ``FILE_WRITE_ATTRIBUTES``; that access is not governed by share
modes, so denying sharing cannot stop ``CreateHardLinkW``. A newly created file
is therefore born with a protected DACL that grants only ``READ_CONTROL`` to
OWNER RIGHTS (which also strips the owner's implicit ``WRITE_DAC``). The
creating handle keeps the access it requested at creation, every other open is
denied, and the file's ordinary inherited DACL is restored only after the link
count is re-verified to still be exactly one.

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
import os
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
_READ_CONTROL = 0x20000
_WRITE_DAC = 0x40000
# Protected DACL: OWNER RIGHTS may only read the descriptor; nobody else may
# open the object until the creator restores its inherited DACL.
_PROTECTIVE_SDDL = "D:P(A;;RC;;;OW)"
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


def _required_security(security: Any) -> Any:
    if security is None:
        raise UnsupportedGuarantee("a security descriptor binding is required")
    return security


class _NativeAPI:
    """Thin, synchronous ctypes binding; injected DLLs are a private test seam."""

    def __init__(
        self,
        *,
        kernel32: Any = None,
        ntdll: Any = None,
        last_error: Callable[[], int] | None = None,
        security: Any = None,
    ) -> None:
        if kernel32 is None or ntdll is None:
            if sys.platform != "win32":
                raise UnsupportedGuarantee("native Windows runtime required")
            kernel32 = c.WinDLL("kernel32", use_last_error=True)
            ntdll = c.WinDLL("ntdll", use_last_error=True)
            last_error = c.get_last_error
            security = _WindowsSecurity()
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
        self._security = _required_security(security)

    def _check(self, result: int, operation: str) -> None:
        if not result:
            code = self._error()
            raise WindowsFilesystemError(code, f"{operation} failed (Win32 {code})")

    def open_root(self, root: str) -> int:
        handle = self._kernel.CreateFileW(
            root,
            _READ_ATTRIBUTES | _READ_CONTROL | _SYNCHRONIZE,
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
            self._creation_descriptor(create),
            None,
        )
        handle, status = _HANDLE(), _IoStatus()
        access = _relative_access(directory, create)
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

    def _creation_descriptor(self, create: bool) -> int | None:
        """New files are born with the protective DACL; opens pass none."""
        return self._security.protective_descriptor() if create else None

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

    def inherited_dacl(self, parent: int) -> bytes:
        """The DACL an ordinary new file in ``parent`` would inherit."""
        return self._security.inherited_dacl(parent)

    def set_dacl(self, handle: int, acl: bytes) -> None:
        """Replace the object's DACL through the creating handle."""
        self._security.set_dacl(handle, acl)

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


_CREATE_ACCESS = (
    _READ_ATTRIBUTES
    | _SYNCHRONIZE
    | _GENERIC_WRITE
    | _DELETE
    | _READ_CONTROL
    | _WRITE_DAC
)


def _relative_access(directory: bool, create: bool) -> int:
    """Creation keeps WRITE_DAC for the restore; directories can read their DACL."""
    if create:
        return _CREATE_ACCESS
    if directory:
        return _READ_ATTRIBUTES | _SYNCHRONIZE | _READ_CONTROL
    return _READ_ATTRIBUTES | _SYNCHRONIZE


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
        restore_dacl: bytes | None = None,
    ) -> None:
        if _key is not _OWNED:
            raise WindowsFilesystemError("numeric handle adoption is unsupported")
        if created and restore_dacl is None:
            raise WindowsFilesystemError("a created file needs its inherited DACL")
        self._owner = owner
        self._raw: int | None = raw
        self._identity = snapshot.identity
        self._directory = snapshot.directory
        self._created = created
        self._restore_dacl = restore_dacl
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
        """Invalidate before native close; a failed close must never be retried.

        A created file keeps its protective DACL unless its link count is still
        exactly one, so a link that bypassed prevention never gains access.
        """
        with self._owner._lock:
            raw, self._raw = self._raw, None
            if raw is not None:
                self._owner._close_owned(raw, self)

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

    def _close_owned(self, raw: int, handle: OwnedHandle) -> None:
        try:
            self._release_if_created(raw, handle)
        except BaseException as error:
            self._close_after_failure(raw, error)
            raise
        self._api.close(raw)

    def _release_if_created(self, raw: int, handle: OwnedHandle) -> None:
        if handle._created and not handle._disposed:
            self._release_created(raw, handle)

    def _close_after_failure(self, raw: int, error: BaseException) -> None:
        try:
            self._api.close(raw)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "release and close failed", [error, cleanup]
            ) from None

    def _creation_dacl(self, parent: int, created: bool) -> bytes | None:
        return self._api.inherited_dacl(parent) if created else None

    def _release_created(self, raw: int, handle: OwnedHandle) -> None:
        self._verify(raw, False, handle.identity)
        if handle._restore_dacl is None:
            raise WindowsFilesystemError("created file lost its inherited DACL")
        self._api.set_dacl(raw, handle._restore_dacl)

    def _admit(
        self,
        raw: int,
        directory: bool,
        *,
        expected: FileIdentity | None = None,
        volume: int | None = None,
        created: bool = False,
        restore_dacl: bytes | None = None,
    ) -> OwnedHandle:
        try:
            snapshot = self._verify(raw, directory, expected)
            if volume is not None and snapshot.identity.volume != volume:
                raise WindowsFilesystemError("volume transition refused")
            return OwnedHandle(
                self,
                raw,
                snapshot,
                _key=_OWNED,
                created=created,
                restore_dacl=restore_dacl,
            )
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

        The file is born with the protective DACL, so no other open (including
        the one ``CreateHardLinkW`` makes) succeeds while this capability is
        live. Closing it re-verifies a single link and restores the DACL an
        ordinary file in ``parent`` inherits. The caller owns cleanup on later
        failure. No lease lifetime or delete-on-close is implied.
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
            restore_dacl = self._creation_dacl(raw, created)
            child = self._api.open_relative(raw, name, directory, created)
            return self._admit(
                child,
                directory,
                expected=expected,
                volume=parent.identity.volume,
                created=created,
                restore_dacl=restore_dacl,
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


class _GenericMapping(c.Structure):
    _fields_ = [
        ("GenericRead", _DWORD),
        ("GenericWrite", _DWORD),
        ("GenericExecute", _DWORD),
        ("GenericAll", _DWORD),
    ]


_FILE_GENERIC_MAPPING = (0x120089, 0x120116, 0x1200A0, 0x1F01FF)
_SE_FILE_OBJECT = 1
_OWNER_GROUP_DACL = 0x1 | 0x2 | 0x4
_DACL_INFORMATION = 0x4
_UNPROTECTED_DACL = 0x20000000
_SEF_DACL_AUTO_INHERIT = 0x1
_DACL_AUTO_INHERITED = 0x0400
_ABSOLUTE_DESCRIPTOR_BYTES = 64
_TOKEN_QUERY = 0x8


class _WindowsSecurity:
    """advapi32 binding for the protective-creation DACL protocol."""

    _advapi: Any
    _kernel: Any
    _last_error: Callable[[], int]

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise UnsupportedGuarantee("native Windows runtime required")
        self._last_error = c.get_last_error
        advapi = c.WinDLL("advapi32", use_last_error=True)
        kernel = c.WinDLL("kernel32", use_last_error=True)
        pointer = c.POINTER(c.c_void_p)
        signatures: tuple[tuple[Any, str, Any, list[Any]], ...] = (
            (
                advapi,
                "ConvertStringSecurityDescriptorToSecurityDescriptorW",
                c.c_int32,
                [c.c_wchar_p, _DWORD, pointer, c.POINTER(_DWORD)],
            ),
            (
                advapi,
                "GetSecurityInfo",
                _DWORD,
                [
                    _HANDLE,
                    c.c_int32,
                    _DWORD,
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                ],
            ),
            (
                advapi,
                "CreatePrivateObjectSecurityEx",
                c.c_int32,
                [
                    c.c_void_p,
                    c.c_void_p,
                    pointer,
                    c.c_void_p,
                    c.c_int32,
                    c.c_uint32,
                    _HANDLE,
                    c.POINTER(_GenericMapping),
                ],
            ),
            (
                advapi,
                "GetSecurityDescriptorDacl",
                c.c_int32,
                [c.c_void_p, c.POINTER(c.c_int32), pointer, c.POINTER(c.c_int32)],
            ),
            (advapi, "InitializeSecurityDescriptor", c.c_int32, [c.c_void_p, _DWORD]),
            (
                advapi,
                "SetSecurityDescriptorDacl",
                c.c_int32,
                [c.c_void_p, c.c_int32, c.c_void_p, c.c_int32],
            ),
            (
                advapi,
                "SetSecurityDescriptorControl",
                c.c_int32,
                [c.c_void_p, c.c_uint16, c.c_uint16],
            ),
            (
                advapi,
                "SetKernelObjectSecurity",
                c.c_int32,
                [_HANDLE, _DWORD, c.c_void_p],
            ),
            (advapi, "DestroyPrivateObjectSecurity", c.c_int32, [pointer]),
            (
                advapi,
                "OpenProcessToken",
                c.c_int32,
                [_HANDLE, _DWORD, c.POINTER(_HANDLE)],
            ),
            (kernel, "GetCurrentProcess", _HANDLE, []),
            (kernel, "LocalFree", c.c_void_p, [c.c_void_p]),
            (kernel, "CloseHandle", c.c_int32, [_HANDLE]),
        )
        for dll, name, result, arguments in signatures:
            function = getattr(dll, name)
            function.restype = result
            function.argtypes = arguments
        self._advapi = advapi
        self._kernel = kernel
        self._protective: c.c_void_p | None = None

    def _check(self, result: int, operation: str) -> None:
        if not result:
            code = self._last_error()
            raise WindowsFilesystemError(code, f"{operation} failed (Win32 {code})")

    @staticmethod
    def _check_status(code: int, operation: str) -> None:
        if code:
            raise WindowsFilesystemError(code, f"{operation} failed (Win32 {code})")

    def protective_descriptor(self) -> int:
        """A process-lifetime self-relative descriptor for new files."""
        if self._protective is None:
            descriptor = c.c_void_p()
            self._check(
                self._advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    _PROTECTIVE_SDDL, 1, c.byref(descriptor), None
                ),
                "ConvertStringSecurityDescriptorToSecurityDescriptorW",
            )
            self._protective = descriptor
        value = self._protective.value
        if not value:
            raise WindowsFilesystemError("protective descriptor unavailable")
        return value

    def inherited_dacl(self, parent: int) -> bytes:
        """Compute the DACL Windows would give an ordinary file in ``parent``."""
        parent_descriptor = c.c_void_p()
        none = c.c_void_p()
        self._check_status(
            self._advapi.GetSecurityInfo(
                parent,
                _SE_FILE_OBJECT,
                _OWNER_GROUP_DACL,
                c.byref(none),
                c.byref(none),
                c.byref(none),
                c.byref(none),
                c.byref(parent_descriptor),
            ),
            "GetSecurityInfo",
        )
        try:
            return self._derive_dacl(parent_descriptor)
        finally:
            self._kernel.LocalFree(parent_descriptor)

    def _derive_dacl(self, parent_descriptor: c.c_void_p) -> bytes:
        token = _HANDLE()
        self._check(
            self._advapi.OpenProcessToken(
                self._kernel.GetCurrentProcess(), _TOKEN_QUERY, c.byref(token)
            ),
            "OpenProcessToken",
        )
        try:
            created = c.c_void_p()
            mapping = _GenericMapping(*_FILE_GENERIC_MAPPING)
            self._check(
                self._advapi.CreatePrivateObjectSecurityEx(
                    parent_descriptor,
                    None,
                    c.byref(created),
                    None,
                    0,
                    _SEF_DACL_AUTO_INHERIT,
                    token,
                    c.byref(mapping),
                ),
                "CreatePrivateObjectSecurityEx",
            )
            try:
                return self._copy_dacl(created)
            finally:
                self._advapi.DestroyPrivateObjectSecurity(c.byref(created))
        finally:
            self._kernel.CloseHandle(token)

    def _copy_dacl(self, descriptor: c.c_void_p) -> bytes:
        present, defaulted, acl = c.c_int32(), c.c_int32(), c.c_void_p()
        self._check(
            self._advapi.GetSecurityDescriptorDacl(
                descriptor, c.byref(present), c.byref(acl), c.byref(defaulted)
            ),
            "GetSecurityDescriptorDacl",
        )
        if not present.value or not acl.value:
            raise WindowsFilesystemError("inherited DACL is absent")
        size = c.c_uint16.from_address(acl.value + 2).value
        return c.string_at(acl.value, size)

    def set_dacl(self, handle: int, acl: bytes) -> None:
        """Apply ``acl`` through the creating handle (which holds WRITE_DAC).

        ``SetKernelObjectSecurity`` writes exactly this DACL on the handle;
        unlike ``SetSecurityInfo`` it never re-reads or re-derives anything.
        """
        acl_buffer = c.create_string_buffer(acl, len(acl))
        descriptor = c.create_string_buffer(_ABSOLUTE_DESCRIPTOR_BYTES)
        self._check(
            self._advapi.InitializeSecurityDescriptor(descriptor, 1),
            "InitializeSecurityDescriptor",
        )
        self._check(
            self._advapi.SetSecurityDescriptorDacl(descriptor, 1, acl_buffer, 0),
            "SetSecurityDescriptorDacl",
        )
        self._check(
            self._advapi.SetSecurityDescriptorControl(
                descriptor, _DACL_AUTO_INHERITED, _DACL_AUTO_INHERITED
            ),
            "SetSecurityDescriptorControl",
        )
        self._check(
            self._advapi.SetKernelObjectSecurity(
                handle, _DACL_INFORMATION | _UNPROTECTED_DACL, descriptor
            ),
            "SetKernelObjectSecurity",
        )


# ---------------------------------------------------------------------------
# Descriptor layer for ``operation_boundary`` and the lease publishers.
#
# POSIX pins a directory with an ``O_NOFOLLOW`` descriptor and resolves every
# later name through ``openat``.  The Windows mechanism below gives the same
# guarantees:
#
# * every component is opened relative to its already-open parent handle with
#   ``NtCreateFile`` + ``OBJ_DONT_REPARSE`` + ``FILE_OPEN_REPARSE_POINT``, and
#   any reparse point (symlink, junction, mount point) is refused on the opened
#   handle itself, so no name is ever resolved through a link;
# * directory handles request ``FILE_LIST_DIRECTORY`` and deny delete sharing,
#   so a pinned directory -- and therefore every ancestor in a pinned chain --
#   cannot be renamed, deleted or replaced while pinned.  The pinned path
#   spelling is thus as stable as ``/proc/self/fd/<n>`` and is what children
#   (``git``) receive as their working directory;
# * a chain stays pinned while any descendant descriptor is open, even when the
#   caller closes an ancestor first, exactly like independent POSIX fds.
#
# Descriptors are ordinary CRT file descriptors (``msvcrt.open_osfhandle``), so
# ``os.fstat``/``os.read``/``os.write``/``os.fsync`` work unchanged on them.
# ---------------------------------------------------------------------------

_FILE_LIST_DIRECTORY = 0x1
_FILE_APPEND_DATA = 0x4
_FILE_GENERIC_READ = 0x120089
_FILE_GENERIC_WRITE = 0x120116
_FILE_SHARE_READ_WRITE = 0x3
_FILE_SHARE_ALL = 0x7
_FILE_OPEN_IF = 3
_FILE_OVERWRITE = 4
_FILE_OVERWRITE_IF = 5
_OPTION_DIRECTORY = 0x1
_OPTION_SYNCHRONOUS = 0x20
_OPTION_NON_DIRECTORY = 0x40
_OPTION_OPEN_REPARSE = 0x200000
_FLAG_BACKUP_SEMANTICS = 0x02000000
_FLAG_OPEN_REPARSE = 0x00200000
_OPEN_EXISTING = 3
_CREATE_NEW = 1
_CREATE_ALWAYS = 2
_OPEN_ALWAYS = 4
_TRUNCATE_EXISTING = 5
_FILE_RENAME_INFORMATION_EX = 65
_RENAME_REPLACE_IF_EXISTS = 0x1
_RENAME_POSIX_SEMANTICS = 0x2
_RENAME_IGNORE_READONLY = 0x40
_FILE_DISPOSITION_INFO_EX = 21
_DISPOSITION_DELETE = 0x1
_DISPOSITION_POSIX_SEMANTICS = 0x2
_DISPOSITION_IGNORE_READONLY = 0x10
_ERROR_CANT_RESOLVE_FILENAME = 1921
_ERROR_DIRECTORY = 267
_ERROR_INVALID_HANDLE = 6


class _DescriptorNative:
    """ctypes binding for the descriptor layer (real Windows only)."""

    kernel: Any
    nt: Any
    _last_error: Callable[[], int]

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise UnsupportedGuarantee("native Windows runtime required")
        self._last_error = c.get_last_error
        kernel = c.WinDLL("kernel32", use_last_error=True)
        nt = c.WinDLL("ntdll", use_last_error=True)
        signatures: tuple[tuple[Any, str, Any, list[Any]], ...] = (
            (
                kernel,
                "CreateFileW",
                _HANDLE,
                [c.c_wchar_p, _DWORD, _DWORD, c.c_void_p, _DWORD, _DWORD, _HANDLE],
            ),
            (kernel, "CloseHandle", c.c_int32, [_HANDLE]),
            (
                kernel,
                "GetFileInformationByHandleEx",
                c.c_int32,
                [_HANDLE, c.c_int32, c.c_void_p, _DWORD],
            ),
            (
                kernel,
                "SetFileInformationByHandle",
                c.c_int32,
                [_HANDLE, c.c_int32, c.c_void_p, _DWORD],
            ),
            (
                nt,
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
            (
                nt,
                "NtSetInformationFile",
                c.c_int32,
                [_HANDLE, c.POINTER(_IoStatus), c.c_void_p, _DWORD, c.c_int32],
            ),
            (nt, "RtlNtStatusToDosError", _DWORD, [_DWORD]),
        )
        for dll, name, result, arguments in signatures:
            function = getattr(dll, name)
            function.restype = result
            function.argtypes = arguments
        self.kernel = kernel
        self.nt = nt

    def status_error(self, status: int, operation: str, name: str) -> OSError:
        code = int(self.nt.RtlNtStatusToDosError(status & 0xFFFFFFFF))
        return OSError(
            0,
            f"{operation} failed (NTSTATUS 0x{status & 0xFFFFFFFF:08x})",
            name,
            code,
        )

    def last_error(self, operation: str, name: str) -> OSError:
        code = self._last_error()
        return OSError(0, f"{operation} failed (Win32 {code})", name, code)

    def open_relative(
        self,
        parent: int,
        name: str,
        *,
        access: int,
        share: int,
        disposition: int,
        options: int,
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
        result = self.nt.NtCreateFile(
            c.byref(handle),
            access | _SYNCHRONIZE,
            c.byref(attributes),
            c.byref(status),
            None,
            0x80,
            share,
            disposition,
            options | _OPTION_SYNCHRONOUS | _OPTION_OPEN_REPARSE,
            None,
            0,
        )
        if result != 0:
            if handle.value:
                self.kernel.CloseHandle(handle)
            raise self.status_error(result, "NtCreateFile", name)
        if not handle.value or handle.value == _INVALID_HANDLE:
            raise OSError(0, "NtCreateFile returned no handle", name, 6)
        return int(handle.value)

    def open_path(
        self, path: str, *, access: int, share: int, disposition: int, flags: int
    ) -> int:
        handle = self.kernel.CreateFileW(
            path, access, share, None, disposition, flags | _FLAG_OPEN_REPARSE, None
        )
        if handle in (None, 0, _INVALID_HANDLE):
            raise self.last_error("CreateFileW", path)
        return int(handle)

    def attributes(self, handle: int) -> int:
        value = _AttributeInfo()
        if not self.kernel.GetFileInformationByHandleEx(
            handle, 9, c.byref(value), c.sizeof(value)
        ):
            raise self.last_error("GetFileInformationByHandleEx", "")
        return int(value.FileAttributes)

    def refuse_reparse(self, handle: int, name: str, *, directory: bool | None) -> None:
        """Close ``handle`` and raise unless it is a plain (non-link) object."""
        try:
            attributes = self.attributes(handle)
        except BaseException:
            self.kernel.CloseHandle(handle)
            raise
        if attributes & _REPARSE_POINT:
            self.kernel.CloseHandle(handle)
            raise OSError(
                0, "reparse point refused", name, _ERROR_CANT_RESOLVE_FILENAME
            )
        if directory is not None and bool(attributes & _DIRECTORY) != directory:
            self.kernel.CloseHandle(handle)
            if directory:
                raise OSError(0, "not a directory", name, _ERROR_DIRECTORY)
            raise OSError(0, "is a directory", name, 5)

    def set_disposition(self, handle: int, flags: int, name: str) -> None:
        value = _DWORD(flags)
        if not self.kernel.SetFileInformationByHandle(
            handle, _FILE_DISPOSITION_INFO_EX, c.byref(value), c.sizeof(value)
        ):
            raise self.last_error("SetFileInformationByHandle", name)

    def rename(self, handle: int, directory: int, target: str) -> None:
        raw = target.encode("utf-16-le")
        size = 24 + len(raw)
        buffer = c.create_string_buffer(size)
        c.c_uint32.from_buffer(buffer, 0).value = (
            _RENAME_REPLACE_IF_EXISTS
            | _RENAME_POSIX_SEMANTICS
            | _RENAME_IGNORE_READONLY
        )
        c.c_void_p.from_buffer(buffer, 8).value = directory
        c.c_uint32.from_buffer(buffer, 16).value = len(raw)
        c.memmove(c.addressof(buffer) + 20, raw, len(raw))
        status = _IoStatus()
        result = self.nt.NtSetInformationFile(
            handle, c.byref(status), buffer, size, _FILE_RENAME_INFORMATION_EX
        )
        if result != 0:
            raise self.status_error(result, "NtSetInformationFile", target)

    def close(self, handle: int) -> None:
        self.kernel.CloseHandle(handle)


_DESCRIPTOR_NATIVE: _DescriptorNative | None = None
_DESCRIPTOR_LOCK = threading.RLock()


def _native() -> _DescriptorNative:
    global _DESCRIPTOR_NATIVE
    with _DESCRIPTOR_LOCK:
        if _DESCRIPTOR_NATIVE is None:
            _DESCRIPTOR_NATIVE = _DescriptorNative()
        return _DESCRIPTOR_NATIVE


class _PinnedNode:
    """One pinned directory handle; alive while referenced by an fd or child."""

    __slots__ = ("fd", "parent", "path", "references")

    def __init__(self, fd: int, path: str, parent: _PinnedNode | None) -> None:
        self.fd = fd
        self.path = path
        self.parent = parent
        self.references = 1
        if parent is not None:
            parent.references += 1


_PINNED: dict[int, _PinnedNode] = {}


def _release(node: _PinnedNode | None) -> None:
    while node is not None:
        node.references -= 1
        if node.references:
            return
        os.close(node.fd)
        node = node.parent


def _to_descriptor(handle: int, flags: int) -> int:
    if sys.platform != "win32":
        raise UnsupportedGuarantee("native Windows runtime required")
    import msvcrt

    try:
        return msvcrt.open_osfhandle(handle, flags | os.O_NOINHERIT)
    except BaseException:
        _native().close(handle)
        raise


def _node(fd: int) -> _PinnedNode:
    node = _PINNED.get(fd)
    if node is None:
        raise OSError(
            0, "not a pinned directory descriptor", None, _ERROR_INVALID_HANDLE
        )
    return node


def _handle(fd: int) -> int:
    if sys.platform != "win32":
        raise UnsupportedGuarantee("native Windows runtime required")
    import msvcrt

    return int(msvcrt.get_osfhandle(fd))


def _single_component(name: str) -> str:
    try:
        return _component(name)
    except WindowsFilesystemError as exc:
        raise OSError(0, str(exc), name, 123) from exc


def _register(handle: int, path: str, parent: _PinnedNode | None) -> int:
    fd = _to_descriptor(handle, os.O_RDONLY)
    with _DESCRIPTOR_LOCK:
        _PINNED[fd] = _PinnedNode(fd, path, parent)
    return fd


_DIRECTORY_ACCESS = _FILE_LIST_DIRECTORY | _READ_ATTRIBUTES | _READ_CONTROL


def descriptor_open_root(anchor: str) -> int:
    """Pin a filesystem root (``C:\\`` or a UNC share root) as a directory fd."""
    native = _native()
    handle = native.open_path(
        anchor,
        access=_DIRECTORY_ACCESS | _SYNCHRONIZE,
        share=_FILE_SHARE_READ_WRITE,
        disposition=_OPEN_EXISTING,
        flags=_FLAG_BACKUP_SEMANTICS,
    )
    native.refuse_reparse(handle, anchor, directory=True)
    return _register(handle, anchor, None)


def descriptor_open_directory(parent_fd: int, name: str) -> int:
    """``openat(parent, name, O_DIRECTORY | O_NOFOLLOW)``; ``..`` re-pins the parent."""
    with _DESCRIPTOR_LOCK:
        parent = _node(parent_fd)
        if name == "..":
            target = parent.parent or parent
            return _register_duplicate(target)
        name = _single_component(name)
        native = _native()
        handle = native.open_relative(
            _handle(parent_fd),
            name,
            access=_DIRECTORY_ACCESS,
            share=_FILE_SHARE_READ_WRITE,
            disposition=_FILE_OPEN,
            options=_OPTION_DIRECTORY,
        )
        native.refuse_reparse(handle, name, directory=True)
        return _register(handle, os.path.join(parent.path, name), parent)


def _register_duplicate(node: _PinnedNode) -> int:
    fd = os.dup(node.fd)
    _PINNED[fd] = _PinnedNode(fd, node.path, node.parent)
    return fd


def descriptor_dup(fd: int) -> int:
    """``dup`` that keeps the same pinned chain alive."""
    with _DESCRIPTOR_LOCK:
        node = _PINNED.get(fd)
        if node is None:
            return os.dup(fd)
        return _register_duplicate(node)


def descriptor_close(fd: int) -> None:
    """Close an fd; a pinned directory closes once no descendant needs it."""
    with _DESCRIPTOR_LOCK:
        node = _PINNED.pop(fd, None)
        if node is None:
            os.close(fd)
            return
        _release(node)


def descriptor_path(fd: int) -> str:
    """The pinned spelling of a directory fd (stable while it stays open)."""
    with _DESCRIPTOR_LOCK:
        return _node(fd).path


def descriptor_listdir(fd: int) -> list[str]:
    """List a pinned directory; its spelling cannot be rebound while pinned."""
    return os.listdir(descriptor_path(fd))


def descriptor_stat(parent_fd: int, name: str) -> os.stat_result:
    """``fstatat(..., AT_SYMLINK_NOFOLLOW)``; a link reports its own metadata."""
    native = _native()
    handle = native.open_relative(
        _handle(parent_fd),
        _single_component(name),
        access=_READ_ATTRIBUTES,
        share=_FILE_SHARE_ALL,
        disposition=_FILE_OPEN,
        options=0,
    )
    fd = _to_descriptor(handle, os.O_RDONLY)
    try:
        return os.fstat(fd)
    finally:
        os.close(fd)


def descriptor_mkdir(parent_fd: int, name: str) -> None:
    """``mkdirat``: create exactly one directory under the pinned parent."""
    native = _native()
    handle = native.open_relative(
        _handle(parent_fd),
        _single_component(name),
        access=_READ_ATTRIBUTES,
        share=_FILE_SHARE_ALL,
        disposition=_FILE_CREATE,
        options=_OPTION_DIRECTORY,
    )
    native.close(handle)


def _file_access(flags: int) -> tuple[int, int, int]:
    mode = flags & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR)
    if mode == os.O_RDONLY:
        return _FILE_GENERIC_READ, _FILE_SHARE_ALL, os.O_RDONLY
    if mode == os.O_WRONLY:
        return _FILE_GENERIC_WRITE | _READ_ATTRIBUTES, _SHARE_READ, os.O_WRONLY
    return _FILE_GENERIC_READ | _FILE_GENERIC_WRITE, _SHARE_READ, os.O_RDWR


def _relative_disposition(flags: int) -> int:
    create, exclusive, truncate = (
        bool(flags & os.O_CREAT),
        bool(flags & os.O_EXCL),
        bool(flags & os.O_TRUNC),
    )
    if create and exclusive:
        return _FILE_CREATE
    if create:
        return _FILE_OVERWRITE_IF if truncate else _FILE_OPEN_IF
    return _FILE_OVERWRITE if truncate else _FILE_OPEN


def descriptor_open_file(parent_fd: int, name: str, flags: int) -> int:
    """``openat(parent, name, flags | O_NOFOLLOW)`` for a regular file."""
    access, share, mode = _file_access(flags)
    native = _native()
    handle = native.open_relative(
        _handle(parent_fd),
        _single_component(name),
        access=access,
        share=share,
        disposition=_relative_disposition(flags),
        options=_OPTION_NON_DIRECTORY,
    )
    native.refuse_reparse(handle, name, directory=False)
    return _to_descriptor(handle, mode | (flags & os.O_APPEND))


def descriptor_remove(parent_fd: int, name: str, *, directory: bool) -> None:
    """``unlinkat``/``unlinkat(AT_REMOVEDIR)`` with POSIX delete semantics.

    A link is removed itself, never its target, exactly like ``unlink(2)``.
    """
    native = _native()
    handle = native.open_relative(
        _handle(parent_fd),
        _single_component(name),
        access=_DELETE | _READ_ATTRIBUTES,
        share=_FILE_SHARE_ALL,
        disposition=_FILE_OPEN,
        options=_OPTION_DIRECTORY if directory else _OPTION_NON_DIRECTORY,
    )
    try:
        native.set_disposition(
            handle,
            _DISPOSITION_DELETE
            | _DISPOSITION_POSIX_SEMANTICS
            | _DISPOSITION_IGNORE_READONLY,
            name,
        )
    finally:
        native.close(handle)


def descriptor_replace(directory_fd: int, source: str, target: str) -> None:
    """``renameat(dir, source, dir, target)`` replacing ``target`` atomically."""
    native = _native()
    directory = _handle(directory_fd)
    handle = native.open_relative(
        directory,
        _single_component(source),
        access=_DELETE | _READ_ATTRIBUTES,
        share=_FILE_SHARE_ALL,
        disposition=_FILE_OPEN,
        options=_OPTION_NON_DIRECTORY,
    )
    native.refuse_reparse(handle, source, directory=False)
    try:
        native.rename(handle, directory, _single_component(target))
    finally:
        native.close(handle)


def descriptor_fsync_directory(fd: int) -> None:
    """Flush a pinned directory's metadata through a write handle to it."""
    path = descriptor_path(fd)
    handle = _native().open_path(
        path,
        access=_GENERIC_WRITE | _READ_ATTRIBUTES,
        share=_FILE_SHARE_ALL,
        disposition=_OPEN_EXISTING,
        flags=_FLAG_BACKUP_SEMANTICS,
    )
    writer = _to_descriptor(handle, os.O_RDONLY)
    try:
        actual, expected = os.fstat(writer), os.fstat(fd)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise OSError(0, "pinned directory identity changed", path, 5)
        os.fsync(writer)
    finally:
        os.close(writer)


def _path_disposition(flags: int) -> int:
    create, exclusive, truncate = (
        bool(flags & os.O_CREAT),
        bool(flags & os.O_EXCL),
        bool(flags & os.O_TRUNC),
    )
    if create and exclusive:
        return _CREATE_NEW
    if create:
        return _CREATE_ALWAYS if truncate else _OPEN_ALWAYS
    return _TRUNCATE_EXISTING if truncate else _OPEN_EXISTING


def path_open_no_follow(path: str, flags: int) -> int:
    """``open(path, flags | O_NOFOLLOW)``: the final component is never a link."""
    access, share, mode = _file_access(flags)
    native = _native()
    handle = native.open_path(
        path,
        access=access,
        share=share,
        disposition=_path_disposition(flags),
        flags=0x80,
    )
    native.refuse_reparse(handle, path, directory=False)
    return _to_descriptor(handle, mode | (flags & os.O_APPEND))


def path_fsync(path: str, *, directory: bool) -> None:
    """Flush a regular file or directory named by ``path`` without links."""
    native = _native()
    handle = native.open_path(
        path,
        access=_GENERIC_WRITE | _READ_ATTRIBUTES,
        share=_FILE_SHARE_ALL,
        disposition=_OPEN_EXISTING,
        flags=_FLAG_BACKUP_SEMANTICS if directory else 0x80,
    )
    native.refuse_reparse(handle, path, directory=directory)
    writer = _to_descriptor(handle, os.O_RDONLY)
    try:
        os.fsync(writer)
    finally:
        os.close(writer)
