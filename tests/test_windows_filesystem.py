"""Deterministic native-call simulations, NOT Windows race qualification.

The fake kernel separates namespace edges, object IDs and reusable handle slots.
Callbacks model an interleaving at a specific native call, without sleeps. Real
Windows sharing, hardlink races, ABI, ACL and crash durability remain unqualified.
"""

import argparse
import copy
import ctypes as c
import hashlib
import importlib.util
import json
import multiprocessing
import os
import platform
import shutil
import struct
import sys
import tempfile
import time
import traceback
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from repository_manager import windows_filesystem as w

PROTECTIVE_DESCRIPTOR = 0x5D5D
INHERITED_DACL = b"inherited-dacl"


@dataclass
class Node:
    number: int
    directory: bool = False
    volume: int = 13
    attributes: int = 0
    tag: int = 0
    links: int = 1
    pending: bool = False
    disk: bool = True
    data: bytearray = field(default_factory=bytearray)
    dacl: bytes | None = None

    @property
    def identity(self):
        return w.FileIdentity(self.volume, self.number.to_bytes(16, "little"))


class Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class Kernel:
    def __init__(self):
        self.root = Node(1, directory=True)
        self.external = Node(2, data=bytearray(b"SENTINEL"))
        self.edges = {}
        self.handles = {}
        self.next_handle = 2**40
        self.next_object = 10
        self.trace = []
        self.callbacks = {}
        self.fail = set()
        self.write_counts = []
        self.status = 0
        self.information = None
        self.error = 5
        self.kernel32 = SimpleNamespace(
            **{
                name: Function(getattr(self, method))
                for name, method in {
                    "CreateFileW": "root_open",
                    "GetFileInformationByHandleEx": "query",
                    "GetFileType": "file_type",
                    "WriteFile": "write",
                    "FlushFileBuffers": "flush",
                    "SetFileInformationByHandle": "discard",
                    "CloseHandle": "close",
                }.items()
            }
        )
        self.ntdll = SimpleNamespace(NtCreateFile=Function(self.open))
        self.security = SimpleNamespace(
            protective_descriptor=lambda: PROTECTIVE_DESCRIPTOR,
            inherited_dacl=self.inherited_dacl,
            set_dacl=self.set_dacl,
        )
        self.api = w._NativeAPI(
            kernel32=self.kernel32,
            ntdll=self.ntdll,
            last_error=lambda: self.error,
            security=self.security,
        )
        self.fs = w.WindowsFilesystem(_api=self.api)

    def event(self, name, *args):
        self.trace.append((name, *args))
        callback = self.callbacks.pop(name, None)
        if callback:
            callback()

    def allocate(self, node):
        raw = self.next_handle
        self.next_handle += 1
        self.handles[raw] = node
        return raw

    def root_open(self, path, access, share, security, disposition, flags, template):
        self.event("root", path, access, share, security, disposition, flags, template)
        if "root" in self.fail:
            return w._INVALID_HANDLE
        return self.allocate(self.root)

    def open(
        self,
        out,
        access,
        attrs,
        iosb,
        allocation,
        fileattrs,
        share,
        disposition,
        options,
        ea,
        ealen,
    ):
        assert allocation is None and fileattrs == 0x80
        assert ea is None and ealen == 0
        attributes = c.cast(attrs, c.POINTER(w._ObjectAttributes)).contents
        string = attributes.ObjectName.contents
        name = c.string_at(string.Buffer, string.Length).decode("utf-16-le")
        parent = attributes.RootDirectory
        self.event(
            "open",
            parent,
            name,
            access,
            share,
            disposition,
            options,
            attributes.Attributes,
            string.Length,
            attributes.SecurityDescriptor,
        )
        status = c.cast(iosb, c.POINTER(w._IoStatus)).contents
        if self.status:
            status.Status = self.status
            return self.status
        key = (self.handles[parent].number, name)
        if disposition == w._FILE_CREATE:
            if key in self.edges:
                status.Status = c.c_int32(0xC0000035).value
                return status.Status
            self.edges[key] = Node(self.next_object)
            self.next_object += 1
        elif key not in self.edges:
            status.Status = c.c_int32(0xC0000034).value
            return status.Status
        node = self.edges[key]
        c.cast(out, c.POINTER(w._HANDLE))[0] = self.allocate(node)
        status.Information = (
            self.information if self.information is not None else disposition
        )
        return 0

    def query(self, handle, kind, out, size):
        self.event("query", handle, kind, size)
        if "query" in self.fail:
            return 0
        node = self.handles[handle]
        if kind == 18:
            result = c.cast(out, c.POINTER(w._FileIdInfo)).contents
            result.VolumeSerialNumber = node.volume
            result.FileId[:] = node.identity.file_id
        elif kind == 1:
            standard = c.cast(out, c.POINTER(w._StandardInfo)).contents
            standard.NumberOfLinks = node.links
            standard.DeletePending = node.pending
            standard.Directory = node.directory
        elif kind == 9:
            attributes = c.cast(out, c.POINTER(w._AttributeInfo)).contents
            attributes.FileAttributes = node.attributes | (
                w._DIRECTORY if node.directory else 0
            )
            attributes.ReparseTag = node.tag
        else:
            raise AssertionError(kind)
        return 1

    def file_type(self, handle):
        return 1 if self.handles[handle].disk else 3

    def write(self, handle, buffer, size, out, overlapped):
        assert overlapped is None
        value = c.string_at(buffer, size)
        self.event("write", handle, value)
        if "write" in self.fail:
            return 0
        count = self.write_counts.pop(0) if self.write_counts else size
        c.cast(out, c.POINTER(w._DWORD))[0] = count
        if 0 < count <= size:
            self.handles[handle].data.extend(value[:count])
        return 1

    def flush(self, handle):
        self.event("flush", handle)
        return int("flush" not in self.fail)

    def discard(self, handle, kind, out, size):
        self.event("discard", handle, kind, size)
        assert kind == 4
        assert c.cast(out, c.POINTER(w._DispositionInfo)).contents.DeleteFile == 1
        if "discard" in self.fail:
            return 0
        self.handles[handle].pending = True
        return 1

    def close(self, handle):
        self.event("close", handle)
        if "close_retains_handle" in self.fail:
            return 0
        node = self.handles.pop(handle)
        if node.pending:
            self.edges = {
                key: value for key, value in self.edges.items() if value is not node
            }
        return int("close" not in self.fail)

    def inherited_dacl(self, parent):
        self.event("inherit", parent)
        return INHERITED_DACL

    def set_dacl(self, handle, acl):
        self.event("dacl", handle, acl)
        if "dacl" in self.fail:
            raise w.WindowsFilesystemError(5, "SetSecurityInfo failed (Win32 5)")
        self.handles[handle].dacl = acl

    def admit_root(self):
        return self.fs.open_drive_root("C:\\", expected=self.root.identity)


@pytest.fixture
def kernel():
    value = Kernel()
    yield value
    assert not value.handles, "test leaked an owned fake kernel handle"
    assert value.external.data == b"SENTINEL"


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "a/../b",
        "a\\..\\b",
        "C:relative",
        "\\rooted",
        "name:stream",
        "name.",
        "name ",
        "NUL",
        "con.txt",
        "COM¹",
        "LPT9.log",
        "CONIN$",
        "\\\\.\\device",
        "x\0y",
        "x\ny",
        "a?",
        "\ud800",
        "x" * 256,
    ],
)
def test_invalid_component_refused_before_native_call(kernel, name):
    with kernel.admit_root() as root:
        trace = list(kernel.trace)
        with pytest.raises(w.WindowsFilesystemError):
            kernel.fs.create_exclusive(root, name)
        assert kernel.trace == trace


@pytest.mark.parametrize(
    "root",
    [
        "C:relative",
        "C:",
        "\\",
        "C:\\a",
        "C:/",
        "\\\\server\\share\\",
        "\\\\?\\C:\\",
        "\\\\.\\C:",
    ],
)
def test_root_namespace_refused_before_native_call(kernel, root):
    with pytest.raises(w.UnsupportedGuarantee):
        kernel.fs.open_drive_root(root, expected=kernel.root.identity)
    assert kernel.trace == []


def test_relative_open_and_native_flags(kernel):
    directory = Node(3, directory=True)
    kernel.edges[(1, "a")] = directory
    with kernel.admit_root() as root:
        with kernel.fs.open_child(root, "a", directory=True) as child:
            with kernel.fs.create_exclusive(child, "é space's 😀") as file:
                kernel.fs.write_all(file, b"hello")
                kernel.fs.flush_file(file)
        opens = [row for row in kernel.trace if row[0] == "open"]
        first, second = opens
        assert first[1] >= 2**40 and second[1] != first[1]
        assert first[2] == "a" and second[2] == "é space's 😀"
        assert first[4:7] == (1, 1, 0x00200021)
        assert second[4:7] == (0, 2, 0x00200060)
        assert first[7] == second[7] == 0x1040  # no OBJ_INHERIT
        assert second[8] == len(second[2].encode("utf-16-le"))
        assert second[3] == 0x40170080  # READ_CONTROL + WRITE_DAC for the restore
        assert second[9] == PROTECTIVE_DESCRIPTOR  # born locked against links
        assert first[9] is None
        assert not kernel.handles[first[1]].pending


def test_ancestor_swap_uses_pinned_object(kernel):
    original = Node(3, directory=True)
    replacement = Node(4, directory=True, attributes=w._REPARSE_POINT, tag=0xA0000003)
    kernel.edges[(1, "parent")] = original
    with kernel.admit_root() as root:
        with kernel.fs.open_child(root, "parent", directory=True) as parent:
            kernel.callbacks["open"] = lambda: kernel.edges.update(
                {(1, "parent"): replacement}
            )
            with kernel.fs.create_exclusive(parent, "child") as file:
                kernel.fs.write_all(file, b"original only")
    assert kernel.edges[(3, "child")].data == b"original only"
    assert (4, "child") not in kernel.edges


@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("tag", [0xA0000003, 0xA000000C, 0x80000099, 0])
def test_all_reparse_types_refused(kernel, directory, tag):
    kernel.edges[(1, "bad")] = Node(
        3, directory=directory, attributes=w._REPARSE_POINT, tag=tag
    )
    with kernel.admit_root() as root:
        with pytest.raises(w.WindowsFilesystemError, match="reparse"):
            kernel.fs.open_child(root, "bad", directory=directory)
        assert len(kernel.handles) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"links": 2},
        {"links": 0},
        {"disk": False},
        {"pending": True},
        {"directory": True},
        {"volume": 99},
        {"tag": 123},
    ],
)
def test_post_open_type_identity_and_links_checked(kernel, change):
    node = Node(3)
    for key, value in change.items():
        setattr(node, key, value)
    kernel.edges[(1, "file")] = node
    with kernel.admit_root() as root:
        with pytest.raises(w.WindowsFilesystemError):
            kernel.fs.open_child(root, "file", directory=False)
    assert not any(row[0] in {"write", "discard"} for row in kernel.trace)


@pytest.mark.parametrize("number,volume", [(3 + 2**64, 13), (3, 14)])
def test_full_identity_final_replacement_before_mutation(kernel, number, volume):
    expected = Node(3).identity
    kernel.edges[(1, "file")] = Node(
        number, volume=volume, data=bytearray(b"REPLACEMENT")
    )
    with kernel.admit_root() as root:
        with pytest.raises(w.WindowsFilesystemError, match="identity"):
            kernel.fs.open_child(root, "file", directory=False, expected=expected)
    assert kernel.edges[(1, "file")].data == b"REPLACEMENT"
    assert not any(row[0] in {"write", "discard"} for row in kernel.trace)


def test_root_identity_mismatch_closes_without_child_open(kernel):
    with pytest.raises(w.WindowsFilesystemError, match="identity"):
        kernel.fs.open_drive_root("C:\\", expected=Node(999, directory=True).identity)
    assert [row[0] for row in kernel.trace].count("close") == 1
    assert not any(row[0] == "open" for row in kernel.trace)


def test_exclusive_create_collision_cannot_write_or_delete_winner(kernel):
    with kernel.admit_root() as root:
        with kernel.fs.create_exclusive(root, "file") as winner:
            kernel.fs.write_all(winner, b"winner")
            with pytest.raises(w.WindowsFilesystemError, match="c0000035"):
                kernel.fs.create_exclusive(root, "file")
    assert kernel.edges[(1, "file")].data == b"winner"
    assert not any(row[0] == "discard" for row in kernel.trace)


def test_cleanup_replacement_targets_original_handle(kernel):
    with kernel.admit_root() as root:
        with kernel.fs.create_exclusive(root, "file") as file:
            original = kernel.edges[(1, "file")]
            replacement = Node(30, data=bytearray(b"REPLACEMENT"))

            def swap():
                kernel.edges[(1, "renamed")] = original
                kernel.edges[(1, "file")] = replacement

            kernel.callbacks["discard"] = swap
            kernel.fs.discard_created(file)
            with pytest.raises(w.WindowsFilesystemError):
                kernel.fs.write_all(file, b"after deletion")
    assert kernel.edges[(1, "file")] is replacement
    assert replacement.data == b"REPLACEMENT"
    assert (1, "renamed") not in kernel.edges


def test_partial_writes_submit_only_remaining_bytes(kernel):
    kernel.write_counts = [2, 3, 2]
    with kernel.admit_root() as root:
        with kernel.fs.create_exclusive(root, "file") as file:
            kernel.fs.write_all(file, b"abcdefg")
            kernel.fs.flush_file(file)
    assert [row[2] for row in kernel.trace if row[0] == "write"] == [
        b"abcdefg",
        b"cdefg",
        b"fg",
    ]
    assert kernel.edges[(1, "file")].data == b"abcdefg"


@pytest.mark.parametrize("count", [0, -1, 4])
def test_invalid_write_progress_fails_without_loop(kernel, count):
    kernel.write_counts = [count]
    with kernel.admit_root() as root:
        with kernel.fs.create_exclusive(root, "file") as file:
            with pytest.raises(w.WindowsFilesystemError, match="progress"):
                kernel.fs.write_all(file, b"abc")
            kernel.fs.discard_created(file)
    assert len([row for row in kernel.trace if row[0] == "write"]) == 1
    assert (1, "file") not in kernel.edges


@pytest.mark.parametrize("failure", ["write", "flush", "discard"])
def test_native_failures_propagate(kernel, failure):
    with kernel.admit_root() as root:
        with kernel.fs.create_exclusive(root, "file") as file:
            kernel.fail.add(failure)
            with pytest.raises(w.WindowsFilesystemError) as error:
                {
                    "write": lambda: kernel.fs.write_all(file, b"abc"),
                    "flush": lambda: kernel.fs.flush_file(file),
                    "discard": lambda: kernel.fs.discard_created(file),
                }[failure]()
            assert error.value.errno == 5


def test_close_failure_invalidates_before_numeric_handle_reuse(kernel):
    root = kernel.admit_root()
    raw = next(iter(kernel.handles))
    kernel.fail.add("close")
    with pytest.raises(w.WindowsFilesystemError, match="CloseHandle"):
        root.close()
    kernel.handles[raw] = kernel.external
    before = list(kernel.trace)
    root.close()
    with pytest.raises(w.WindowsFilesystemError, match="closed"):
        kernel.fs.create_exclusive(root, "file")
    assert kernel.trace == before
    assert kernel.handles.pop(raw) is kernel.external


def test_context_preserves_primary_and_close_error(kernel):
    root = kernel.admit_root()
    kernel.fail.add("close")
    with pytest.raises(BaseExceptionGroup) as caught:
        with root:
            raise RuntimeError("primary failure")
    assert isinstance(caught.value.exceptions[0], RuntimeError)
    assert isinstance(caught.value.exceptions[1], w.WindowsFilesystemError)


def test_handle_copy_and_foreign_backend_refused(kernel):
    with kernel.admit_root() as root:
        with pytest.raises(w.WindowsFilesystemError):
            copy.copy(root)
        with pytest.raises(w.WindowsFilesystemError):
            copy.deepcopy(root)
        other = w.WindowsFilesystem(_api=kernel.api)
        before = list(kernel.trace)
        with pytest.raises(w.WindowsFilesystemError, match="foreign"):
            other.create_exclusive(root, "file")
        assert kernel.trace == before


def test_existing_file_mutation_and_durability_stay_unsupported(kernel):
    kernel.edges[(1, "file")] = Node(3, data=bytearray(b"existing"))
    with kernel.admit_root() as root:
        with kernel.fs.open_child(root, "file", directory=False) as file:
            for operation in (
                lambda: kernel.fs.write_all(file, b"bad"),
                lambda: kernel.fs.discard_created(file),
                kernel.fs.require_durable_publication,
            ):
                before = list(kernel.trace)
                with pytest.raises(w.UnsupportedGuarantee):
                    operation()
                assert kernel.trace == before
    assert kernel.edges[(1, "file")].data == b"existing"


@pytest.mark.parametrize("status", [0xC000050B, 0xC0000022, 0xC0000043, 0x103])
def test_nt_errors_and_pending_never_become_admission(kernel, status):
    kernel.status = c.c_int32(status).value
    with kernel.admit_root() as root:
        with pytest.raises(w.WindowsFilesystemError, match="NtCreateFile"):
            kernel.fs.create_exclusive(root, "file")
        assert len(kernel.handles) == 1


def test_unexpected_disposition_closes_returned_handle(kernel):
    kernel.information = 3  # FILE_OVERWRITTEN is never acceptable
    with kernel.admit_root() as root:
        with pytest.raises(w.WindowsFilesystemError, match="disposition"):
            kernel.fs.create_exclusive(root, "file")
        assert len(kernel.handles) == 1


def test_query_failure_closes_and_retains_close_failure(kernel):
    kernel.fail.update({"query", "close"})
    with pytest.raises(BaseExceptionGroup) as caught:
        kernel.admit_root()
    assert len(caught.value.exceptions) == 2


def test_abi_fixed_width_types_and_signatures(kernel):
    assert c.sizeof(w._DWORD) == 4
    assert c.sizeof(w._FileIdInfo) == 24
    assert c.sizeof(w._StandardInfo) == 24
    assert c.sizeof(w._AttributeInfo) == 8
    assert c.sizeof(w._IoStatus) == 2 * c.sizeof(c.c_void_p)
    assert kernel.ntdll.NtCreateFile.restype is c.c_int32
    assert kernel.kernel32.CreateFileW.restype is c.c_void_p
    assert kernel.kernel32.CloseHandle.argtypes == [c.c_void_p]


def test_missing_native_entry_point_refused():
    with pytest.raises(w.UnsupportedGuarantee, match="entry point"):
        w._NativeAPI(kernel32=SimpleNamespace(), ntdll=SimpleNamespace())


def test_non_windows_runtime_refuses_without_loading_dll(monkeypatch):
    monkeypatch.setattr(w.sys, "platform", "linux")
    with pytest.raises(w.UnsupportedGuarantee, match="Windows runtime"):
        w.WindowsFilesystem()


def test_import_does_not_load_native_dlls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("module import must not load DLLs")

    monkeypatch.setattr(c, "WinDLL", forbidden, raising=False)
    spec = importlib.util.spec_from_file_location("dormant_backend_import", w.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)


def test_closed_capability_cannot_operate_on_reused_slot(kernel):
    with kernel.admit_root() as root:
        file = kernel.fs.create_exclusive(root, "file")
        raw = file._raw
        file.close()
        kernel.handles[raw] = kernel.external
        before = list(kernel.trace)
        for operation in (
            lambda: kernel.fs.write_all(file, b"bad"),
            lambda: kernel.fs.flush_file(file),
            lambda: kernel.fs.discard_created(file),
        ):
            with pytest.raises(w.WindowsFilesystemError):
                operation()
        file.close()
        assert kernel.trace == before
        assert kernel.handles.pop(raw) is kernel.external


def test_hardlink_appearing_before_write_check_refused(kernel):
    with kernel.admit_root() as root:
        file = kernel.fs.create_exclusive(root, "file")
        raw = file._raw
        kernel.edges[(1, "file")].links = 2
        with pytest.raises(w.WindowsFilesystemError, match="links"):
            kernel.fs.write_all(file, b"bad")
        # Closing re-verifies the link count: a link that bypassed prevention
        # keeps the protective DACL (no restore) while the handle still closes.
        with pytest.raises(w.WindowsFilesystemError, match="links"):
            file.close()
        assert raw not in kernel.handles
    assert not any(row[0] in {"write", "dacl"} for row in kernel.trace)


def test_created_file_restores_inherited_dacl_only_at_close(kernel):
    with kernel.admit_root() as root:
        with kernel.fs.create_exclusive(root, "file") as file:
            raw = file._raw
            kernel.fs.write_all(file, b"data")
            assert kernel.edges[(1, "file")].dacl is None
        inherit = [row for row in kernel.trace if row[0] == "inherit"]
        dacl = [row for row in kernel.trace if row[0] == "dacl"]
        assert inherit == [("inherit", root._raw)]
        assert dacl == [("dacl", raw, INHERITED_DACL)]
        assert kernel.trace.index(dacl[0]) < kernel.trace.index(("close", raw))
    assert kernel.edges[(1, "file")].dacl == INHERITED_DACL


def test_partial_write_then_error_remains_error_and_cleanup_uses_handle(kernel):
    kernel.write_counts = [2]
    with kernel.admit_root() as root:
        with kernel.fs.create_exclusive(root, "file") as file:

            def fail_next_write():
                kernel.callbacks["write"] = lambda: kernel.fail.add("write")

            kernel.callbacks["write"] = fail_next_write
            with pytest.raises(w.WindowsFilesystemError, match="WriteFile"):
                kernel.fs.write_all(file, b"abcdef")
            assert kernel.edges[(1, "file")].data == b"ab"
            kernel.fs.discard_created(file)
    assert (1, "file") not in kernel.edges
    assert not any(row[0] == "flush" for row in kernel.trace)


def test_file_flush_does_not_satisfy_namespace_durability(kernel):
    with kernel.admit_root() as root:
        with kernel.fs.create_exclusive(root, "file") as file:
            kernel.fs.write_all(file, b"complete")
            kernel.fs.flush_file(file)
            before = list(kernel.trace)
            with pytest.raises(w.UnsupportedGuarantee):
                kernel.fs.require_durable_publication()
            assert kernel.trace == before


def test_successful_flush_then_close_failure_is_visible(kernel):
    with kernel.admit_root() as root:
        file = kernel.fs.create_exclusive(root, "file")
        kernel.fs.write_all(file, b"complete")
        kernel.fs.flush_file(file)
        kernel.fail.add("close")
        with pytest.raises(w.WindowsFilesystemError, match="CloseHandle"):
            file.close()
        kernel.fail.remove("close")


@pytest.mark.parametrize("value", [None, 0, w._INVALID_HANDLE])
def test_invalid_native_handle_never_admitted_or_closed(kernel, value):
    def invalid(out, access, attrs, iosb, *args):
        c.cast(out, c.POINTER(w._HANDLE))[0] = value
        c.cast(iosb, c.POINTER(w._IoStatus)).contents.Information = 2
        return 0

    with kernel.admit_root() as root:
        kernel.ntdll.NtCreateFile.function = invalid
        with pytest.raises(w.WindowsFilesystemError, match="invalid handle"):
            kernel.fs.create_exclusive(root, "file")
        assert not any(row[0] == "close" for row in kernel.trace)


def test_nt_success_with_failed_io_status_refused(kernel):
    def failure(out, access, attrs, iosb, *args):
        c.cast(iosb, c.POINTER(w._IoStatus)).contents.Status = -1
        return 0

    with kernel.admit_root() as root:
        kernel.ntdll.NtCreateFile.function = failure
        with pytest.raises(w.WindowsFilesystemError, match="IO status 0xffffffff"):
            kernel.fs.create_exclusive(root, "file")


def test_handle_identity_changed_mid_query_refused(kernel):
    query = kernel.kernel32.GetFileInformationByHandleEx.function
    count = 0

    def changing(handle, kind, out, size):
        nonlocal count
        count += 1
        if count == 4:
            kernel.root.number += 2**64
        return query(handle, kind, out, size)

    kernel.kernel32.GetFileInformationByHandleEx.function = changing
    with pytest.raises(w.WindowsFilesystemError, match="during handle inspection"):
        kernel.admit_root()


def test_numeric_handle_adoption_and_identity_assignment_refused(kernel):
    with kernel.admit_root() as root:
        snapshot = kernel.api.query(root._raw)
        with pytest.raises(w.WindowsFilesystemError, match="adoption"):
            w.OwnedHandle(kernel.fs, root._raw, snapshot, _key=object())
        with pytest.raises(AttributeError):
            root.identity = kernel.external.identity


def test_closed_parent_refused_but_child_close_does_not_close_parent(kernel):
    root = kernel.admit_root()
    with kernel.fs.create_exclusive(root, "first"):
        pass
    assert len(kernel.handles) == 1
    root.close()
    before = list(kernel.trace)
    with pytest.raises(w.WindowsFilesystemError):
        kernel.fs.create_exclusive(root, "second")
    assert kernel.trace == before


def test_failed_close_can_leak_but_cannot_retry_or_act_on_reused_handle(kernel):
    root = kernel.admit_root()
    raw = root._raw
    kernel.fail.add("close_retains_handle")
    with pytest.raises(w.WindowsFilesystemError, match="CloseHandle") as error:
        root.close()
    assert error.value.errno == 5
    assert "resource may remain open" in error.value.__notes__[0]
    assert "Do not retry" in error.value.__notes__[0]
    assert kernel.handles[raw] is kernel.root  # the kernel resource really leaked
    assert root._raw is None  # failure invalidated the Python capability
    before = list(kernel.trace)
    root.close()
    with pytest.raises(w.WindowsFilesystemError, match="closed"):
        kernel.fs.create_exclusive(root, "bad")
    assert kernel.trace == before
    # The external kernel eventually closes the leak and recycles that slot.
    kernel.handles[raw] = kernel.external
    root.close()
    with pytest.raises(w.WindowsFilesystemError, match="closed"):
        kernel.fs.open_child(root, "bad", directory=True)
    assert kernel.trace == before
    assert kernel.handles.pop(raw) is kernel.external


# Explicit, real-Windows qualification command. These functions deliberately do
# not use pytest collection/skip markers or replace native results with fakes.
# Invoke from the checkout root using runpy (keeps that root authoritative):
# python -c "import runpy; runpy.run_path('tests/test_windows_filesystem.py', run_name='__main__')" \
#   --qualify-windows --expected-backend-sha256 <reviewed hash> \
#   --expected-tests-sha256 <reviewed hash> --output <report.json>
# A native PASS here is still NOT production/lease/namespace-durability approval.


class QualificationBlocked(RuntimeError):
    """A required runtime capability is absent; never a skipped success."""


def _require_windows_abi(system, pointer_size, wchar_size):
    if not __debug__:
        raise QualificationBlocked("qualification requires Python assertions enabled")
    if (system, pointer_size, wchar_size) != ("win32", 8, 2):
        raise QualificationBlocked("requires real 64-bit Windows with UTF-16 wchar_t")


def _qualification_sources(expected):
    test_path = Path(__file__).resolve()
    backend_path = test_path.parent.parent / "repository_manager/windows_filesystem.py"
    if Path(w.__file__).resolve() != backend_path:
        raise QualificationBlocked("imported backend is not this checkout's source")
    actual = {
        "backend": hashlib.sha256(backend_path.read_bytes()).hexdigest(),
        "tests": hashlib.sha256(test_path.read_bytes()).hexdigest(),
    }
    if actual != expected:
        raise QualificationBlocked("source hashes differ from reviewed inputs")
    return actual


def _native_error(error):
    """Record codes without publishing private temporary paths from OSError."""
    result = {
        "type": type(error).__name__,
        "errno": getattr(error, "errno", None),
        "winerror": getattr(error, "winerror", None),
    }
    if isinstance(error, w.WindowsFilesystemError):
        result["native_status"] = str(error)
    result.update(_failure_location(error))
    return result


def _failure_location(error):
    """Assertion text (built from outcome records, never temp paths) and lines."""
    frames = traceback.extract_tb(error.__traceback__)
    lines = [frame.lineno for frame in frames if frame.filename == __file__]
    details: dict[str, Any] = {"lines": lines[-6:]}
    if isinstance(error, AssertionError):
        details["assertion"] = str(error)[:2000]
    return details


class WindowsOracle:
    """Independent Win32 queries, decoded as byte buffers, not backend structs.

    Junction layout follows Microsoft's REPARSE_DATA_BUFFER mount-point member.
    The only DeviceIoControl used is FSCTL_SET_REPARSE_POINT on our own empty
    temporary directory. No privilege, ACL, mount or device configuration changes.
    """

    dll: Any
    last_error: Any

    def __init__(self):
        if sys.platform != "win32":
            raise QualificationBlocked("the independent oracle requires Windows")
        self.dll = c.WinDLL("kernel32", use_last_error=True)
        self.last_error = c.get_last_error
        signatures = {
            "CreateFileW": (
                c.c_void_p,
                [
                    c.c_wchar_p,
                    c.c_uint32,
                    c.c_uint32,
                    c.c_void_p,
                    c.c_uint32,
                    c.c_uint32,
                    c.c_void_p,
                ],
            ),
            "CloseHandle": (c.c_int32, [c.c_void_p]),
            "GetFileInformationByHandleEx": (
                c.c_int32,
                [c.c_void_p, c.c_int32, c.c_void_p, c.c_uint32],
            ),
            "GetHandleInformation": (c.c_int32, [c.c_void_p, c.POINTER(c.c_uint32)]),
            "GetVolumeInformationW": (
                c.c_int32,
                [
                    c.c_wchar_p,
                    c.c_wchar_p,
                    c.c_uint32,
                    c.POINTER(c.c_uint32),
                    c.POINTER(c.c_uint32),
                    c.POINTER(c.c_uint32),
                    c.c_wchar_p,
                    c.c_uint32,
                ],
            ),
            "DeviceIoControl": (
                c.c_int32,
                [
                    c.c_void_p,
                    c.c_uint32,
                    c.c_void_p,
                    c.c_uint32,
                    c.c_void_p,
                    c.c_uint32,
                    c.POINTER(c.c_uint32),
                    c.c_void_p,
                ],
            ),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self.dll, name)
            function.restype, function.argtypes = result, arguments

    def check(self, result):
        if not result:
            raise OSError(self.last_error(), "independent Windows oracle failed")

    @contextmanager
    def open(self, path, access=0x80):
        handle = self.dll.CreateFileW(str(path), access, 7, None, 3, 0x02200000, None)
        if handle in (None, 0, c.c_void_p(-1).value):
            self.check(0)
        try:
            yield handle
        finally:
            self.check(self.dll.CloseHandle(handle))

    def query(self, handle, kind, size):
        buffer = c.create_string_buffer(size)
        self.check(self.dll.GetFileInformationByHandleEx(handle, kind, buffer, size))
        return buffer.raw

    def identity(self, handle):
        volume, file_id = struct.unpack("<Q16s", self.query(handle, 18, 24))
        return w.FileIdentity(volume, file_id)

    def path_identity(self, path):
        with self.open(path) as handle:
            return self.identity(handle)

    def properties(self, handle):
        standard = self.query(handle, 1, 24)
        attributes, tag = struct.unpack("<II", self.query(handle, 9, 8))
        flags = c.c_uint32()
        self.check(self.dll.GetHandleInformation(handle, c.byref(flags)))
        return {
            "links": struct.unpack_from("<I", standard, 16)[0],
            "delete_pending": bool(standard[20]),
            "directory": bool(standard[21]),
            "attributes": attributes,
            "tag": tag,
            "inheritable": bool(flags.value & 1),
        }

    def filesystem(self, path):
        name, filesystem = c.create_unicode_buffer(261), c.create_unicode_buffer(261)
        serial, maximum, flags = c.c_uint32(), c.c_uint32(), c.c_uint32()
        self.check(
            self.dll.GetVolumeInformationW(
                Path(path).anchor,
                name,
                261,
                c.byref(serial),
                c.byref(maximum),
                c.byref(flags),
                filesystem,
                261,
            )
        )
        return filesystem.value

    def junction(self, link, target):
        link.mkdir()
        substitute = ("\\??\\" + str(target)).encode("utf-16-le")
        display = str(target).encode("utf-16-le")
        names = substitute + b"\0\0" + display + b"\0\0"
        data = (
            struct.pack(
                "<IHHHHHH",
                0xA0000003,
                8 + len(names),
                0,
                0,
                len(substitute),
                len(substitute) + 2,
                len(display),
            )
            + names
        )
        returned = c.c_uint32()
        with self.open(link, 0x40000000) as handle:
            buffer = c.create_string_buffer(data)
            self.check(
                self.dll.DeviceIoControl(
                    handle,
                    0x000900A4,
                    buffer,
                    len(data),
                    None,
                    0,
                    c.byref(returned),
                    None,
                )
            )


class TracedWindowsAPI(w._NativeAPI):
    """All calls are real; callbacks only place a barrier before a native call."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.owned = set()
        self.before_write = None
        self.before_discard = None

    def open_root(self, root):
        raw = super().open_root(root)
        self.owned.add(raw)
        self.events.append({"call": "root", "handle": raw})
        return raw

    def open_relative(self, parent, name, directory, create):
        raw = super().open_relative(parent, name, directory, create)
        self.owned.add(raw)
        self.events.append(
            {
                "call": "relative",
                "parent": parent,
                "name": name,
                "handle": raw,
                "directory": directory,
                "create": create,
            }
        )
        return raw

    def query(self, raw):
        snapshot = super().query(raw)
        self.events.append(
            {
                "call": "query",
                "handle": raw,
                "volume": snapshot.identity.volume,
                "file_id": snapshot.identity.file_id.hex(),
                "links": snapshot.links,
                "tag": snapshot.reparse_tag,
            }
        )
        return snapshot

    def write(self, raw, value):
        callback, self.before_write = self.before_write, None
        if callback is not None:
            callback()
        count = super().write(raw, value)
        self.events.append({"call": "write", "handle": raw, "bytes": count})
        return count

    def flush(self, raw):
        super().flush(raw)
        self.events.append({"call": "flush", "handle": raw})

    def discard(self, raw):
        callback, self.before_discard = self.before_discard, None
        if callback is not None:
            callback()
        super().discard(raw)
        self.events.append({"call": "discard", "handle": raw})

    def close(self, raw):
        super().close(raw)
        self.owned.discard(raw)
        self.events.append({"call": "close", "handle": raw})


@contextmanager
def _native_pin(fs, oracle, path):
    path = Path(path)
    with ExitStack() as stack:
        current = stack.enter_context(
            fs.open_drive_root(path.anchor, expected=oracle.path_identity(path.anchor))
        )
        for part in path.parts[1:]:
            current = stack.enter_context(fs.open_child(current, part, directory=True))
        yield current


class NativeFixture:
    def __init__(self, expected):
        self.expected = expected
        self.oracle = WindowsOracle()
        self.api = TracedWindowsAPI()
        self.fs = w.WindowsFilesystem(_api=self.api)
        self.base = Path(tempfile.mkdtemp(prefix="rm native é's "))
        self.workspace = self.base / "workspace"
        self.external = self.base / "outside"
        self.workspace.mkdir()
        self.external.mkdir()
        self.sentinel = self.external / "sentinel.bin"
        self.sentinel.write_bytes(b"EXTERNAL-SENTINEL")
        self.redirects = []
        self.observations = {}

    def preflight(self):
        filesystem = self.oracle.filesystem(self.base)
        if filesystem != "NTFS":
            raise QualificationBlocked("this bounded qualification requires local NTFS")
        self.observations["filesystem"] = filesystem
        self.observations["sentinel_before"] = hashlib.sha256(
            self.sentinel.read_bytes()
        ).hexdigest()

    def pin(self, path=None):
        return _native_pin(self.fs, self.oracle, path or self.workspace)

    def finish(self):
        self.observations["native_trace"] = self.api.events
        self.observations["cleanup_policy"] = (
            "preserve fixture on sentinel or handle failure"
        )
        assert self.sentinel.read_bytes() == b"EXTERNAL-SENTINEL", (
            "external sentinel changed"
        )
        assert not self.api.owned, "backend has unresolved owned native handles"
        self.observations["sentinel_after"] = hashlib.sha256(
            self.sentinel.read_bytes()
        ).hexdigest()
        self.observations["native_trace"] = self.api.events
        for path, kind in reversed(self.redirects):
            if kind == "junction":
                os.rmdir(path)
            else:
                os.unlink(path)
        # Every created redirect has been removed explicitly, without traversal.
        shutil.rmtree(self.base)

    def redirect(self, name, kind):
        path = self.workspace / name
        if kind == "junction":
            self.oracle.junction(path, self.external)
        else:
            try:
                os.symlink(self.external, path, target_is_directory=True)
            except OSError as error:
                raise QualificationBlocked(
                    "directory symlink unavailable with existing rights"
                ) from error
        self.redirects.append((path, kind))
        return path


def _mutation_swap_parent(payload):
    source, target = Path(payload["source"]), Path(payload["target"])
    source.rename(target)
    try:
        WindowsOracle().junction(source, Path(payload["external"]))
    except OSError as error:
        raise RuntimeError(
            "rename succeeded but junction installation failed"
        ) from error
    return {"outcome": "swapped"}


def _mutation_replace_file(payload):
    source = Path(payload["source"])
    replacement = source.with_name("replacement.bin")
    replacement.write_bytes(b"REPLACEMENT")
    os.replace(replacement, source)
    return {"outcome": "swapped"}


def _mutation_replace_owned(payload):
    source = Path(payload["source"])
    source.rename(Path(payload["target"]))
    try:
        source.write_bytes(b"REPLACEMENT")
    except OSError as error:
        raise RuntimeError(
            "rename succeeded but replacement creation failed"
        ) from error
    return {"outcome": "swapped"}


def _mutation_hardlink(payload):
    os.link(payload["source"], payload["target"])
    return {"outcome": "linked"}


def _mutation_write_open(payload):
    with open(payload["source"], "ab") as stream:
        stream.write(b"ATTACK")
    return {"outcome": "wrote"}


def _contender_create(fs, parent):
    # Only failure of this FILE_CREATE attempt counts as lost contention.
    try:
        return fs.create_exclusive(parent, "contended.bin"), None
    except w.WindowsFilesystemError as error:
        if "c0000035" in str(error) or "c0000043" in str(error):
            return None, {"outcome": "refused", "error": _native_error(error)}
        raise


def _mutation_contender(payload, release):
    api = TracedWindowsAPI()
    fs, oracle = w.WindowsFilesystem(_api=api), WindowsOracle()
    try:
        with _native_pin(fs, oracle, payload["workspace"]) as parent:
            file, refusal = _contender_create(fs, parent)
            if refusal is not None:
                return refusal
            with file:
                fs.write_all(file, payload["bytes"])
                fs.flush_file(file)
                # The parent needs the result before releasing this winner.
                payload["queue"].put(
                    {"outcome": "winner", "bytes": payload["bytes"].decode()}
                )
                if not release.wait(30):
                    raise TimeoutError("winner release barrier timed out")
    finally:
        if api.owned:
            raise AssertionError("contender leaked native handles")
    return None


def _windows_worker(action, payload, ready, start, release, results):
    try:
        _require_windows_abi(sys.platform, c.sizeof(c.c_void_p), c.sizeof(c.c_wchar))
        _qualification_sources(payload["expected"])
        ready.set()
        if not start.wait(30):
            raise TimeoutError("helper start barrier timed out")
        if action == "contend":
            outcome = _mutation_contender({**payload, "queue": results}, release)
        else:
            handlers = {
                "swap_parent": _mutation_swap_parent,
                "replace_file": _mutation_replace_file,
                "replace_owned": _mutation_replace_owned,
                "hardlink": _mutation_hardlink,
                "write_open": _mutation_write_open,
            }
            outcome = handlers[action](payload)
        if outcome is not None:
            results.put(outcome)
    except OSError as error:
        results.put({"outcome": "os_error", "error": _native_error(error)})
        if action == "contend":
            raise
    except Exception as error:
        results.put({"outcome": "error", "error": _native_error(error)})
        raise


@contextmanager
def _native_peers(fixture, actions):
    context = multiprocessing.get_context("spawn")
    start, release, results = context.Event(), context.Event(), context.Queue()
    processes, ready_events = [], []
    try:
        for action, payload in actions:
            ready = context.Event()
            process = context.Process(
                target=_windows_worker,
                args=(
                    action,
                    {**payload, "expected": fixture.expected},
                    ready,
                    start,
                    release,
                    results,
                ),
            )
            process.start()
            processes.append(process)
            ready_events.append(ready)
        for ready in ready_events:
            if not ready.wait(30):
                raise QualificationBlocked(
                    "native helper failed to reach ready barrier"
                )
        yield start, results
    finally:
        start.set()
        release.set()
        try:
            _reap_native_peers(processes, fixture)
        finally:
            results.close()
            results.join_thread()


def _reap_native_peers(processes, fixture):
    observations = []
    for process in processes:
        process.join(10)
        forced = process.is_alive()
        if forced:
            process.terminate()  # Only the exact helper spawned by this harness.
            process.join(5)
        observations.append({"exitcode": process.exitcode, "forced": forced})
        if process.is_alive():
            process.kill()
            process.join(5)
        process.close()
    fixture.observations.setdefault("children", []).extend(observations)
    assert all(row == {"exitcode": 0, "forced": False} for row in observations), (
        "helper did not exit cleanly"
    )


def _trigger_peer(start, results):
    start.set()
    return results.get(timeout=30)


def _native_attack_control(fixture, action, payload, expected):
    # Repeat the same child-process attack without retained protection. An
    # access/privilege/environment failure here must never qualify prevention.
    assert not fixture.api.owned, "control requires all backend handles released"
    with _native_peers(fixture, [(action, payload)]) as (start, results):
        outcome = _trigger_peer(start, results)
        fixture.observations.setdefault("unprotected_controls", []).append(
            {"action": action, **outcome}
        )
        assert outcome["outcome"] == expected, outcome
    return outcome


def _assert_prevented(outcome):
    assert outcome["outcome"] == "os_error", outcome
    error = outcome["error"]
    assert error["winerror"] in {5, 32, 33}, outcome


def _native_relative(fixture):
    path = fixture.workspace / "é space's 😀.bin"
    with fixture.pin() as parent:
        previous = Path.cwd()
        try:
            os.chdir(fixture.external)
            with fixture.fs.create_exclusive(parent, path.name) as file:
                assert fixture.oracle.identity(file._raw) == file.identity
                fixture.fs.write_all(file, b"native relative")
                fixture.fs.flush_file(file)
                identity = file.identity
        finally:
            os.chdir(previous)
    assert path.read_bytes() == b"native relative"
    assert fixture.oracle.path_identity(path) == identity
    return {"identity": {"volume": identity.volume, "file_id": identity.file_id.hex()}}


def _native_namespaces(fixture):
    with fixture.pin() as parent:
        for name in (
            "..",
            "a/../b",
            "a\\..\\b",
            "C:relative",
            "name:stream",
            "name.",
            "name ",
            "NUL",
            "CON",
            "x\0y",
            "\\\\.\\device",
        ):
            before = len(fixture.api.events)
            with pytest.raises(w.WindowsFilesystemError):
                fixture.fs.create_exclusive(parent, name)
            assert len(fixture.api.events) == before
        with pytest.raises(w.UnsupportedGuarantee):
            fixture.fs.open_drive_root(
                "\\\\unapproved\\share\\", expected=parent.identity
            )
    return {
        "unc": "UNSUPPORTED_REFUSAL_VERIFIED",
        "ambiguous_names": "refused before native call",
    }


def _native_reparses(fixture):
    for kind in ("junction", "symlink"):
        path = fixture.redirect(kind, kind)
        with fixture.pin() as parent:
            with pytest.raises(w.WindowsFilesystemError):
                fixture.fs.open_child(parent, path.name, directory=True)
        with pytest.raises(w.WindowsFilesystemError):
            with fixture.pin(path):
                raise AssertionError("redirected ancestor admitted")
    return {
        "junction_and_directory_symlink": "refused",
        "unknown_reparse_tags": "MOCK_ONLY",
    }


def _native_parent_swap(fixture):
    source = fixture.workspace / "parent"
    source.mkdir()
    moved = fixture.workspace / "original-parent"
    payload = {
        "source": str(source),
        "target": str(moved),
        "external": str(fixture.external),
    }
    with fixture.pin(source) as parent:
        original = parent.identity
        with _native_peers(fixture, [("swap_parent", payload)]) as (start, results):
            outcome = _trigger_peer(start, results)
            fixture.observations["swap"] = outcome
            if outcome["outcome"] == "swapped":
                fixture.redirects.append((source, "junction"))
            else:
                _assert_prevented(outcome)
            with fixture.fs.create_exclusive(parent, "child.bin") as file:
                fixture.fs.write_all(file, b"ORIGINAL")
    retained = moved if outcome["outcome"] == "swapped" else source
    assert fixture.oracle.path_identity(retained) == original
    assert (retained / "child.bin").read_bytes() == b"ORIGINAL"
    assert not (fixture.external / "child.bin").exists()
    control = fixture.workspace / "control-parent"
    control.mkdir()
    control_identity = fixture.oracle.path_identity(control)
    control_moved = fixture.workspace / "control-original"
    _native_attack_control(
        fixture,
        "swap_parent",
        {
            "source": str(control),
            "target": str(control_moved),
            "external": str(fixture.external),
        },
        "swapped",
    )
    fixture.redirects.append((control, "junction"))
    assert fixture.oracle.path_identity(control_moved) == control_identity
    assert (control / "sentinel.bin").read_bytes() == b"EXTERNAL-SENTINEL"
    return {"swap": outcome["outcome"], "original_object_preserved": True}


def _native_final_swap(fixture):
    source = fixture.workspace / "final.bin"
    source.write_bytes(b"ORIGINAL")
    expected = fixture.oracle.path_identity(source)
    with fixture.pin() as parent:
        with _native_peers(fixture, [("replace_file", {"source": str(source)})]) as (
            start,
            results,
        ):
            outcome = _trigger_peer(start, results)
            assert outcome["outcome"] == "swapped", outcome
            with pytest.raises(w.WindowsFilesystemError, match="identity"):
                fixture.fs.open_child(
                    parent, source.name, directory=False, expected=expected
                )
    assert source.read_bytes() == b"REPLACEMENT"
    source.unlink()
    try:
        os.symlink(fixture.sentinel, source)
    except OSError as error:
        raise QualificationBlocked(
            "file symlink unavailable with existing rights"
        ) from error
    fixture.redirects.append((source, "symlink"))
    with fixture.pin() as parent:
        with pytest.raises(w.WindowsFilesystemError):
            fixture.fs.open_child(parent, source.name, directory=False)
    return {"ordinary_swap": "identity refusal", "final_symlink": "reparse refusal"}


def _native_hardlinks(fixture):
    existing = fixture.workspace / "existing-link.bin"
    os.link(fixture.sentinel, existing)
    with fixture.pin() as parent:
        with pytest.raises(w.WindowsFilesystemError, match="links"):
            fixture.fs.open_child(parent, existing.name, directory=False)
        with fixture.fs.create_exclusive(parent, "new.bin") as file:
            payload = {
                "source": str(fixture.workspace / "new.bin"),
                "target": str(fixture.external / "late-link.bin"),
            }
            with _native_peers(fixture, [("hardlink", payload)]) as (start, results):

                def after_last_check():
                    outcome = _trigger_peer(start, results)
                    fixture.observations["post_check_hardlink"] = outcome
                    _assert_prevented(outcome)

                fixture.api.before_write = after_last_check
                fixture.fs.write_all(file, b"SAFE")
    assert not (fixture.external / "late-link.bin").exists()
    _native_attack_control(fixture, "hardlink", payload, "linked")
    assert fixture.oracle.path_identity(
        Path(payload["source"])
    ) == fixture.oracle.path_identity(Path(payload["target"]))
    assert Path(payload["target"]).read_bytes() == b"SAFE"
    return {"existing_link": "refused", "post_check_link": "native prevention observed"}


def _native_lifetime(fixture):
    slots = []
    with fixture.pin() as parent:
        for index in range(32):
            file = fixture.fs.create_exclusive(parent, f"lifetime-{index}.bin")
            slots.append(file._raw)
            file.close()
            before = len(fixture.api.events)
            with pytest.raises(w.WindowsFilesystemError, match="closed"):
                fixture.fs.write_all(file, b"bad")
            file.close()
            assert len(fixture.api.events) == before
        assert fixture.oracle.identity(parent._raw) == parent.identity
    return {
        "closed_capabilities": 32,
        "numeric_reuse_observed": len(slots) - len(set(slots)),
        "uncertain_close_failure": "MOCK_ONLY; no unsafe native double-close injection",
    }


def _native_contenders(fixture):
    actions = [
        ("contend", {"workspace": str(fixture.workspace), "bytes": value})
        for value in (b"contender-0", b"contender-1")
    ]
    with _native_peers(fixture, actions) as (start, results):
        start.set()
        outcomes = [results.get(timeout=30), results.get(timeout=30)]
        winners = [row for row in outcomes if row["outcome"] == "winner"]
        assert len(winners) == 1, outcomes
        assert sum(row["outcome"] == "refused" for row in outcomes) == 1, outcomes
    assert (fixture.workspace / "contended.bin").read_bytes() == winners[0][
        "bytes"
    ].encode()
    source = fixture.workspace / "contended.bin"
    before_bytes = source.read_bytes()
    before_identity = fixture.oracle.path_identity(source)
    # All child processes and their winner handles are now gone. FILE_OPEN_IF
    # could have looked exclusive solely because of sharing while they ran.
    with fixture.pin() as parent:
        with pytest.raises(w.WindowsFilesystemError, match="c0000035") as caught:
            with fixture.fs.create_exclusive(parent, source.name):
                raise AssertionError("existing file was admitted after winner closed")
    assert source.read_bytes() == before_bytes
    assert fixture.oracle.path_identity(source) == before_identity
    return {
        "contenders": outcomes,
        "closed_winner_collision": _native_error(caught.value),
        "atomic_complete_publication": "UNSUPPORTED",
    }


def _native_payload(fixture):
    payload = b"a" * (1024 * 1024) + "é\0😀".encode()
    with fixture.pin() as parent:
        with fixture.fs.create_exclusive(parent, "payload.bin") as file:
            fixture.fs.write_all(file, payload)
            fixture.fs.flush_file(file)
    assert (fixture.workspace / "payload.bin").read_bytes() == payload
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "partial_write_faults": "MOCK_ONLY; real successful byte transfer exercised",
    }


def _native_error_paths(fixture):
    source = fixture.workspace / "readonly-handle.bin"
    source.write_bytes(b"UNCHANGED")
    errors = []
    with fixture.pin() as parent:
        with fixture.fs.open_child(parent, source.name, directory=False) as file:
            # Deliberately test the binding with our borrowed READ_ATTRIBUTES-only
            # handle. No external close, permission change or fake return values.
            operations = (
                lambda: fixture.api.write(file._raw, b"bad"),
                lambda: fixture.api.flush(file._raw),
                lambda: fixture.api.discard(file._raw),
            )
            for operation in operations:
                with pytest.raises(w.WindowsFilesystemError) as caught:
                    operation()
                assert caught.value.errno == 5, "expected genuine access-denied failure"
                errors.append(_native_error(caught.value))
    assert source.read_bytes() == b"UNCHANGED"
    return {
        "real_access_denied": errors,
        "uncertain_close_and_combined_cleanup_failures": "MOCK_ONLY",
    }


def _native_cleanup_swap(fixture):
    source = fixture.workspace / "created.bin"
    moved = fixture.workspace / "owned-original.bin"
    payload = {"source": str(source), "target": str(moved)}
    with fixture.pin() as parent:
        with fixture.fs.create_exclusive(parent, source.name) as file:
            fixture.fs.write_all(file, b"OWNED")
            with _native_peers(fixture, [("replace_owned", payload)]) as (
                start,
                results,
            ):

                def after_identity_check():
                    outcome = _trigger_peer(start, results)
                    fixture.observations["cleanup_swap"] = outcome
                    if outcome["outcome"] != "swapped":
                        _assert_prevented(outcome)

                fixture.api.before_discard = after_identity_check
                fixture.fs.discard_created(file)
    outcome = fixture.observations["cleanup_swap"]
    if outcome["outcome"] == "swapped":
        assert source.read_bytes() == b"REPLACEMENT"
        assert not moved.exists()
    else:
        assert not source.exists()
    control = fixture.workspace / "control-created.bin"
    control_moved = fixture.workspace / "control-owned-original.bin"
    with fixture.pin() as parent:
        with fixture.fs.create_exclusive(parent, control.name) as file:
            fixture.fs.write_all(file, b"CONTROL-OWNED")
    control_identity = fixture.oracle.path_identity(control)
    _native_attack_control(
        fixture,
        "replace_owned",
        {"source": str(control), "target": str(control_moved)},
        "swapped",
    )
    assert control.read_bytes() == b"REPLACEMENT"
    assert control_moved.read_bytes() == b"CONTROL-OWNED"
    assert fixture.oracle.path_identity(control_moved) == control_identity
    return {"swap": outcome["outcome"], "owned_object_removed": True}


def _native_unsupported_publication(fixture):
    before = list(fixture.workspace.iterdir())
    with pytest.raises(w.UnsupportedGuarantee):
        fixture.fs.require_durable_publication()
    assert list(fixture.workspace.iterdir()) == before
    assert not fixture.api.events
    return {
        "status": "UNSUPPORTED_REFUSAL_VERIFIED",
        "atomic_complete_publication": False,
    }


def _native_types_sharing(fixture):
    with fixture.pin() as parent:
        assert fixture.oracle.properties(parent._raw)["directory"]
        with fixture.fs.create_exclusive(parent, "private-handle.bin") as file:
            properties = fixture.oracle.properties(file._raw)
            assert properties == {
                "links": 1,
                "delete_pending": False,
                "directory": False,
                "attributes": properties["attributes"],
                "tag": 0,
                "inheritable": False,
            }
            payload = {"source": str(fixture.workspace / "private-handle.bin")}
            with _native_peers(fixture, [("write_open", payload)]) as (start, results):
                outcome = _trigger_peer(start, results)
                _assert_prevented(outcome)
    _native_attack_control(fixture, "write_open", payload, "wrote")
    assert Path(payload["source"]).read_bytes() == b"ATTACK"
    return {
        "same_principal_write": outcome,
        "owner_only_acl": "UNSUPPORTED_NOT_QUALIFIED",
        "second_principal_acl": "NOT_EXECUTED",
    }


def _native_abi(fixture):
    assert c.sizeof(w._ObjectAttributes) == 48
    assert w._ObjectAttributes.RootDirectory.offset == 8
    assert w._ObjectAttributes.ObjectName.offset == 16
    assert c.sizeof(w._UnicodeString) == 16
    assert c.sizeof(w._IoStatus) == 16
    assert w._IoStatus.Information.offset == 8
    assert c.sizeof(w._FileIdInfo) == 24
    result = _native_relative(fixture)
    result["architecture"] = platform.machine()
    return result


def _native_durability(fixture):
    _native_payload(fixture)
    with pytest.raises(w.UnsupportedGuarantee):
        fixture.fs.require_durable_publication()
    return {
        "status": "UNSUPPORTED_REFUSAL_VERIFIED",
        "file_flush": "native success",
        "directory_entry_or_power_loss_durability": "NOT_QUALIFIED",
    }


def _native_no_activation(fixture):
    assert _qualification_sources(fixture.expected) == fixture.expected
    assert not fixture.api.events
    return {
        "explicit_backend_only": True,
        "production_adapters": "NOT_EXERCISED; source isolation reviewed separately",
        "activation_qualified": False,
    }


_NATIVE_QUALIFICATION_CASES = (
    ("W01_component_relative", _native_relative),
    ("W02_namespace_admission", _native_namespaces),
    ("W03_reparse_refusal", _native_reparses),
    ("W04_parent_swap", _native_parent_swap),
    ("W05_final_swap_before_truncate", _native_final_swap),
    ("W06_hardlinks", _native_hardlinks),
    ("W07_handle_lifetime", _native_lifetime),
    ("W08_exclusive_creation", _native_contenders),
    ("W09_partial_write", _native_payload),
    ("W10_flush_and_close_failures", _native_error_paths),
    ("W11_cleanup_replacement", _native_cleanup_swap),
    ("W12_complete_publication", _native_unsupported_publication),
    ("W13_type_acl_and_share_contract", _native_types_sharing),
    ("W14_binding_abi_and_errors", _native_abi),
    ("W15_unsupported_durability", _native_durability),
    ("W16_no_activation", _native_no_activation),
)


def _run_native_case(name, function, expected):
    result = {"case": name, "execution": "real Windows API, no injected return values"}
    fixture = None
    try:
        fixture = NativeFixture(expected)
        fixture.preflight()
        result["observations"] = function(fixture)
        result["result"] = "PASS_BOUNDED_CASE_ONLY"
    except QualificationBlocked as error:
        result.update(result="BLOCKED", error=_native_error(error))
    except Exception as error:
        result.update(result="FAIL", error=_native_error(error))
    finally:
        if fixture is not None:
            try:
                fixture.finish()
            except Exception as error:
                result.update(result="FAIL", cleanup_error=_native_error(error))
            result["fixture"] = fixture.observations
    return result


def windows_qualification_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Explicit dormant Windows qualification, never activation"
    )
    parser.add_argument("--qualify-windows", action="store_true", required=True)
    parser.add_argument("--expected-backend-sha256", required=True)
    parser.add_argument("--expected-tests-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, choices=range(1, 21), default=3)
    args = parser.parse_args(argv)
    expected = {
        "backend": args.expected_backend_sha256,
        "tests": args.expected_tests_sha256,
    }
    report: dict[str, Any] = {
        "status": "BLOCKED",
        "activation_qualified": False,
        "source": expected,
        "system": sys.platform,
        "python": sys.version,
        "architecture": platform.machine(),
        "windows_build": platform.version(),
        "cases": [],
        "limits": [
            "No owner-only ACL, atomic publication, UNC or namespace durability qualification.",
            "Partial IO/uncertain close fault injection remains separately mocked.",
            "No workflow changes, privilege changes, package installs or native builds.",
        ],
    }
    try:
        _qualification_sources(expected)
        _require_windows_abi(sys.platform, c.sizeof(c.c_void_p), c.sizeof(c.c_wchar))
    except QualificationBlocked as error:
        report["reason"] = str(error)
        _write_qualification_report(args.output, report)
        return 2
    deadline = time.monotonic() + 600
    for repetition in range(args.repetitions):
        for name, function in _NATIVE_QUALIFICATION_CASES:
            if time.monotonic() >= deadline:
                report["reason"] = "qualification deadline exceeded"
                _write_qualification_report(args.output, report)
                return 2
            result = _run_native_case(name, function, expected)
            result["repetition"] = repetition
            report["cases"].append(result)
            _write_qualification_report(args.output, report)
    passed = all(row["result"] == "PASS_BOUNDED_CASE_ONLY" for row in report["cases"])
    report["status"] = (
        "BOUNDED_CASES_PASSED_NOT_ACTIVATION" if passed else "BLOCKED_OR_FAILED"
    )
    _write_qualification_report(args.output, report)
    return 0 if passed else 1


def _write_qualification_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    "system,pointer,wchar",
    [("linux", 8, 4), ("darwin", 8, 4), ("win32", 4, 2), ("win32", 8, 4)],
)
def test_qualification_rejects_unqualified_platform_shapes(system, pointer, wchar):
    with pytest.raises(QualificationBlocked):
        _require_windows_abi(system, pointer, wchar)


@pytest.mark.parametrize("status", ["c0000035", "c0000043", "c0000022"])
def test_contender_classification_is_limited_to_create_status(status):
    error = w.WindowsFilesystemError(status)

    def create(parent, name):
        assert parent == "parent" and name == "contended.bin"
        raise error

    fs = SimpleNamespace(create_exclusive=create)
    if status == "c0000022":
        with pytest.raises(w.WindowsFilesystemError) as caught:
            _contender_create(fs, "parent")
        assert caught.value is error
    else:
        file, refusal = _contender_create(fs, "parent")
        assert file is None and refusal["outcome"] == "refused"


@pytest.mark.parametrize("outcome", ["swapped", "os_error", "error"])
def test_attack_control_requires_success_not_denial(monkeypatch, outcome):
    fixture = SimpleNamespace(api=SimpleNamespace(owned=set()), observations={})

    @contextmanager
    def peers(actual_fixture, actions):
        assert actual_fixture is fixture and actions == [("swap_parent", {})]
        yield None, None

    def trigger(start, results):
        assert start is None and results is None
        return {"outcome": outcome}

    monkeypatch.setattr(sys.modules[__name__], "_native_peers", peers)
    monkeypatch.setattr(sys.modules[__name__], "_trigger_peer", trigger)
    if outcome == "swapped":
        assert _native_attack_control(fixture, "swap_parent", {}, "swapped") == {
            "outcome": "swapped"
        }
    else:
        with pytest.raises(AssertionError):
            _native_attack_control(fixture, "swap_parent", {}, "swapped")


if __name__ == "__main__":
    raise SystemExit(windows_qualification_main())
