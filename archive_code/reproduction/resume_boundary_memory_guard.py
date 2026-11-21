"""Resume frozen boundary calculations after primary XJTU releases large MATs."""

import argparse
import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
from pathlib import Path
import time

import supplement_voltage_boundaries as frozen


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("memory_load", wintypes.DWORD),
                ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                ("available_extended", ctypes.c_ulonglong)]


def available_gib():
    state = MemoryStatus()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise ctypes.WinError()
    return state.available_physical / 1024 ** 3


def main(options):
    root = Path("outputs/full_reproduction/feature_audit")
    state = dict(started_at_unix=time.time(), primary_pid_gate=options.wait_pid,
                 min_available_gib_before_new_mat=options.min_available_gib,
                 calculation_implementation_changed=False,
                 wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 timing_caveat="Per-cell wall time can include RAM guard waiting; it is not pure computation time")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    state["status"] = "waiting_primary"
    frozen.write_json(root / "boundary_memory_guard.json", state)
    handle = kernel.OpenProcess(0x00100000, False, options.wait_pid)
    if handle:
        print(json.dumps(dict(status="waiting_primary_xjtu_exit", primary_pid=options.wait_pid)), flush=True)
        while kernel.WaitForSingleObject(handle, 0) == 258:
            time.sleep(10)
        kernel.CloseHandle(handle)
    for path in (root / "boundary_extraction").rglob("manifest.json"):
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["status"] == "running":
            frozen.write_json(path.with_name("interrupted_before_memory_guard.json"),
                              dict(status="aborted", original_manifest=previous,
                                   reason="Boundary worker stopped at observed completed-cell boundary before serial MAT resumption"))
    original_load = frozen.load_xjtu

    def guarded_load(*arguments):
        gc.collect()
        archive, member, _ = arguments
        # MATLAB object-array expansion and transient copies use substantially more
        # RAM than compressed files; reserve two GiB plus 18x the stored MAT size.
        required_gib = max(3.0, min(options.min_available_gib,
                                  2.0 + archive.getinfo(member).file_size * 18 / 1024 ** 3))
        while available_gib() < required_gib:
            print(json.dumps(dict(status="waiting_available_ram", available_gib=available_gib(),
                                  required_gib=required_gib, member=member)), flush=True)
            time.sleep(10)
        try:
            yield from original_load(*arguments)
        finally:
            gc.collect()

    frozen.load_xjtu = guarded_load
    state["resumed_at_unix"] = time.time()
    state["status"] = "running"
    frozen.write_json(root / "boundary_memory_guard.json", state)
    frozen.main(True)
    state.update(status="complete", completed_at_unix=time.time())
    frozen.write_json(root / "boundary_memory_guard.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, default=25284)
    parser.add_argument("--min-available-gib", type=float, default=4.5)
    main(parser.parse_args())
