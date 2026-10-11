"""Load-time device memory (VRAM) read from a llama-server log (see docs/server-log.md).

llama-server has no endpoint for device memory, so its log is the only source. Stdlib only,
file I/O and pure logic: finding the log that belongs to the server being talked to, cutting
out its last startup block, and adding up the buffer lines. Everything is lenient (a
garbled line is skipped, nothing raises) and the log is read once, at startup.

The log is found by fingerprint, never by a marker: the server logs no request, so nothing
can say which file belongs to the live server. The newest file whose last startup matches
/props is taken, and the caller labels the figure as possibly from an earlier run.
"""

import os
import re
from urllib.parse import urlsplit

LOG_SUFFIXES = (".txt", ".log")
MAX_FILES = 20
LOOPBACK = ("127.0.0.1", "localhost", "[::1]", "::1")

# Matched with re.search: a launch may prefix every line (--log-prefix, --log-timestamps)
# and the function tag in front of the message differs between versions, so only the
# message text is used.
_LISTEN = re.compile(r"listening on (?:https?://)?(?P<host>\[[^\]]+\]|[^:/\s]+):(?P<port>\d+)")
_LOAD = re.compile(
    r"loaded meta data with \d+ key-value pairs and \d+ tensors from (?P<path>.+?)(?: \(version .*)?$"
)
_INIT = re.compile(r"initializing, n_slots = (?P<slots>\d+), n_ctx_slot = (?P<ctx>\d+)")
_BUFFER = re.compile(
    r"(?P<name>\S+)\s+(?P<kind>model|KV|compute|output) buffer size\s*=\s*(?P<mib>\d+(?:\.\d+)?) MiB"
)
_RELEVANT = ("buffer size", "loaded meta data", "initializing, n_slots", "listening on")

# model and KV: lines on one buffer name add up (an iSWA model logs a base and an SWA cache,
# a split model several model buffers). compute and output: each line replaces the last.
_SUMMED = ("model", "KV")


def is_loopback_url(url):
    """True when the URL's host is this machine."""
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    return host in ("127.0.0.1", "localhost", "::1")


def url_port(url):
    try:
        return urlsplit(url).port
    except ValueError:
        return None


def last_startup_block(path):
    """The relevant lines of the last complete startup block of a log, or None.

    A block is the text between one `listening on` line (or the file start) and the next,
    so a run that never came up has none and the lines after the last one (the running
    server's own output) are not part of it. Streams the file and keeps only the lines
    the parser reads, so memory stays small however long the log is.
    """
    last, current = None, []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not any(word in line for word in _RELEVANT):
                    continue
                current.append(line.rstrip("\n"))
                if _LISTEN.search(line):
                    last, current = current, []
    except OSError:
        return None
    return last


def parse_block(lines):
    """Read a startup block: {'segments', 'n_slots', 'n_ctx_slot', 'host', 'port'}.

    A segment starts at each `loaded meta data` line: {'path', 'buffers'} where buffers maps
    a buffer name to {'model', 'KV', 'compute', 'output'} in MiB. Fields not found are None.
    """
    block = {"segments": [], "n_slots": None, "n_ctx_slot": None, "host": None, "port": None}
    for line in lines:
        m = _LOAD.search(line)
        if m:
            block["segments"].append({"path": m.group("path"), "buffers": {}})
            continue
        m = _BUFFER.search(line)
        if m:
            if not block["segments"]:
                continue
            kinds = block["segments"][-1]["buffers"].setdefault(m.group("name"), {})
            kind, mib = m.group("kind"), float(m.group("mib"))
            kinds[kind] = kinds.get(kind, 0.0) + mib if kind in _SUMMED else mib
            continue
        m = _INIT.search(line)
        if m:
            block["n_slots"], block["n_ctx_slot"] = int(m.group("slots")), int(m.group("ctx"))
            continue
        m = _LISTEN.search(line)
        if m:
            block["host"], block["port"] = m.group("host"), int(m.group("port"))
    return block


def _same_model(log_path, props_path):
    if not log_path or not props_path:
        return False
    return log_path == props_path or os.path.basename(log_path) == os.path.basename(props_path)


def chosen_segment(block, props):
    """The last load segment whose path is the served model's: this drops the --fit probe
    loads (earlier segments) and a draft model (another path)."""
    path = props.get("model_path") if isinstance(props, dict) else None
    for segment in reversed(block["segments"]):
        if _same_model(segment["path"], path):
            return segment
    return None


def matches(lines, props, port):
    """True when this startup block is the one that /props describes.

    Strict: the model path, the loopback `listening on` port, and the slot count and slot
    context against /props (the same value in a healthy server). A mismatch means "not
    this log", never "close enough".
    """
    if not isinstance(props, dict):
        return False
    block = parse_block(lines)
    if chosen_segment(block, props) is None:
        return False
    if block["host"] not in LOOPBACK or port is None or block["port"] != port:
        return False
    try:
        n_ctx = int(props["default_generation_settings"]["n_ctx"])
        slots = int(props["total_slots"])
    except (KeyError, TypeError, ValueError):
        return False
    return block["n_slots"] == slots and block["n_ctx_slot"] == n_ctx


def is_host_buffer(name):
    """CPU and CPU_Mapped (the mmap buffer), and pinned host memory such as CUDA_Host."""
    return name.startswith("CPU") or name.endswith("_Host")


def vram_summary(lines, props):
    """Device memory at load, in MiB, from a block that matches.

    {'device', 'model', 'kv', 'compute', 'output'} for one device buffer name;
    {'multi_gpu': True} for two or more; None when there is none or the model is not found.
    Only one device is supported; buffers are kept per name so more could be added.
    """
    segment = chosen_segment(parse_block(lines), props)
    if segment is None:
        return None
    devices = {
        name: kinds
        for name, kinds in segment["buffers"].items()
        if not is_host_buffer(name) and sum(kinds.values()) > 0
    }
    if not devices:
        return None
    if len(devices) > 1:
        return {"multi_gpu": True}
    ((name, kinds),) = devices.items()
    return {
        "device": name,
        "model": kinds.get("model", 0.0),
        "kv": kinds.get("KV", 0.0),
        "compute": kinds.get("compute", 0.0),
        "output": kinds.get("output", 0.0),
    }


def candidate_files(log_dir):
    """The newest *.txt and *.log files of a directory, newest first, at most MAX_FILES."""
    found = []
    try:
        with os.scandir(log_dir) as entries:
            for entry in entries:
                if entry.name.endswith(LOG_SUFFIXES):
                    try:
                        if entry.is_file():
                            found.append((entry.stat().st_mtime, entry.path))
                    except OSError:
                        continue
    except OSError:
        return []
    found.sort(reverse=True)
    return [path for _, path in found[:MAX_FILES]]


def find_log(log_dir, log_file, props, url):
    """(path, summary) of the newest log that matches the server, or None.

    Does nothing unless a directory or file is configured and the URL is this machine's:
    a log here cannot belong to a remote server. `log_file` skips the directory listing but
    is checked like any candidate. Never raises.
    """
    if not (log_dir or log_file) or not is_loopback_url(url) or not isinstance(props, dict):
        return None
    port = url_port(url)
    paths = [os.path.expanduser(os.path.expandvars(log_file))] if log_file else []
    if not paths:
        paths = candidate_files(os.path.expanduser(os.path.expandvars(log_dir)))
    for path in paths:
        block = last_startup_block(path)
        if block and matches(block, props, port):
            summary = vram_summary(block, props)
            return (path, summary) if summary else None
    return None
