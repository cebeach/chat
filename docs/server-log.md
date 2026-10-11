# Server log: VRAM at load

llama-server has no endpoint that reports device memory, so its log is the only source.
When you point chat.py at your logs, `/info` shows what the server used when it loaded:

```
VRAM (at load)   CUDA0 model 11.8 GiB, KV 4.7, compute 2.1 (log: llama-server_x.txt, may belong to an earlier run)
```

## Setup

Keep the server's output in a file, for example `llama-server ... 2>&1 | tee ~/logs/llama-server_$(date +%s).txt`,
then set it once in `config.toml`:

```toml
server_log_dir = "~/logs"   # ~ and $VAR are expanded; or --server-log-dir on the command line
```

`server_log = "/path/file.txt"` names one file instead of a directory. Nothing is read
unless one of them is set, and only when the server URL is on this machine.

## How the log is found

The server logs no request and writes no run ID, so chat.py cannot ask which file is its
own. At startup it reads the 20 newest `*.txt` and `*.log` files and takes the newest whose
last startup matches `/props`: the model path, the port, and the slot count and slot context.
A stale log of an earlier launch with the same model, port and context cannot be told apart,
which is why the figure says it *may belong to an earlier run*.

## What is counted

From the last startup block (the lines up to the last `listening on`), for the last load of the
served model: model and KV buffers are summed, the compute buffer is the last one logged.
Host memory (`CPU`, `CPU_Mapped`, `*_Host`) is not VRAM.

## Limits

- One GPU. With two or more device buffers `/info` says "multi-GPU not supported"; CPU-only shows nothing.
- The log is read once, at startup. If the server is later found serving another model the
  figure is dropped ("not available (model changed since startup)"); a restart with the same
  model and different settings is not noticed.
- Nothing is shown when no log matches.
