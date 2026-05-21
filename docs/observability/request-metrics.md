---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Per-Request Metrics Logging
---

Dynamo can emit a structured JSON record per completed inference request for operations and analytics. Records are published at request completion and written asynchronously by background sink workers (off the hot path).

## Enable

```bash
export DYN_REQUEST_METRICS_ENABLED=true
export DYN_REQUEST_METRICS_SINKS=stderr   # or "file" or "stderr,file"
export DYN_LOGGING_JSONL=true             # recommended for log platforms

# KV hit fields require KV router mode
python -m dynamo.frontend --router-mode kv ...
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DYN_REQUEST_METRICS_ENABLED` | Enable per-request metrics logging | `false` |
| `DYN_REQUEST_METRICS_SINKS` | Comma-separated sinks: `stderr`, `file` | `stderr` |
| `DYN_REQUEST_METRICS_FILE` | Append path when `file` sink is used | `/tmp/dynamo_request_metrics.jsonl` |
| `DYN_REQUEST_METRICS_CAPACITY` | Broadcast channel capacity | `1024` |

## Record Schema

Each line is a JSON object with `log_type=request_metrics` when using the stderr sink (via `tracing`).

| Field | Description |
|-------|-------------|
| `request_id` | Request context ID |
| `model` | Model name |
| `input_tokens` | Input sequence length in tokens |
| `output_tokens` | Output sequence length in tokens |
| `request_received_ms` | Epoch milliseconds when the frontend received the request |
| `ttft_ms` | Time to first token (milliseconds) |
| `total_time_ms` | Total request duration (milliseconds) |
| `worker` | Prefill/decode worker IDs and DP ranks (if recorded) |
| `selected_kv_hit_rate` | Router-predicted KV hit rate on the **selected** worker (0.0–1.0). KV mode only. |
| `max_kv_hit_rate` | Router-predicted **maximum** KV hit rate across all workers (0.0–1.0). KV mode only. |
| `phase_at_routing` | Request phase when KV metrics were first recorded (`prefill`, `decode`, or `aggregated`) |
| `router_mode` | `"kv"` when KV metrics are present |

### KV Hit Rate Semantics

Both `selected_kv_hit_rate` and `max_kv_hit_rate` are **router predictions at routing time** (overlap blocks / input blocks), not engine-measured prefix cache hits.

- **Selected**: overlap on the worker chosen by the scheduler (may not be the worker with the highest overlap).
- **Max**: best overlap among all workers returned by the KV indexer at routing time.

In disaggregated serving, KV metrics use first-write-wins semantics on the shared request tracker (typically the prefill routing phase).

## Example Record

```json
{
  "schema_version": 1,
  "request_id": "abc-123",
  "model": "Qwen/Qwen3-0.6B",
  "input_tokens": 512,
  "output_tokens": 128,
  "request_received_ms": 1716123456789,
  "ttft_ms": 45.2,
  "total_time_ms": 3200.5,
  "worker": {
    "prefill_worker_id": 1,
    "prefill_dp_rank": 0,
    "decode_worker_id": 1,
    "decode_dp_rank": 0
  },
  "selected_kv_hit_rate": 0.3,
  "max_kv_hit_rate": 0.7,
  "phase_at_routing": "aggregated",
  "router_mode": "kv"
}
```

## Loki Query Example

```logql
{app="dynamo-frontend"} | json | log_type="request_metrics"
```

## File Sink

```bash
export DYN_REQUEST_METRICS_SINKS=file
export DYN_REQUEST_METRICS_FILE=/var/log/dynamo/request_metrics.jsonl
```

Each completed request appends one JSON line to the file.

## Related Documentation

- [Metrics](metrics.md) — Prometheus histograms and gauges
- [Logging](logging.md) — JSONL logging and trace context
