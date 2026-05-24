// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async per-request metrics logging for operations and analytics.
//!
//! Enable with `DYN_REQUEST_METRICS_ENABLED=true`. Records are published at request
//! completion and written by background sink workers (stderr JSONL or file append).

pub mod bus;
pub mod config;
pub mod record;
pub mod sink;

pub use config::enabled;
pub use record::RequestMetricsRecord;

pub fn init_bus(capacity: usize) {
    bus::init(capacity);
}

pub fn publish(rec: RequestMetricsRecord) {
    bus::publish(rec);
}

/// Publish a metrics snapshot from a request tracker at stream completion.
pub fn publish_from_tracker(
    tracker: &crate::protocols::common::timing::RequestTracker,
    request_id: &str,
    model: &str,
) {
    if !enabled() {
        return;
    }
    let include_kv = tracker.kv_hit_rate().is_some() || tracker.max_kv_hit_rate().is_some();
    let rec = RequestMetricsRecord::from_tracker(
        tracker,
        request_id,
        model,
        if include_kv { Some("kv") } else { None },
        include_kv,
    );
    publish(rec);
}
