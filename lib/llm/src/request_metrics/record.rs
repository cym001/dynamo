// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;

use crate::protocols::common::timing::RequestTracker;
use crate::protocols::openai::nvext::WorkerIdInfo;

/// Per-request metrics snapshot emitted to async sinks at request completion.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RequestMetricsRecord {
    pub schema_version: u32,
    pub request_id: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub request_received_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<WorkerIdInfo>,
    /// Router-predicted KV hit rate on the selected worker (0.0–1.0). KV router mode only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_kv_hit_rate: Option<f64>,
    /// Router-predicted maximum KV hit rate across all workers (0.0–1.0). KV router mode only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_kv_hit_rate: Option<f64>,
    /// Request phase when KV routing metrics were first recorded (disagg: first routing wins).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_at_routing: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_mode: Option<String>,
}

impl RequestMetricsRecord {
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn from_tracker(
        tracker: &RequestTracker,
        request_id: impl Into<String>,
        model: impl Into<String>,
        router_mode: Option<&str>,
        include_kv_metrics: bool,
    ) -> Self {
        let (selected_kv_hit_rate, max_kv_hit_rate, phase_at_routing) = if include_kv_metrics {
            (
                tracker.kv_hit_rate(),
                tracker.max_kv_hit_rate(),
                Some(tracker.phase().to_string()),
            )
        } else {
            (None, None, None)
        };

        Self {
            schema_version: Self::SCHEMA_VERSION,
            request_id: request_id.into(),
            model: model.into(),
            input_tokens: tracker.isl_tokens().map(|t| t as u64).unwrap_or(0),
            output_tokens: tracker.osl_tokens(),
            request_received_ms: tracker.request_received_epoch_ms(),
            ttft_ms: tracker.ttft_ms(),
            total_time_ms: tracker.total_time_ms(),
            worker: tracker.get_worker_info(),
            selected_kv_hit_rate,
            max_kv_hit_rate,
            phase_at_routing,
            router_mode: router_mode.map(str::to_string),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::timing::RequestTracker;

    #[test]
    fn test_from_tracker_with_kv_metrics() {
        let tracker = RequestTracker::new();
        tracker.record_isl(100, Some(30));
        tracker.record_kv_hit(3, 10);
        tracker.record_max_kv_hit(7, 10);
        tracker.record_osl(50);

        let rec =
            RequestMetricsRecord::from_tracker(&tracker, "req-1", "test-model", Some("kv"), true);

        assert_eq!(rec.request_id, "req-1");
        assert_eq!(rec.input_tokens, 100);
        assert_eq!(rec.output_tokens, 50);
        assert_eq!(rec.selected_kv_hit_rate, Some(0.3));
        assert_eq!(rec.max_kv_hit_rate, Some(0.7));
        assert_eq!(rec.router_mode, Some("kv".to_string()));
    }

    #[test]
    fn test_from_tracker_without_kv_metrics() {
        let tracker = RequestTracker::new();
        tracker.record_isl(64, Some(0));

        let rec = RequestMetricsRecord::from_tracker(&tracker, "req-2", "m", None, false);

        assert!(rec.selected_kv_hit_rate.is_none());
        assert!(rec.max_kv_hit_rate.is_none());
        assert!(rec.phase_at_routing.is_none());
    }
}
