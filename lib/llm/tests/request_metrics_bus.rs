// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test for request metrics bus publish/receive.

use dynamo_llm::protocols::common::timing::RequestTracker;
use dynamo_llm::request_metrics::{RequestMetricsRecord, init_bus};
use std::sync::OnceLock;

static INIT: OnceLock<()> = OnceLock::new();

fn setup() {
    INIT.get_or_init(|| {
        unsafe { std::env::set_var("DYN_REQUEST_METRICS_ENABLED", "true") };
        init_bus(8);
    });
}

#[tokio::test]
async fn test_request_metrics_bus_publish() {
    setup();

    let mut rx = dynamo_llm::request_metrics::bus::subscribe();

    let tracker = RequestTracker::new();
    tracker.record_isl(100, Some(30));
    tracker.record_kv_hit(3, 10);
    tracker.record_max_kv_hit(7, 10);
    tracker.record_osl(42);

    let rec = RequestMetricsRecord::from_tracker(&tracker, "test-req", "test-model", Some("kv"), true);
    dynamo_llm::request_metrics::publish(rec.clone());

    let received = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
        .await
        .expect("timed out waiting for metrics")
        .expect("channel closed");

    assert_eq!(received.request_id, "test-req");
    assert_eq!(received.input_tokens, 100);
    assert_eq!(received.output_tokens, 42);
    assert_eq!(received.selected_kv_hit_rate, Some(0.3));
    assert_eq!(received.max_kv_hit_rate, Some(0.7));
}
