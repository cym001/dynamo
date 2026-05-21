// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::record::RequestMetricsRecord;
use std::sync::OnceLock;
use tokio::sync::broadcast;

static BUS: OnceLock<broadcast::Sender<RequestMetricsRecord>> = OnceLock::new();

pub fn init(capacity: usize) {
    let (tx, _rx) = broadcast::channel::<RequestMetricsRecord>(capacity);
    let _ = BUS.set(tx);
}

pub fn subscribe() -> broadcast::Receiver<RequestMetricsRecord> {
    BUS.get()
        .expect("request_metrics bus not initialized")
        .subscribe()
}

pub fn publish(rec: RequestMetricsRecord) {
    if !super::config::enabled() {
        return;
    }
    if let Some(tx) = BUS.get() {
        let _ = tx.send(rec);
    }
}
