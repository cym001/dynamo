// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug)]
pub struct RequestMetricsPolicy {
    pub enabled: bool,
}

static POLICY: OnceLock<RequestMetricsPolicy> = OnceLock::new();

pub fn init_from_env() -> RequestMetricsPolicy {
    let enabled = std::env::var("DYN_REQUEST_METRICS_ENABLED")
        .ok()
        .and_then(|v| v.parse::<bool>().ok())
        .unwrap_or(false);
    RequestMetricsPolicy { enabled }
}

pub fn policy() -> RequestMetricsPolicy {
    *POLICY.get_or_init(init_from_env)
}

pub fn enabled() -> bool {
    policy().enabled
}
