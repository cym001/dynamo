// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use super::record::RequestMetricsRecord;

#[async_trait]
pub trait RequestMetricsSink: Send + Sync {
    fn name(&self) -> &'static str;
    async fn emit(&self, rec: &RequestMetricsRecord);
}

pub struct StderrSink;

#[async_trait]
impl RequestMetricsSink for StderrSink {
    fn name(&self) -> &'static str {
        "stderr"
    }

    async fn emit(&self, rec: &RequestMetricsRecord) {
        match serde_json::to_string(rec) {
            Ok(js) => {
                tracing::info!(
                    target = "dynamo_llm::request_metrics",
                    log_type = "request_metrics",
                    record = %js,
                    "request_metrics"
                );
            }
            Err(e) => tracing::warn!("request_metrics: serialize failed: {e}"),
        }
    }
}

pub struct FileSink {
    path: PathBuf,
    file: Mutex<tokio::fs::File>,
}

impl FileSink {
    pub async fn new(path: PathBuf) -> anyhow::Result<Self> {
        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;
        Ok(Self {
            path,
            file: Mutex::new(file),
        })
    }
}

#[async_trait]
impl RequestMetricsSink for FileSink {
    fn name(&self) -> &'static str {
        "file"
    }

    async fn emit(&self, rec: &RequestMetricsRecord) {
        let line = match serde_json::to_string(rec) {
            Ok(js) => format!("{js}\n"),
            Err(e) => {
                tracing::warn!("request_metrics: serialize failed: {e}");
                return;
            }
        };
        let mut file = self.file.lock().await;
        if let Err(e) = file.write_all(line.as_bytes()).await {
            tracing::warn!(path = %self.path.display(), "request_metrics: file write failed: {e}");
        }
    }
}

async fn parse_sinks_from_env() -> anyhow::Result<Vec<Arc<dyn RequestMetricsSink>>> {
    let cfg = std::env::var("DYN_REQUEST_METRICS_SINKS").unwrap_or_else(|_| "stderr".into());
    let mut out: Vec<Arc<dyn RequestMetricsSink>> = Vec::new();
    for name in cfg.split(',').map(|s| s.trim().to_lowercase()) {
        match name.as_str() {
            "stderr" | "" => out.push(Arc::new(StderrSink)),
            "file" => {
                let path = std::env::var("DYN_REQUEST_METRICS_FILE")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("/tmp/dynamo_request_metrics.jsonl"));
                let sink = FileSink::new(path).await?;
                out.push(Arc::new(sink));
            }
            other => tracing::warn!(%other, "request_metrics: unknown sink ignored"),
        }
    }
    Ok(out)
}

/// Spawn one worker per sink; each subscribes to the bus (off hot path).
pub async fn spawn_workers_from_env() -> anyhow::Result<()> {
    if !super::config::enabled() {
        return Ok(());
    }

    let sinks = parse_sinks_from_env().await?;
    for sink in sinks {
        let name = sink.name();
        let mut rx = super::bus::subscribe();
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(rec) => sink.emit(&rec).await,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(
                            sink = name,
                            dropped = n,
                            "request_metrics bus lagged; dropped records"
                        );
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }
    tracing::info!("Request metrics sinks ready.");
    Ok(())
}
