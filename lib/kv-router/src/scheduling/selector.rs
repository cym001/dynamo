// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use rand::Rng;
use rustc_hash::FxHashMap;

use super::config::KvRouterConfig;
use super::types::{KvSchedulerError, SchedulingRequest, pinned_worker_config};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank};

/// A trait that users can implement to define custom selection logic.
///
/// Generic over `C` so that the scheduling layer does not depend on a concrete config type.
pub trait WorkerSelector<C: WorkerConfigLike> {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;
}

/// Helper function for softmax sampling.
/// Returns the selected worker and its logit.
fn softmax_sample(
    logits: &FxHashMap<WorkerWithDpRank, f64>,
    temperature: f64,
) -> (WorkerWithDpRank, f64) {
    let mut rng = rand::rng();
    softmax_sample_with_sample(logits, temperature, rng.random())
}

fn softmax_sample_with_sample(
    logits: &FxHashMap<WorkerWithDpRank, f64>,
    temperature: f64,
    sample: f64,
) -> (WorkerWithDpRank, f64) {
    if logits.is_empty() {
        panic!("Empty logits for softmax sampling");
    }

    // Guard: at zero temperature, return a minimum-logit worker directly.
    if temperature == 0.0 {
        let mut logit_iter = logits.iter();
        let (first_key, first_logit) = logit_iter.next().unwrap();

        let mut min_logit = first_logit;
        let mut min_key = first_key;
        for (key, logit) in logit_iter {
            if logit < min_logit {
                min_logit = logit;
                min_key = key;
            }
        }

        return (*min_key, *min_logit);
    }

    let entries: Vec<_> = logits
        .iter()
        .map(|(worker, logit)| (*worker, *logit))
        .collect();
    let values: Vec<_> = entries.iter().map(|(_, logit)| *logit).collect();

    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let probabilities = if min_val == max_val {
        vec![1.0 / entries.len() as f64; entries.len()]
    } else {
        // Fused normalize -> negate -> scale -> exp, then normalize probabilities
        let range = max_val - min_val;
        let scaled: Vec<f64> = values.iter().map(|&v| -(v / range) / temperature).collect();
        let max_scaled = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut probs: Vec<f64> = scaled.iter().map(|&v| (v - max_scaled).exp()).collect();
        let sum: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        probs
    };

    let mut cumsum = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cumsum += prob;
        if sample <= cumsum {
            return entries[i];
        }
    }

    // Fallback to last key (shouldn't normally reach here)
    entries[entries.len() - 1]
}

const DYN_ROUTER_WORKER_SELECTION_FORMULA: &str = "DYN_ROUTER_WORKER_SELECTION_FORMULA";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerSelectionFormula {
    OverlapLoad,
    Lmetric,
    Vllm,
    Random,
    KvAware,
}

impl WorkerSelectionFormula {
    fn from_env() -> Self {
        std::env::var(DYN_ROUTER_WORKER_SELECTION_FORMULA)
            .map(|value| Self::parse(&value))
            .unwrap_or(Self::OverlapLoad)
    }

    fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "default" | "overlap-load" | "overlap_load" | "overlapload" => Self::OverlapLoad,
            "lmetric" => Self::Lmetric,
            "vllm" => Self::Vllm,
            "random" => Self::Random,
            "kv-aware" | "kv_aware" | "kvaware" => Self::KvAware,
            other => {
                tracing::warn!(
                    env_var = DYN_ROUTER_WORKER_SELECTION_FORMULA,
                    value = other,
                    "unknown worker selection formula, falling back to overlap-load"
                );
                Self::OverlapLoad
            }
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::OverlapLoad => "overlap-load",
            Self::Lmetric => "lmetric",
            Self::Vllm => "vllm",
            Self::Random => "random",
            Self::KvAware => "kv-aware",
        }
    }
}

fn overlap_load_score(overlap_weight: f64, potential_prefill_block: f64, decode_block: f64) -> f64 {
    overlap_weight * potential_prefill_block + decode_block
}

fn lmetric_score(isl: usize, overlap_blocks: u32, block_size: u32, active_requests: usize) -> f64 {
    let cached_tokens = overlap_blocks as usize * block_size as usize;
    let new_tokens = isl.saturating_sub(cached_tokens);
    (new_tokens as f64) * ((active_requests + 1) as f64)
}

fn vllm_score(active_requests: usize, max_batch: usize) -> (f64, usize, usize) {
    let candidate_active_requests = active_requests + 1;
    let running = candidate_active_requests.min(max_batch);
    let waiting = candidate_active_requests.saturating_sub(max_batch);
    ((waiting * 4 + running) as f64, waiting, running)
}

fn kv_aware_running(active_requests: usize) -> usize {
    active_requests + 1
}

fn kv_aware_score(
    request_blocks: usize,
    overlap_blocks: u32,
    active_requests: usize,
    overlap_weight: f64,
) -> f64 {
    let miss_blocks = request_blocks.saturating_sub(overlap_blocks as usize);
    let running = kv_aware_running(active_requests);
    (miss_blocks as f64) * overlap_weight + running as f64
}

fn break_tied_workers_by_tree_size(
    min_workers: &[WorkerWithDpRank],
    request: &SchedulingRequest,
    min_score: f64,
) -> (WorkerWithDpRank, f64) {
    let tree_sizes: Vec<(usize, &WorkerWithDpRank)> = min_workers
        .iter()
        .map(|w| (request.overlaps.tree_sizes.get(w).copied().unwrap_or(0), w))
        .collect();

    if tree_sizes.iter().all(|(s, _)| *s == tree_sizes[0].0) {
        let idx = rand::rng().random_range(0..min_workers.len());
        (min_workers[idx], min_score)
    } else {
        let (_, worker) = *tree_sizes.iter().min_by_key(|(s, _)| *s).unwrap();
        (*worker, min_score)
    }
}

fn break_kv_aware_tie(
    min_workers: &[WorkerWithDpRank],
    request: &SchedulingRequest,
    min_score: f64,
) -> (WorkerWithDpRank, f64) {
    let min_running = min_workers
        .iter()
        .map(|w| kv_aware_running(request.active_request_counts.get(w).copied().unwrap_or(0)))
        .min()
        .unwrap_or(1);

    let min_running_workers: Vec<WorkerWithDpRank> = min_workers
        .iter()
        .copied()
        .filter(|w| {
            kv_aware_running(request.active_request_counts.get(w).copied().unwrap_or(0))
                == min_running
        })
        .collect();

    if min_running_workers.len() == 1 {
        (min_running_workers[0], min_score)
    } else {
        break_tied_workers_by_tree_size(&min_running_workers, request, min_score)
    }
}

/// Matches the override injected by disaggregated decode routing after prefill completes.
fn is_disaggregated_decode_request(request: &SchedulingRequest) -> bool {
    let Some(override_cfg) = request.router_config_override.as_ref() else {
        return false;
    };

    override_cfg.overlap_score_weight == Some(0.0)
        && override_cfg.track_prefill_tokens == Some(false)
        && override_cfg.assume_kv_reuse == Some(false)
}

/// Default implementation matching the Python _cost_function.
#[derive(Debug, Clone)]
pub struct DefaultWorkerSelector {
    pub kv_router_config: KvRouterConfig,
    pub worker_type: &'static str,
    worker_selection_formula: WorkerSelectionFormula,
}

#[derive(Debug, Clone, Copy)]
struct WorkerScore {
    overlap_blocks: u32,
    logit: f64,
}

impl Default for DefaultWorkerSelector {
    fn default() -> Self {
        Self {
            kv_router_config: KvRouterConfig::default(),
            worker_type: "unknown",
            worker_selection_formula: WorkerSelectionFormula::OverlapLoad,
        }
    }
}

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>, worker_type: &'static str) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
            worker_type,
            worker_selection_formula: WorkerSelectionFormula::from_env(),
        }
    }

    fn effective_formula(&self, request: &SchedulingRequest) -> WorkerSelectionFormula {
        if self.worker_type == "decode" && is_disaggregated_decode_request(request) {
            WorkerSelectionFormula::OverlapLoad
        } else {
            self.worker_selection_formula
        }
    }

    fn worker_score<C: WorkerConfigLike>(
        &self,
        request: &SchedulingRequest,
        worker: WorkerWithDpRank,
        config: &C,
        block_size: u32,
        overlap_weight: f64,
        formula: WorkerSelectionFormula,
    ) -> WorkerScore {
        let isl = request.isl_tokens;
        let overlap_blocks = request.overlaps.scores.get(&worker).copied().unwrap_or(0);
        let default_prefill_token = if request.track_prefill_tokens { isl } else { 0 };
        let prefill_token = request
            .prefill_tokens
            .get(&worker)
            .copied()
            .unwrap_or(default_prefill_token);
        let potential_prefill_block = (prefill_token as f64) / (block_size as f64);
        let decode_block = request
            .decode_blocks
            .get(&worker)
            .copied()
            .unwrap_or(potential_prefill_block.floor() as usize) as f64;
        let active_requests = request
            .active_request_counts
            .get(&worker)
            .copied()
            .unwrap_or(0);
        let max_batch = config
            .max_num_seqs()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(usize::MAX);

        let logit = match formula {
            WorkerSelectionFormula::OverlapLoad => {
                let score =
                    overlap_load_score(overlap_weight, potential_prefill_block, decode_block);
                tracing::debug!(
                    worker_id = worker.worker_id,
                    dp_rank = ?worker.dp_rank,
                    formula = formula.as_str(),
                    overlap_blocks,
                    logit = score,
                    overlap_weight,
                    potential_prefill_block,
                    decode_block,
                    "Worker selection score: overlap-load = overlap_weight * prefill_blocks + decode_blocks"
                );
                score
            }
            WorkerSelectionFormula::Lmetric => {
                let score = lmetric_score(isl, overlap_blocks, block_size, active_requests);
                let cached_tokens = overlap_blocks as usize * block_size as usize;
                let new_tokens = isl.saturating_sub(cached_tokens);
                tracing::debug!(
                    worker_id = worker.worker_id,
                    dp_rank = ?worker.dp_rank,
                    formula = formula.as_str(),
                    overlap_blocks,
                    new_tokens,
                    active_requests,
                    logit = score,
                    "Worker selection score: lmetric = new_tokens * (active_requests + 1)"
                );
                score
            }
            WorkerSelectionFormula::Vllm => {
                let (score, waiting, running) = vllm_score(active_requests, max_batch);
                tracing::debug!(
                    worker_id = worker.worker_id,
                    dp_rank = ?worker.dp_rank,
                    formula = formula.as_str(),
                    active_requests,
                    max_batch,
                    waiting,
                    running,
                    logit = score,
                    "Worker selection score: vllm = waiting * 4 + running"
                );
                score
            }
            WorkerSelectionFormula::Random => {
                tracing::debug!(
                    worker_id = worker.worker_id,
                    dp_rank = ?worker.dp_rank,
                    formula = formula.as_str(),
                    "Worker selection score: random (uniform candidate)"
                );
                0.0
            }
            WorkerSelectionFormula::KvAware => {
                let request_blocks = isl.div_ceil(block_size as usize);
                let score =
                    kv_aware_score(request_blocks, overlap_blocks, active_requests, overlap_weight);
                let miss_blocks = request_blocks.saturating_sub(overlap_blocks as usize);
                let running = kv_aware_running(active_requests);
                tracing::debug!(
                    worker_id = worker.worker_id,
                    dp_rank = ?worker.dp_rank,
                    formula = formula.as_str(),
                    overlap_blocks,
                    miss_blocks,
                    overlap_weight,
                    active_requests,
                    running,
                    logit = score,
                    "Worker selection score: kv-aware = miss_blocks * overlap_weight + running"
                );
                score
            }
        };

        WorkerScore {
            overlap_blocks,
            logit,
        }
    }
}

impl<C: WorkerConfigLike> WorkerSelector<C> for DefaultWorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);
        request.validate_worker_constraints()?;

        let allowed_ids = request.allowed_worker_ids.as_ref();
        let pinned_worker = request.pinned_worker;

        if pinned_worker.is_none()
            && allowed_ids.map_or(workers.is_empty(), |ids| {
                !workers.keys().any(|wid| ids.contains(wid))
            })
        {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let isl = request.isl_tokens;
        let request_blocks = isl.div_ceil(block_size as usize);
        let overlaps = &request.overlaps.scores;

        let formula = self.effective_formula(request);

        if let Some(worker) = pinned_worker {
            let config = pinned_worker_config(workers, worker)?;

            let overlap_weight = request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.overlap_score_weight)
                .unwrap_or(self.kv_router_config.overlap_score_weight);
            let score =
                self.worker_score(request, worker, config, block_size, overlap_weight, formula);

            return Ok(WorkerSelectionResult {
                worker,
                required_blocks: request_blocks as u64,
                overlap_blocks: score.overlap_blocks,
            });
        }

        let overlap_weight = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.overlap_score_weight)
            .unwrap_or(self.kv_router_config.overlap_score_weight);

        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.kv_router_config.router_temperature);

        let get_score = |worker: WorkerWithDpRank, config: &C| -> f64 {
            self.worker_score(request, worker, config, block_size, overlap_weight, formula)
                .logit
        };

        let worker_iter = workers
            .iter()
            .filter(move |(wid, _)| allowed_ids.is_none_or(|ids| ids.contains(wid)))
            .flat_map(|(worker_id, config)| {
                let data_parallel_size = config.data_parallel_size();
                let data_parallel_start_rank = config.data_parallel_start_rank();
                (data_parallel_start_rank..(data_parallel_start_rank + data_parallel_size))
                    .map(move |dp_rank| (WorkerWithDpRank::new(*worker_id, dp_rank), config))
            });

        let (best_worker, best_logit) =
            if temperature == 0.0 && formula != WorkerSelectionFormula::Random {
                let mut min_workers = Vec::new();
                let mut min_score = f64::INFINITY;
                for (worker, config) in worker_iter {
                    let score = get_score(worker, config);
                    if score < min_score {
                        min_workers.clear();
                        min_workers.push(worker);
                        min_score = score;
                    } else if score == min_score {
                        min_workers.push(worker);
                    }
                }

                if min_workers.len() > 1 {
                    if formula == WorkerSelectionFormula::KvAware {
                        tracing::debug!(
                            "Multiple workers tied with same kv-aware score, using running count as tie-breaker"
                        );
                        break_kv_aware_tie(&min_workers, request, min_score)
                    } else {
                        tracing::debug!(
                            "Multiple workers tied with same logit, using tree size as tie-breaker"
                        );
                        break_tied_workers_by_tree_size(&min_workers, request, min_score)
                    }
                } else {
                    (min_workers[0], min_score)
                }
            } else {
                let mut worker_logits = FxHashMap::default();
                for (worker, config) in worker_iter {
                    let score = get_score(worker, config);
                    worker_logits.insert(worker, score);
                }

                if formula == WorkerSelectionFormula::Random {
                    let entries: Vec<_> = worker_logits.into_iter().collect();
                    let idx = rand::rng().random_range(0..entries.len());
                    entries[idx]
                } else {
                    softmax_sample(&worker_logits, temperature)
                }
            };

        if self.worker_type == "decode" {
            tracing::info!(
                "Selected worker: worker_type={}, worker_id={} dp_rank={:?}, logit: {:.3}",
                self.worker_type,
                best_worker.worker_id,
                best_worker.dp_rank,
                best_logit,
            );
            return Ok(WorkerSelectionResult {
                worker: best_worker,
                required_blocks: request_blocks as u64,
                overlap_blocks: overlaps.get(&best_worker).copied().unwrap_or(0),
            });
        }

        let best_overlap = *overlaps.get(&best_worker).unwrap_or(&0);

        let total_blocks_info = workers
            .get(&best_worker.worker_id)
            .and_then(|cfg| cfg.total_kv_blocks())
            .map(|blocks| format!(", total blocks: {}", blocks))
            .unwrap_or_default();

        let tree_size = request
            .overlaps
            .tree_sizes
            .get(&best_worker)
            .copied()
            .unwrap_or(0);

        tracing::info!(
            "Selected worker: worker_type={}, worker_id={} dp_rank={:?}, logit: {:.3}, cached blocks: {}, tree size: {}{}",
            self.worker_type,
            best_worker.worker_id,
            best_worker.dp_rank,
            best_logit,
            best_overlap,
            tree_size,
            total_blocks_info
        );

        Ok(WorkerSelectionResult {
            worker: best_worker,
            required_blocks: request_blocks as u64,
            overlap_blocks: overlaps.get(&best_worker).copied().unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::OverlapScores;
    use crate::scheduling::config::{KvRouterConfig, RouterConfigOverride};
    use crate::test_utils::SimpleWorkerConfig;

    fn test_selector(formula: WorkerSelectionFormula) -> DefaultWorkerSelector {
        DefaultWorkerSelector {
            kv_router_config: KvRouterConfig {
                router_temperature: 0.0,
                ..Default::default()
            },
            worker_type: "prefill",
            worker_selection_formula: formula,
        }
    }

    fn make_scheduling_request(
        isl_tokens: usize,
        overlaps: OverlapScores,
        active_request_counts: FxHashMap<WorkerWithDpRank, usize>,
        router_config_override: Option<RouterConfigOverride>,
    ) -> SchedulingRequest {
        SchedulingRequest {
            maybe_request_id: None,
            token_seq: None,
            isl_tokens,
            overlaps,
            decode_blocks: FxHashMap::default(),
            prefill_tokens: FxHashMap::default(),
            active_request_counts,
            track_prefill_tokens: true,
            router_config_override,
            update_states: true,
            lora_name: None,
            priority_jump: 0.0,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            resp_tx: None,
        }
    }

    #[test]
    fn test_kv_aware_score() {
        assert_eq!(kv_aware_score(10, 8, 2, 1.0), 5.0);
        assert_eq!(kv_aware_score(10, 5, 0, 1.0), 6.0);
        assert_eq!(kv_aware_running(2), 3);
    }

    #[test]
    fn test_worker_selection_formula_parse() {
        assert_eq!(
            WorkerSelectionFormula::parse("kv-aware"),
            WorkerSelectionFormula::KvAware
        );
        assert_eq!(
            WorkerSelectionFormula::parse("kv_aware"),
            WorkerSelectionFormula::KvAware
        );
        assert_eq!(
            WorkerSelectionFormula::parse("kvaware"),
            WorkerSelectionFormula::KvAware
        );
        assert_eq!(
            WorkerSelectionFormula::parse("unknown-formula"),
            WorkerSelectionFormula::OverlapLoad
        );
    }

    #[test]
    fn test_kv_aware_select_worker_lowest_score() {
        let worker_a = WorkerWithDpRank::new(0, 0);
        let worker_b = WorkerWithDpRank::new(1, 0);
        let mut overlaps = OverlapScores::new();
        overlaps.scores.insert(worker_a, 8);
        overlaps.scores.insert(worker_b, 5);

        let mut active_request_counts = FxHashMap::default();
        active_request_counts.insert(worker_a, 2);
        active_request_counts.insert(worker_b, 0);

        let request = make_scheduling_request(160, overlaps, active_request_counts, None);
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);

        let selector = test_selector(WorkerSelectionFormula::KvAware);
        let result = selector
            .select_worker(&workers, &request, 16)
            .expect("selection should succeed");

        assert_eq!(result.worker, worker_a);
        assert_eq!(result.overlap_blocks, 8);
    }

    #[test]
    fn test_kv_aware_tie_break_by_running() {
        let worker_a = WorkerWithDpRank::new(0, 0);
        let worker_b = WorkerWithDpRank::new(1, 0);
        let mut overlaps = OverlapScores::new();
        overlaps.scores.insert(worker_a, 7);
        overlaps.scores.insert(worker_b, 6);

        let mut active_request_counts = FxHashMap::default();
        active_request_counts.insert(worker_a, 1);
        active_request_counts.insert(worker_b, 0);

        let request = make_scheduling_request(160, overlaps, active_request_counts, None);
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);

        let selector = test_selector(WorkerSelectionFormula::KvAware);
        let result = selector
            .select_worker(&workers, &request, 16)
            .expect("selection should succeed");

        assert_eq!(result.worker, worker_b);
        assert_eq!(result.overlap_blocks, 6);
    }

    #[test]
    fn test_disagg_decode_override_uses_overlap_load() {
        let worker_a = WorkerWithDpRank::new(0, 0);
        let worker_b = WorkerWithDpRank::new(1, 0);
        let mut overlaps = OverlapScores::new();
        overlaps.scores.insert(worker_a, 8);
        overlaps.scores.insert(worker_b, 5);

        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(worker_a, 100);
        decode_blocks.insert(worker_b, 10);

        let request = SchedulingRequest {
            maybe_request_id: None,
            token_seq: None,
            isl_tokens: 160,
            overlaps,
            decode_blocks,
            prefill_tokens: FxHashMap::default(),
            active_request_counts: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: Some(RouterConfigOverride {
                overlap_score_weight: Some(0.0),
                track_prefill_tokens: Some(false),
                assume_kv_reuse: Some(false),
                router_temperature: Some(0.0),
            }),
            update_states: true,
            lora_name: None,
            priority_jump: 0.0,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            resp_tx: None,
        };

        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);

        let selector = DefaultWorkerSelector {
            kv_router_config: KvRouterConfig {
                router_temperature: 0.0,
                ..Default::default()
            },
            worker_type: "decode",
            worker_selection_formula: WorkerSelectionFormula::KvAware,
        };

        let result = selector
            .select_worker(&workers, &request, 16)
            .expect("selection should succeed");

        assert_eq!(result.worker, worker_b);
    }

    #[test]
    fn test_softmax_sample_single_key() {
        let mut logits = FxHashMap::default();
        let worker = WorkerWithDpRank::from_worker_id(42);
        for (logit, temperature) in [
            (0.5, 0.1),
            (0.5, 1.0),
            (0.5, 10.0),
            (-100.0, 1.0),
            (100.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ] {
            logits.clear();
            logits.insert(worker, logit);

            let result = softmax_sample(&logits, temperature);
            assert_eq!(result.0, worker, "Should return the only available worker");
            assert_eq!(result.1, logit, "Should return the selected worker's logit");
        }
    }

    #[test]
    fn test_softmax_sample_zero_temperature() {
        let mut logits = FxHashMap::default();
        let worker1 = WorkerWithDpRank::from_worker_id(1);
        let worker2 = WorkerWithDpRank::from_worker_id(2);
        let worker3 = WorkerWithDpRank::from_worker_id(3);
        let worker4 = WorkerWithDpRank::from_worker_id(4);
        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0);
        logits.insert(worker3, 7.0);
        logits.insert(worker4, 3.5);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(
            result.0, worker2,
            "Should return worker with smallest logit when temperature is 0"
        );
        assert_eq!(
            result.1, 3.0,
            "Should return the smallest logit when temperature is 0"
        );

        logits.clear();
        let worker5 = WorkerWithDpRank::from_worker_id(5);
        let worker6 = WorkerWithDpRank::from_worker_id(6);
        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0);
        logits.insert(worker5, 3.0);
        logits.insert(worker6, 7.0);

        let result = softmax_sample(&logits, 0.0);
        assert!(
            result.0 == worker2 || result.0 == worker5,
            "Should return one of the workers tied for the smallest logit"
        );
        assert_eq!(result.1, 3.0, "Should return the tied minimum logit");

        logits.clear();
        let worker10 = WorkerWithDpRank::from_worker_id(10);
        let worker20 = WorkerWithDpRank::from_worker_id(20);
        let worker30 = WorkerWithDpRank::from_worker_id(30);
        logits.insert(worker10, -1.0);
        logits.insert(worker20, -5.0);
        logits.insert(worker30, 0.0);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(
            result.0, worker20,
            "Should handle negative logits correctly"
        );
        assert_eq!(result.1, -5.0, "Should return the minimum negative logit");
    }

    #[test]
    fn test_softmax_sample_with_sample_returns_selected_logit() {
        let worker1 = WorkerWithDpRank::from_worker_id(1);
        let worker2 = WorkerWithDpRank::from_worker_id(2);
        let worker3 = WorkerWithDpRank::from_worker_id(3);

        let logits = FxHashMap::from_iter([(worker1, 0.0), (worker2, 3.0), (worker3, 9.0)]);
        let entries: Vec<_> = logits
            .iter()
            .map(|(worker, logit)| (*worker, *logit))
            .collect();
        let values: Vec<_> = entries.iter().map(|(_, logit)| *logit).collect();

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let temperature = 1.0;
        let range = max_val - min_val;
        let scaled: Vec<f64> = values.iter().map(|&v| -(v / range) / temperature).collect();
        let max_scaled = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut probabilities: Vec<f64> = scaled.iter().map(|&v| (v - max_scaled).exp()).collect();
        let sum: f64 = probabilities.iter().sum();
        probabilities.iter_mut().for_each(|p| *p /= sum);

        let target_idx = entries
            .iter()
            .position(|(_, logit)| *logit > min_val)
            .expect("expected at least one non-minimum logit");
        let cumsum_before: f64 = probabilities.iter().take(target_idx).sum();
        let sample = cumsum_before + probabilities[target_idx] / 2.0;

        let result = softmax_sample_with_sample(&logits, temperature, sample);
        assert_eq!(result, entries[target_idx]);
    }
}
