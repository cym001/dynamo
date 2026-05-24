# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge]


def _parse_config(*args: str) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    namespace = parser.parse_args(list(args))
    config = FrontendConfig.from_cli_args(namespace)
    config.validate()
    return config


def test_router_mode_accepts_lmetric() -> None:
    config = _parse_config("--router-mode", "lmetric")

    assert config.router_mode == "lmetric"


def test_lmetric_requires_active_block_tracking() -> None:
    with pytest.raises(ValueError, match="--router-mode=lmetric requires --router-track-active-blocks"):
        _parse_config("--router-mode", "lmetric", "--no-router-track-active-blocks")


def test_kv_only_options_accept_lmetric() -> None:
    config = _parse_config("--router-mode", "lmetric", "--serve-indexer")

    assert config.router_mode == "lmetric"
    assert config.serve_indexer is True
