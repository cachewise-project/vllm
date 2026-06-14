# SPDX-License-Identifier: Apache-2.0
"""Tests for Cachewise prefix-aware waiting-queue scheduling."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import EOS_TOKEN_ID, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test

_BLOCK_HASHER_INIT = False


def _ensure_block_hasher(block_size: int = 16):
    global _BLOCK_HASHER_INIT
    if not _BLOCK_HASHER_INIT:
        init_none_hash(sha256)
        _BLOCK_HASHER_INIT = True
    return get_request_block_hasher(block_size, sha256)


def _make_request(
    req_id: str,
    prompt_token_ids: list[int],
    *,
    arrival_time: float,
    max_tokens: int = 1,
    block_size: int = 16,
) -> Request:
    block_hasher = _ensure_block_hasher(block_size)
    sampling_params = SamplingParams(max_tokens=max_tokens)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=req_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=arrival_time,
        block_hasher=block_hasher,
    )


def _scheduler(*, prefix_priority: bool = True, **kwargs):
    return create_scheduler(
        prioritize_waiting_by_prefix_cache=prefix_priority,
        **kwargs,
    )


def _make_output(scheduler) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[req.request_id for req in scheduler.running],
        req_id_to_index={
            req.request_id: i for i, req in enumerate(scheduler.running)
        },
        sampled_token_ids=[[1000]] * len(scheduler.running),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _run_request_to_completion(scheduler) -> None:
    output = scheduler.schedule()
    assert output.scheduled_new_reqs, "expected request to be scheduled"
    scheduler.update_from_output(output, _make_output(scheduler))
    while scheduler.running:
        output = scheduler.schedule()
        scheduler.update_from_output(output, _make_output(scheduler))


# --- direct tests: _pick_waiting_request_by_prefix_cache --------------------


def test_pick_waiting_prefers_larger_local_prefix_match():
    scheduler = _scheduler(enable_prefix_caching=True)

    for req in (
        _make_request("fcfs-head", [5] * 32, arrival_time=1.0),
        _make_request("best", [0] * 32, arrival_time=2.0),
        _make_request("other", [1] * 16 + [2] * 16, arrival_time=3.0),
    ):
        scheduler.add_request(req)

    def fake_get_computed_blocks(req):
        scores = {"fcfs-head": 0, "best": 32, "other": 16}
        return Mock(), scores[req.request_id]

    scheduler.kv_cache_manager.get_computed_blocks = fake_get_computed_blocks

    picked = scheduler._pick_waiting_request_by_prefix_cache(set())
    assert picked is not None
    assert picked.request_id == "best"


def test_pick_waiting_tiebreaks_by_arrival_time():
    scheduler = _scheduler()

    scheduler.add_request(_make_request("late", [0] * 32, arrival_time=2.0))
    scheduler.add_request(_make_request("early", [0] * 32, arrival_time=1.0))

    scheduler.kv_cache_manager.get_computed_blocks = lambda req: (Mock(), 32)

    picked = scheduler._pick_waiting_request_by_prefix_cache(set())
    assert picked is not None
    assert picked.request_id == "early"


def test_pick_waiting_includes_connector_prefix():
    scheduler = _scheduler(
        use_kv_connector=mock_kv(matched_tokens=64, is_async=False),
    )

    scheduler.add_request(_make_request("local", [0] * 32, arrival_time=1.0))
    scheduler.add_request(_make_request("connector", [9] * 32, arrival_time=2.0))

    def fake_get_computed_blocks(req):
        return Mock(), 48 if req.request_id == "local" else 0

    scheduler.kv_cache_manager.get_computed_blocks = fake_get_computed_blocks

    picked = scheduler._pick_waiting_request_by_prefix_cache(set())
    assert picked is not None
    # 48 local + 64 connector = 112 vs 0 + 64 = 64
    assert picked.request_id == "local"


def test_pick_waiting_skips_blocked_requests():
    scheduler = _scheduler()

    blocked = _make_request("blocked", [0] * 32, arrival_time=1.0)
    blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    ready = _make_request("ready", [1] * 32, arrival_time=2.0)

    scheduler.add_request(blocked)
    scheduler.add_request(ready)

    scheduler.kv_cache_manager.get_computed_blocks = lambda req: (Mock(), 999)

    picked = scheduler._pick_waiting_request_by_prefix_cache(set())
    assert picked is not None
    assert picked.request_id == "ready"


# --- integration tests: schedule() ----------------------------------------


def test_schedule_prefers_prefix_match_over_fcfs_head():
    block_size = 16
    scheduler = _scheduler(
        enable_prefix_caching=True,
        max_num_seqs=1,
        max_num_batched_tokens=256,
        block_size=block_size,
        num_blocks=256,
    )

    # Warm GPU prefix cache with [0] * 32.
    warm = _make_request("warm", [0] * 32, arrival_time=0.0, block_size=block_size)
    scheduler.add_request(warm)
    _run_request_to_completion(scheduler)
    scheduler.finish_requests(warm.request_id, RequestStatus.FINISHED_STOPPED)

    # FCFS head has no match; second waiter shares the warm prefix.
    scheduler.add_request(_make_request("no-match", [5] * 32, arrival_time=1.0, block_size=block_size))
    scheduler.add_request(_make_request("best-match", [0] * 32, arrival_time=2.0, block_size=block_size))
    scheduler.add_request(_make_request("other", [6] * 32, arrival_time=3.0, block_size=block_size))

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "best-match"


def test_schedule_fcfs_when_prefix_priority_disabled():
    block_size = 16
    scheduler = _scheduler(
        prefix_priority=False,
        enable_prefix_caching=True,
        max_num_seqs=1,
        max_num_batched_tokens=256,
        block_size=block_size,
        num_blocks=256,
    )

    warm = _make_request("warm", [0] * 32, arrival_time=0.0, block_size=block_size)
    scheduler.add_request(warm)
    _run_request_to_completion(scheduler)
    scheduler.finish_requests(warm.request_id, RequestStatus.FINISHED_STOPPED)

    scheduler.add_request(_make_request("no-match", [5] * 32, arrival_time=1.0, block_size=block_size))
    scheduler.add_request(_make_request("best-match", [0] * 32, arrival_time=2.0, block_size=block_size))

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "no-match"