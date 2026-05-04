# SPDX-License-Identifier: Apache-2.0
"""BlockPool free-heap + prefix-cache tests (small pools so heap has no extra ties)."""

from __future__ import annotations

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.cachewise_policy import cachewise_eviction_score
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id


def _policy(ground_truth_idle_s: float) -> dict:
    return {
        "version": 1,
        "hints": {"oracle": {"ground_truth_idle_s": ground_truth_idle_s}},
    }


def _unique_prefix_key(block_id: int) -> object:
    digest = block_id.to_bytes(32, "big")
    return make_block_hash_with_group_id(BlockHash(digest), group_id=0)


def _install_prefix_cached_block(
    pool: BlockPool,
    block,
    *,
    ground_truth_idle_s: float,
) -> object:
    assert block.block_hash is None
    key = _unique_prefix_key(block.block_id)
    block.block_hash = key
    block.cachewise_policy = _policy(ground_truth_idle_s)
    pool.cached_block_hash_to_block.insert(key, block)
    return key


def _pool(*, n: int, enable_caching: bool = True) -> BlockPool:
    return BlockPool(
        num_gpu_blocks=n,
        enable_caching=enable_caching,
        hash_block_size=16,
        enable_kv_cache_events=False,
        metrics_collector=None,
    )


# --- scoring -----------------------------------------------------------------


def test_cachewise_eviction_score_reads_oracle_idle():
    assert cachewise_eviction_score(_policy(123.5)) == pytest.approx(123.5)
    assert cachewise_eviction_score(None) == 0.0


# --- heap: only three non-null blocks (null + 3) -----------------------------


@pytest.fixture
def pool3() -> BlockPool:
    """null + exactly 3 KV rows — all free except null after init."""
    return _pool(n=4, enable_caching=True)


def test_heap_prefers_higher_score_among_uncached_free(pool3: BlockPool):
    pool = pool3
    a, b, c = pool.get_new_blocks(3)
    a.cachewise_policy = _policy(100.0)
    b.cachewise_policy = _policy(50.0)
    c.cachewise_policy = _policy(0.0)
    pool.free_blocks([c, a, b])

    out = [pool.get_new_blocks(1)[0].block_id for _ in range(3)]
    assert out == [a.block_id, b.block_id, c.block_id]


def test_heap_lru_tiebreaker_same_score_fifo_free_order(pool3: BlockPool):
    pool = pool3
    first_free, second_free = pool.get_new_blocks(2)
    first_free.cachewise_policy = _policy(42.0)
    second_free.cachewise_policy = _policy(42.0)
    pool.free_blocks([first_free, second_free])

    x = pool.get_new_blocks(1)[0]
    y = pool.get_new_blocks(1)[0]
    assert x.block_id == first_free.block_id
    assert y.block_id == second_free.block_id


def test_heap_prefers_higher_score_among_prefix_cached_free(pool3: BlockPool):
    pool = pool3
    hi, mid, lo = pool.get_new_blocks(3)
    _install_prefix_cached_block(pool, hi, ground_truth_idle_s=100.0)
    _install_prefix_cached_block(pool, mid, ground_truth_idle_s=50.0)
    _install_prefix_cached_block(pool, lo, ground_truth_idle_s=0.0)
    pool.free_blocks([lo, hi, mid])

    out = [pool.get_new_blocks(1)[0].block_id for _ in range(3)]
    assert out == [hi.block_id, mid.block_id, lo.block_id]


def test_heap_prefers_uncached_over_prefix_cached_even_if_cached_score_higher(
    pool3: BlockPool,
):
    pool = pool3
    uncached, cached_slot = pool.get_new_blocks(2)
    uncached.cachewise_policy = _policy(1.0)
    _install_prefix_cached_block(pool, cached_slot, ground_truth_idle_s=999.0)
    pool.free_blocks([uncached, cached_slot])

    first = pool.get_new_blocks(1)[0]
    assert first.block_id == uncached.block_id


def test_touch_removes_from_free_set_heap_skips_stale_entries():
    # null + blocks 1,2 only — no extra free row with a stale tiny heap seq.
    pool = _pool(n=3, enable_caching=True)
    b_keep, b_touch = pool.get_new_blocks(2)
    b_keep.cachewise_policy = _policy(0.0)
    b_touch.cachewise_policy = _policy(999.0)
    pool.free_blocks([b_keep, b_touch])

    pool.touch([b_touch])

    out = pool.get_new_blocks(1)[0]
    assert out.block_id == b_keep.block_id
    assert b_touch.ref_cnt == 1


# --- only null + one physical block -----------------------------------------


def test_get_new_blocks_evicts_prefix_cache_for_reused_slot():
    pool = _pool(n=2, enable_caching=True)
    blk = pool.get_new_blocks(1)[0]
    key = _install_prefix_cached_block(pool, blk, ground_truth_idle_s=7.0)
    assert pool.cached_block_hash_to_block.get_one_block(key) is blk
    pool.free_blocks([blk])

    again = pool.get_new_blocks(1)[0]
    assert again.block_id == blk.block_id
    assert again.block_hash is None
    assert again.cachewise_policy is None
    assert pool.cached_block_hash_to_block.get_one_block(key) is None


def test_three_prefix_cached_frees_then_alloc_evicts_all_from_map():
    pool = _pool(n=4, enable_caching=True)
    blocks = pool.get_new_blocks(3)
    keys = [_install_prefix_cached_block(pool, b, ground_truth_idle_s=float(i))
            for i, b in enumerate(blocks)]
    for k in keys:
        assert pool.cached_block_hash_to_block.get_one_block(k) is not None
    pool.free_blocks(blocks)

    pool.get_new_blocks(3)
    for k in keys:
        assert pool.cached_block_hash_to_block.get_one_block(k) is None


# --- reset_prefix_cache: fresh pool, everything free except null ------------


def test_reset_prefix_cache_rebuilds_heap_and_clears_hashes():
    pool = _pool(n=4, enable_caching=True)
    a, b = pool.get_new_blocks(2)
    ka = _install_prefix_cached_block(pool, a, ground_truth_idle_s=10.0)
    kb = _install_prefix_cached_block(pool, b, ground_truth_idle_s=20.0)
    pool.free_blocks([a, b])

    assert pool.reset_prefix_cache() is True

    assert pool.cached_block_hash_to_block.get_one_block(ka) is None
    assert pool.cached_block_hash_to_block.get_one_block(kb) is None
    assert a.block_hash is None and b.block_hash is None
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    assert len(pool._free_ids) == pool.get_num_free_blocks()


# --- enable_caching=False: only two non-null rows ---------------------------


def test_caching_disabled_scores_zero_tier_and_seq_still_order():
    pool = _pool(n=3, enable_caching=False)
    uncached, cached_slot = pool.get_new_blocks(2)
    uncached.cachewise_policy = _policy(999.0)
    _install_prefix_cached_block(pool, cached_slot, ground_truth_idle_s=0.0)
    pool.free_blocks([uncached, cached_slot])

    first = pool.get_new_blocks(1)[0]
    assert first.block_id == uncached.block_id


# --- stress -----------------------------------------------------------------


def test_many_alloc_free_rounds_free_ids_matches_dll_count():
    pool = _pool(n=48, enable_caching=True)
    for _ in range(30):
        blocks = pool.get_new_blocks(6)
        for i, b in enumerate(blocks):
            b.cachewise_policy = _policy(float(i))
        pool.free_blocks(blocks)
    assert len(pool._free_ids) == pool.get_num_free_blocks()
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1


def test_rebuild_heap_matches_free_set_size():
    pool = _pool(n=4, enable_caching=True)
    pool._rebuild_free_heap()
    assert len(pool._free_heap) == len(pool._free_ids)
    assert len(pool._free_ids) == pool.get_num_free_blocks()