"""Benchmark helpers: resume measurement and CI threshold logic."""
import copy
import os
import sys
from test_core import env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402


def good_result():
    return {
        "ann": {"sweep": [{"nprobe": 32, "rerank": 8, "recall@10": 0.97, "p50_ms": 2.0}]},
        "text_to_modality_p@10": {m: {"mean_p@10": 0.9} for m in ("sat", "dna", "cell")},
        "filtered_search_p50_ms": 2.0, "ingest": {"items_per_s": 9000.0},
        "agent": [{"grounded": True, "claims": {"verified": 5, "contradicted": []}}],
        "resume": {"dropped_batches": 4, "recomputed_batches": 4, "identical_after_resume": True, "second_resume_items": 0},
    }


def failing(mutate):
    R = copy.deepcopy(good_result()); mutate(R)
    return {c["check"] for c in bench.check_thresholds(R) if not c["ok"]}


def test_thresholds_pass_on_healthy_results():
    assert all(c["ok"] for c in bench.check_thresholds(good_result()))


def test_thresholds_catch_each_regression():
    assert failing(lambda R: R["ann"]["sweep"][0].update({"recall@10": 0.85})) == {"recall@10 (nprobe 32 + rerank)"}
    assert failing(lambda R: R["text_to_modality_p@10"]["dna"].update({"mean_p@10": 0.5})) == {"text->dna p@10"}
    assert failing(lambda R: R["ann"]["sweep"][0].update({"p50_ms": 80})) == {"search p50 ms"}
    assert failing(lambda R: R["ingest"].update({"items_per_s": 10})) == {"ingest items/s"}
    assert failing(lambda R: R["agent"][0].update({"grounded": False})) == {"agent runs grounded"}
    assert failing(lambda R: R["agent"][0]["claims"].update({"contradicted": [{}]})) == {"agent claims contradicted"}
    assert failing(lambda R: R["resume"].update({"recomputed_batches": 9})) == {"resume recomputes only dropped batches"}
    assert failing(lambda R: R["resume"].update({"identical_after_resume": False})) == {"resume output identical to original"}
    assert failing(lambda R: R["resume"].update({"second_resume_items": 1024})) == {"second resume recomputes nothing"}


def test_resume_check_after_simulated_crash():
    D, DESCS, ENG = env()
    r = bench.resume_check(DESCS, f"{D}/model.npz", f"{D}/stage", workers=2)
    assert r["dropped_batches"] == len(DESCS) - int(len(DESCS) * 0.9) > 0
    assert r["recomputed_batches"] == r["dropped_batches"] and r["skipped_batches"] == len(DESCS) - r["dropped_batches"]
    assert r["identical_after_resume"] and r["second_resume_items"] == 0
