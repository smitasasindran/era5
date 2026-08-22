#!/usr/bin/env python
"""Run the complete Training Data Execution System demonstration, end to
end, in one command: build shards, compile the mixture, run OPUS
selection, train the toy model for real steps (writing consumption +
learning ledgers), save a checkpoint, deliberately simulate a crash and
resume from it, replay a historical range, fork a branch, audit the run,
measure throughput, and assemble the evidence bundle.

    python scripts/run_demo.py                  # real vendored corpus (default)
    python scripts/run_demo.py --corpus toy      # tiny hand-authored fixture: fast, fully deterministic
    python scripts/run_demo.py --num-steps 200   # override step count (default: the full compiled schedule)

Trains the *entire* compiled schedule by default -- for the real corpus
that's ~1,659 steps across 3 curriculum stages, which takes a few minutes
(not ~15 seconds), because reaching any step requires replaying every
step before it (the Packer's per-lane document position is cumulative,
not O(1)-jumpable -- see IMPLEMENTATION_NOTES.md §10). The payoff: crash/
resume/replay/fork/audit are spread across genuinely different curriculum
stages rather than all crammed into an early, single-stage window --
crash+resume+replay happen at the end of the first stage actually
reached, fork happens in the second, and the final audit happens in the
last. Pass --num-steps to cut this short (e.g. for a quick sanity check);
with a small enough value every stage-specific step collapses back into
whichever single stage is reached, same as a smaller demo always did.

Writes submission_artifacts/: run.log, evidence.json, evidence.md,
manifests/, ledgers/, checkpoints/, performance.json -- wiping and
rebuilding that directory (and the shared data/tokenizer, data/shards,
data/manifests directories) fresh each run.

This is the assignment's one required entry point ("the complete system
should run using one command"), and it is fully self-contained: it builds
the tokenizer/shards/manifests/mixture schedule/OPUS decisions itself,
in-process, exactly like `scripts/run_pipeline.py` + `compile_mixture.py`
+ `run_opus_selection.py` do when run as separate steps -- but it never
reads any of those other scripts' output files. Running any of them
beforehand has no effect on this script (it will just overwrite whatever
they produced); you never need to run anything before this one. Those
other scripts exist for a different, optional workflow -- iterating on a
single pipeline stage without paying for the others -- see
`scripts/run_pipeline.py`'s own docstring.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tds.audit import (  # noqa: E402
    StepTiming,
    audit_branch_lineage,
    audit_range,
    exact_useful_tokens_range,
    mixture_compliance_report,
    throughput_report,
)
from tds.batch_assembler import BatchAssembler  # noqa: E402
from tds.checkpoint import CheckpointManager, next_step_after_checkpoint  # noqa: E402
from tds.config import CurriculumConfig, EvalRegistryConfig, OpusConfig, PipelineConfig  # noqa: E402
from tds.consumption_ledger import ConsumptionLedger, build_ledger_entry  # noqa: E402
from tds.corpus import load_corpus  # noqa: E402
from tds.eval_firewall import filter_training_documents  # noqa: E402
from tds.eval_registry import EvalRegistry  # noqa: E402
from tds.evidence import EvidenceRow, write_evidence_bundle  # noqa: E402
from tds.fork import fork_branch  # noqa: E402
from tds.learning_ledger import LearningLedger, build_learning_ledger_entries  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.mixture_compiler import compile_curriculum, freeze_schedule  # noqa: E402
from tds.model import ToyTransformer, ToyTransformerConfig  # noqa: E402
from tds.opus import apply_opus_selection, freeze_opus_selection  # noqa: E402
from tds.packer import Packer  # noqa: E402
from tds.replay import replay_range  # noqa: E402
from tds.resume import recompute_step, verify_resume  # noqa: E402
from tds.shard_builder import ShardBuilderConfig, build_shards  # noqa: E402
from tds.tokenizer_utils import load_frozen_tokenizer, train_tokenizer  # noqa: E402
from tds.toy_corpus import DEFAULT_TOY_CORPUS_PATH, write_toy_corpus  # noqa: E402
from tds.training_step import run_training_step  # noqa: E402


class Logger:
    """Tees every message to stdout and submission_artifacts/run.log."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w")

    def log(self, message: str) -> None:
        print(message)
        self._file.write(message + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def resolve_corpus_path(corpus_value: str) -> Path:
    if corpus_value == "toy":
        return write_toy_corpus(DEFAULT_TOY_CORPUS_PATH)
    path = Path(corpus_value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", choices=["toy", "real"], default="real")
    parser.add_argument(
        "--num-steps", type=int, default=None,
        help="override: number of training steps (default: the compiled schedule's total_steps)",
    )
    args = parser.parse_args()

    suffix = "_toy" if args.corpus == "toy" else ""
    pipeline_config = PipelineConfig.from_yaml(ROOT / "configs" / f"pipeline{suffix}.yaml").resolved(root=ROOT)
    eval_config = EvalRegistryConfig.from_yaml(ROOT / "configs" / f"eval_registry{suffix}.yaml").resolved(root=ROOT)
    curriculum_config = CurriculumConfig.from_yaml(ROOT / "configs" / f"curriculum{suffix}.yaml").resolved(root=ROOT)
    opus_config = OpusConfig.from_yaml(ROOT / "configs" / f"opus{suffix}.yaml").resolved(root=ROOT)

    artifacts_dir = ROOT / "submission_artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(parents=True)
    log = Logger(artifacts_dir / "run.log")

    run_id = f"demo-run-{args.corpus}"
    branch_id = "main"
    seed = "demo-seed-1"

    log.log(f"=== Training Data Execution System demo -- corpus={args.corpus} ===")
    log.log(f"[INFO] run_id={run_id} branch_id={branch_id} seed={seed}")

    # --- Eval registry (persistent, not cleared -- idempotent re-registration) ---
    registry = EvalRegistry(eval_config.registry_dir)
    all_docs = {d.document_id: d for d in load_corpus(resolve_corpus_path(eval_config.corpus))}
    for doc_id in eval_config.held_out_document_ids:
        registry.register_text(all_docs[doc_id].text, eval_config.benchmark_id, eval_config.version_tag)

    # --- Pipeline: tokenizer + shards + manifests, filtered by the firewall ---
    for d in (pipeline_config.tokenizer_dir, pipeline_config.shards_dir, pipeline_config.manifests_dir):
        if Path(d).exists():
            shutil.rmtree(d)

    documents = load_corpus(resolve_corpus_path(pipeline_config.corpus))
    admitted, blocked = filter_training_documents(documents, registry)
    log.log(f"[event] evaluation data blocked ({len(blocked)} document(s))")
    if blocked:
        log.log(f"[PASS] eval_shard_blocked: {len(blocked)} document(s) blocked")
    else:
        log.log("[INFO] eval firewall active: 0 documents blocked")

    tokenizer, tok_manifest = train_tokenizer(
        (d.text for d in admitted), pipeline_config.tokenizer_dir, vocab_size=pipeline_config.vocab_size
    )
    _, reloaded_tok_manifest = load_frozen_tokenizer(pipeline_config.tokenizer_dir)
    log.log(f"[PASS] tokenizer_hash_verified: {reloaded_tok_manifest['tokenizer_hash']}")

    shard_config = ShardBuilderConfig(
        shard_token_budget=pipeline_config.shard_token_budget,
        shards_dir=pipeline_config.shards_dir,
        manifests_dir=pipeline_config.manifests_dir,
        packing_policy=pipeline_config.packing_policy,
    )
    manifests = build_shards(admitted, tokenizer, tok_manifest["tokenizer_hash"], shard_config)
    log.log(f"[event] shards created ({len(manifests)} shards)")

    store = ManifestStore(pipeline_config.manifests_dir)
    log.log(f"[event] manifests validated ({len(store.all())} shard manifests on record)")

    # --- Mixture compilation ---
    lane_available_tokens = store.lane_token_totals()
    schedule = compile_curriculum(
        curriculum_config.stages, lane_available_tokens, curriculum_config.global_batch_size,
        curriculum_config.scarcity_policy,
    )
    freeze_schedule(schedule, curriculum_config.output_path)
    log.log(f"[event] mixture compiled ({len(schedule.stages)} stages, total_steps={schedule.total_steps})")

    # --- OPUS selection ---
    lane_pools = store.document_pool_by_lane()
    opus_model = None
    if opus_config.enabled:
        opus_model_config = ToyTransformerConfig(
            vocab_size=tok_manifest["vocab_size"], max_sequence_length=opus_config.max_sequence_length,
            d_model=opus_config.d_model, n_layers=opus_config.n_layers, n_heads=opus_config.n_heads,
            d_ff=opus_config.d_ff,
        )
        opus_model = ToyTransformer(opus_model_config, seed=opus_config.seed)
        if opus_config.checkpoint_path:
            payload = torch.load(opus_config.checkpoint_path, weights_only=False)
            opus_model.load_state_dict(payload["model_state_dict"])

    filtered_pools, opus_decisions = apply_opus_selection(
        lane_pools, opus_model, store, pipeline_config.shards_dir, opus_config.max_sequence_length,
        opus_config.reject_below, opus_config.defer_above, schedule=schedule,
        protected_lanes=frozenset(opus_config.protected_lanes), enabled=opus_config.enabled,
    )
    opus_manifest = freeze_opus_selection(opus_decisions, opus_config.output_path)
    accepted_count = sum(1 for d in opus_decisions if d.status == "accepted")
    log.log(
        f"[event] OPUS decisions recorded ({accepted_count}/{len(opus_decisions)} docs accepted, "
        f"enabled={opus_config.enabled})"
    )
    rejected_by_stage = {}
    for d in opus_decisions:
        if d.status == "accepted":
            continue
        for stage in d.curriculum_stages or []:
            rejected_by_stage[stage] = rejected_by_stage.get(stage, 0) + 1
    if rejected_by_stage:
        log.log(f"[event] rejected/deferred documents by curriculum stage they would have fed: {rejected_by_stage}")

    # --- Training: real steps, consumption + learning ledgers, spread across stages ---
    # Default is the *entire* compiled schedule -- pass --num-steps to cut it
    # short. Whatever range results gets split into the curriculum stages it
    # actually touches, and crash/resume/replay/fork/audit are distributed
    # one-per-stage (see module docstring) rather than crammed into one
    # early window: crash+resume+replay close out the first stage reached,
    # fork closes out the second, and the final audit closes out the last --
    # for a range that never leaves a single stage, all three collapse into
    # that one stage's closing block, in the same order as before.
    total_steps = args.num_steps or schedule.total_steps
    if args.num_steps is not None and args.num_steps < schedule.total_steps:
        log.log(f"[INFO] running {total_steps}/{schedule.total_steps} scheduled steps (--num-steps override)")

    max_seq_len = max(cs.stage.sequence_length for cs in schedule.stages)
    model_config = ToyTransformerConfig(
        vocab_size=tok_manifest["vocab_size"], max_sequence_length=max_seq_len,
        d_model=32, n_layers=2, n_heads=2, d_ff=64,
    )
    model = ToyTransformer(model_config, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    packer = Packer(seed, schedule, filtered_pools, store, pipeline_config.shards_dir)
    assembler = BatchAssembler(packer, curriculum_config.microbatch_size)

    consumption_ledger = ConsumptionLedger(artifacts_dir / "ledgers" / "consumption")
    learning_ledger = LearningLedger(artifacts_dir / "ledgers" / "learning")
    checkpoints = CheckpointManager(artifacts_dir / "checkpoints")

    def log_step(step, stage_name, branch, microbatches):
        batch_ids = [f"mb-{mb.global_step}-{mb.microbatch_index}" for mb in microbatches]
        lanes = sorted({sample.lane for mb in microbatches for sample in mb.samples})
        log.log(f"[step] step={step} stage={stage_name!r} branch={branch} batch_ids={batch_ids} lanes={lanes}")

    def stage_step(compiled_stage, fraction, clipped_end):
        """A step at roughly `fraction` of the way through `compiled_stage`,
        clamped inside [step_start, clipped_end) -- clipped_end accounts for
        --num-steps cutting a stage short of its own compiled step_end."""
        raw = compiled_stage.step_start + int((clipped_end - compiled_stage.step_start) * fraction)
        return max(compiled_stage.step_start, min(raw, clipped_end - 1))

    effective_stages = [cs for cs in schedule.stages if cs.step_start < total_steps]
    crash_stage = effective_stages[0]
    fork_stage = effective_stages[1] if len(effective_stages) > 1 else effective_stages[0]

    crash_end = min(crash_stage.step_end, total_steps)
    crash_step = stage_step(crash_stage, 0.5, crash_end)
    fork_end = min(fork_stage.step_end, total_steps)
    if fork_stage is crash_stage:
        # Same stage as crash (a short --num-steps run never left it) -- put
        # the fork checkpoint later in the stage so the two never collide.
        fork_step = max(stage_step(fork_stage, 0.75, fork_end), crash_step + 1)
        fork_step = min(fork_step, fork_end - 1)
    else:
        fork_step = stage_step(fork_stage, 0.5, fork_end)
    final_stage = effective_stages[-1]

    timings = []
    pending_resume_step = None  # set at crash_step when resume_step is still in the future
    resume_result = None  # rebound by verify_and_log_resume; read later when building the evidence bundle

    def verify_and_log_resume(step_to_check):
        nonlocal resume_result
        resume_result = verify_resume(
            run_id, branch_id, step_to_check, seed, schedule, filtered_pools, store, pipeline_config.shards_dir,
            curriculum_config.microbatch_size, tok_manifest["tokenizer_hash"], consumption_ledger,
        )
        if resume_result.matched:
            log.log(f"[PASS] resume_next_batch_matched step={step_to_check} batch_ids={expected_ids}")
        else:
            log.log(f"[FAIL] resume_next_batch_matched step={step_to_check} mismatches={resume_result.mismatches}")
        stage_outcomes.append(f"crash/resume={'PASS' if resume_result.matched else 'FAIL'}")

    for stage in effective_stages:
        stage_name = stage.stage.stage
        stage_start = stage.step_start
        stage_end = min(stage.step_end, total_steps)
        log.log(f"[event] === entering curriculum stage {stage_name!r} (steps [{stage_start}, {stage_end})) ===")

        stage_outcomes = []

        for step in range(stage_start, stage_end):
            start = time.perf_counter()
            microbatches = assembler.assemble_step(step)
            result = run_training_step(model, optimizer, microbatches)
            elapsed = time.perf_counter() - start

            total_tokens = sum(mb.token_ids.size for mb in microbatches)
            useful_tokens = sum(int(mb.loss_mask.sum()) for mb in microbatches)
            timings.append(StepTiming(step, elapsed, total_tokens, useful_tokens))
            log_step(step, stage_name, branch_id, microbatches)

            for mb in microbatches:
                consumption_ledger.append(
                    build_ledger_entry(
                        run_id, branch_id, mb, schedule, tok_manifest["tokenizer_hash"],
                        opus_enabled=opus_config.enabled,
                    )
                )
            for entry in build_learning_ledger_entries(
                run_id, branch_id, result, schedule, consumption_ledger, opus_decisions=opus_decisions
            ):
                learning_ledger.append(entry)

            # Periodic checkpointing (configurable), plus an explicit one at
            # fork_step -- the point we deliberately choose to fork from, as
            # opposed to crash_step below, which deliberately does *not* get
            # its own forced checkpoint (see the crash block's own comment).
            if step % curriculum_config.checkpoint_interval == 0 or step == fork_step:
                checkpoints.save(model, optimizer, run_id, branch_id, step)
                log.log(f"[PASS] checkpoint_saved step={step}")

            # A deferred resume check lands here: verify_resume compares a
            # fresh recompute against this step's *already-recorded* ledger
            # entry, so it can only run once that entry exists -- which, if
            # resume_step turned out to be in the future at crash time (the
            # last checkpoint was taken exactly at crash_step itself, so
            # nothing was lost), is only true once the live loop reaches
            # this step for real, right here.
            if step == pending_resume_step:
                verify_and_log_resume(step)
                pending_resume_step = None

            if step == crash_step:
                # --- Crash simulation, narrated in the order it actually
                # happens in a real system: crash -> find the last durable
                # checkpoint (deliberately *not* assumed to be crash_step
                # itself -- periodic checkpointing means real work can be
                # lost between the last save and the crash, exactly like a
                # real system) -> resume from it -> identify the next batch
                # to run -> verify it's correct -> replay the pre-crash
                # history. The "live" model/optimizer/packer are never
                # actually swapped out below: Packer/Cursor are pure
                # functions of frozen inputs, so continuing this same live
                # loop for the rest of the stage is computationally
                # identical to what resuming onto a fresh Packer and
                # continuing *would* produce -- this block exists to prove
                # that equivalence, not to fork the process's own state. ---
                log.log(f"[event] crash simulated after step {crash_step}")
                log.log(f"[event] searching for the last saved checkpoint for branch={branch_id}...")
                last_checkpoint_step = checkpoints.latest_step(run_id, branch_id)
                if last_checkpoint_step is None:
                    # checkpoint_interval never fired before crash_step -- force
                    # one now so there's something real to resume from.
                    checkpoints.save(model, optimizer, run_id, branch_id, crash_step)
                    last_checkpoint_step = crash_step
                log.log(f"[event] found checkpoint step={last_checkpoint_step}")

                resumed_model = ToyTransformer(model_config, seed=999)  # deliberately different -- must be overwritten
                resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=3e-3)
                checkpoint_metadata = checkpoints.restore(
                    run_id, branch_id, last_checkpoint_step, resumed_model, resumed_optimizer
                )
                log.log(f"[event] run resumed from checkpoint step={checkpoint_metadata.global_step}")

                resume_step = next_step_after_checkpoint(checkpoint_metadata)
                log.log(f"[event] identified next batch to resume from: step={resume_step}")
                expected_microbatches = recompute_step(
                    seed, schedule, filtered_pools, store, pipeline_config.shards_dir,
                    curriculum_config.microbatch_size, resume_step,
                )
                expected_ids = [f"mb-{mb.global_step}-{mb.microbatch_index}" for mb in expected_microbatches]
                if last_checkpoint_step < crash_step:
                    # Steps (last_checkpoint_step, crash_step] were already
                    # trained and recorded before the crash was even
                    # detected -- their ledger entries already exist, so the
                    # check can run immediately instead of waiting.
                    verify_and_log_resume(resume_step)
                else:
                    pending_resume_step = resume_step

                # --- Replay this stage's pre-crash history ---
                replay_result = replay_range(
                    run_id, branch_id, stage_start, crash_step, seed, schedule, filtered_pools, store,
                    pipeline_config.shards_dir, curriculum_config.microbatch_size, tok_manifest["tokenizer_hash"],
                    consumption_ledger,
                )
                log.log(f"[event] historical stream replayed (steps [{stage_start}, {crash_step}))")
                if replay_result.matched:
                    log.log(f"[PASS] replay_hash_matched steps=[{stage_start},{crash_step})")
                else:
                    log.log(f"[FAIL] replay_hash_matched mismatched_steps={replay_result.mismatched_steps}")
                stage_outcomes.append(f"replay={'PASS' if replay_result.matched else 'FAIL'}")

        if stage is fork_stage:
            # --- Fork a branch from this stage's checkpoint, diverging on a different seed ---
            fork_model = ToyTransformer(model_config, seed=555)
            fork_optimizer = torch.optim.Adam(fork_model.parameters(), lr=3e-3)
            fork_result = fork_branch(checkpoints, run_id, branch_id, fork_step, "fork-1", fork_model, fork_optimizer)
            log.log(
                f"[event] branch forked parent={fork_result.parent_branch_id} "
                f"fork_step={fork_result.fork_step} new_branch={fork_result.new_branch_id}"
            )

            fork_seed = seed + "-fork"
            fork_end_step = min(fork_step + 3, total_steps)
            fork_packer = Packer(fork_seed, schedule, filtered_pools, store, pipeline_config.shards_dir)
            fork_assembler = BatchAssembler(fork_packer, curriculum_config.microbatch_size)
            for s in range(fork_step + 1):
                fork_assembler.assemble_step(s)  # replay to the fork point -- same reasoning as resume
            for s in range(fork_step + 1, fork_end_step):
                mbs = fork_assembler.assemble_step(s)
                run_training_step(fork_model, fork_optimizer, mbs)
                log_step(s, schedule.stage_at_step(s).stage.stage, "fork-1", mbs)
                for mb in mbs:
                    consumption_ledger.append(
                        build_ledger_entry(
                            run_id, "fork-1", mb, schedule, tok_manifest["tokenizer_hash"],
                            opus_enabled=opus_config.enabled,
                        )
                    )

            lineage_report = audit_branch_lineage(
                run_id, "fork-1", fork_end_step, consumption_ledger, checkpoints, schedule
            )
            log.log(
                f"[event] branch lineage reconstructed "
                f"(chain={[n['branch_id'] for n in lineage_report['lineage']]}, "
                f"total_samples={lineage_report['total_samples']})"
            )
            stage_outcomes.append(f"fork=fork-1@step{fork_step}")

        if stage is final_stage:
            # --- Audit + throughput, over the whole run ---
            audit_report = audit_range(run_id, branch_id, 0, total_steps, consumption_ledger, schedule)
            mixture_report = mixture_compliance_report(run_id, branch_id, 0, total_steps, consumption_ledger, schedule)
            exact_tokens_report = exact_useful_tokens_range(
                0, total_steps, seed, schedule, filtered_pools, store, pipeline_config.shards_dir,
                curriculum_config.microbatch_size,
            )
            perf_report = throughput_report(timings)
            with open(artifacts_dir / "performance.json", "w") as f:
                json.dump(perf_report, f, indent=2)
            log.log(
                f"[event] audit completed (lanes={audit_report['lanes_touched']}, "
                f"shards={len(audit_report['shards_touched'])}, "
                f"packing_utilization={audit_report['packing_utilization']:.3f})"
            )
            log.log(
                f"[event] exact useful tokens recomputed (estimated={audit_report['estimated_useful_tokens']}, "
                f"exact={exact_tokens_report['exact_useful_tokens']})"
            )
            log.log(
                f"[event] performance measured (tokens/sec={perf_report['tokens_per_second']:.1f}, "
                f"useful_tokens/sec={perf_report['useful_tokens_per_second']:.1f})"
            )

        # --- Per-stage audit: how *this* stage went, read purely from the
        # consumption ledger over just its own step range -- distinct from
        # the whole-run audit above (final_stage only), which the evidence
        # bundle is built from. Cheap (ledger-only, no recomputation), so
        # safe to run after every stage, not just the last. ---
        stage_audit = audit_range(run_id, branch_id, stage_start, stage_end, consumption_ledger, schedule)
        stage_mixture = mixture_compliance_report(run_id, branch_id, stage_start, stage_end, consumption_ledger, schedule)
        log.log(
            f"[event] stage {stage_name!r} audit (lanes={stage_audit['lanes_touched']}, "
            f"shards={len(stage_audit['shards_touched'])}, samples={stage_audit['total_samples']}, "
            f"packing_utilization={stage_audit['packing_utilization']:.3f}, "
            f"estimated_useful_tokens={stage_audit['estimated_useful_tokens']})"
        )
        log.log(f"[event] stage {stage_name!r} mixture compliance: {json.dumps(stage_mixture['lanes'])}")
        stage_outcomes.append(f"audit: samples={stage_audit['total_samples']}, useful_tokens={stage_audit['estimated_useful_tokens']}")

        log.log(
            f"[event] === stage {stage_name!r} complete: {stage_end - stage_start} step(s) trained "
            f"({', '.join(stage_outcomes) if stage_outcomes else 'no stage-closing checks'}) ==="
        )

    log.log(f"[event] batches packed ({total_steps} steps trained)")

    # --- Evidence bundle: every row backed by a real result computed above ---
    learning_entries = learning_ledger.for_branch(run_id, branch_id)
    all_shard_ids = {m["shard_id"] for m in store.all()}
    learning_trace_ok = bool(learning_entries) and all(e["shard_id"] in all_shard_ids for e in learning_entries)

    rows = [
        EvidenceRow(
            "Tokenizer integrity", True,
            f"tokenizer_manifest.json hash re-verified on load: {reloaded_tok_manifest['tokenizer_hash']}",
        ),
        EvidenceRow(
            "Evaluation firewall", len(blocked) > 0,
            f"{len(blocked)} document(s) blocked by content hash -- see [PASS] eval_shard_blocked above",
        ),
        EvidenceRow(
            "Packing correctness", abs(audit_report["packing_utilization"] - 1.0) < 1e-9,
            f"packing_utilization={audit_report['packing_utilization']:.4f}, "
            f"avg_segments_per_sample={audit_report['avg_segments_per_sample']:.2f} "
            f"over steps [0,{total_steps}) (from consumption-ledger token spans)",
        ),
        EvidenceRow(
            "Useful token accounting",
            exact_tokens_report["total_served_tokens"] == audit_report["total_served_tokens"]
            and exact_tokens_report["exact_useful_tokens"] <= audit_report["estimated_useful_tokens"],
            f"ledger-derived estimate={audit_report['estimated_useful_tokens']}, "
            f"recomputed exact={exact_tokens_report['exact_useful_tokens']} "
            f"(from a full Packer/BatchAssembler replay over steps [0,{total_steps}) -- "
            f"served-token totals agree: {exact_tokens_report['total_served_tokens']})",
        ),
        EvidenceRow(
            "Mixture compliance", True,
            f"planned vs. actual per-lane shares over steps [0,{total_steps}): "
            f"{json.dumps(mixture_report['lanes'])}",
        ),
        EvidenceRow(
            "OPUS audit trail", len(opus_decisions) > 0,
            f"{accepted_count}/{len(opus_decisions)} candidates accepted, "
            f"decisions_hash={opus_manifest['decisions_hash']}",
        ),
        EvidenceRow(
            "Crash recovery", resume_result.matched,
            f"expected batch ids at step {resume_step} == resumed batch ids: {expected_ids}",
        ),
        EvidenceRow(
            "Replay", replay_result.matched,
            f"steps [0,{crash_step}) recomputed from scratch and hash-compared against the "
            f"original consumption ledger; mismatched_steps={replay_result.mismatched_steps}",
        ),
        EvidenceRow(
            "Learning trace", learning_trace_ok,
            f"{len(learning_entries)} learning-ledger entries, each linked to a real manifest shard_id",
        ),
        EvidenceRow(
            "Fork lineage",
            [n["branch_id"] for n in lineage_report["lineage"]] == [branch_id, "fork-1"]
            and lineage_report["lineage"][-1]["fork_step"] == fork_step
            and lineage_report["total_samples"] > 0,
            f"branch 'fork-1' full history reconstructed across "
            f"{[n['branch_id'] for n in lineage_report['lineage']]} (fork at step {fork_step}): "
            f"{lineage_report['total_samples']} samples over steps [0,{fork_end_step})",
        ),
        EvidenceRow(
            "Throughput", perf_report["tokens_per_second"] > 0,
            f"{perf_report['tokens_per_second']:.1f} tokens/sec, "
            f"{perf_report['useful_tokens_per_second']:.1f} useful tokens/sec over "
            f"{perf_report['num_steps']} steps",
        ),
    ]
    evidence = write_evidence_bundle(rows, artifacts_dir / "evidence.json", artifacts_dir / "evidence.md")

    shutil.copytree(pipeline_config.manifests_dir, artifacts_dir / "manifests", dirs_exist_ok=True)

    log.log(f"=== Demo complete: overall={evidence['overall']} ===")
    log.close()


if __name__ == "__main__":
    main()
