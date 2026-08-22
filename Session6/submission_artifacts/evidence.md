# Evidence Bundle

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | PASS | tokenizer_manifest.json hash re-verified on load: sha256:32f4aec77e728a6613c014374d6b3e5efa81b301640aab4a9157783ebeb1df83 |
| Evaluation firewall | PASS | 2 document(s) blocked by content hash -- see [PASS] eval_shard_blocked above |
| Packing correctness | PASS | packing_utilization=1.0000, avg_segments_per_sample=1.06 over steps [0,165) (from consumption-ledger token spans) |
| Useful token accounting | PASS | ledger-derived estimate=187096, recomputed exact=185972 (from a full Packer/BatchAssembler replay over steps [0,165) -- served-token totals agree: 188416) |
| Mixture compliance | PASS | planned vs. actual per-lane shares over steps [0,165): {"code": {"planned_share": 0.2642424242424243, "actual_share": 0.2765151515151515, "delta": 0.012272727272727213}, "general_web": {"planned_share": 0.2559090909090909, "actual_share": 0.23712121212121212, "delta": -0.018787878787878798}, "indic": {"planned_share": 0.19075757575757574, "actual_share": 0.18409090909090908, "delta": -0.006666666666666654}, "instruction": {"planned_share": 0.08939393939393918, "actual_share": 0.09318181818181819, "delta": 0.0037878787878790066}, "math_science": {"planned_share": 0.1996969696969698, "actual_share": 0.20909090909090908, "delta": 0.009393939393939288}} |
| OPUS audit trail | PASS | 850/850 candidates accepted, decisions_hash=sha256:9d40d846b40dde9cc5d41c8e73a8e5fbd76ad0493fe3f04e62a566a34b84f909 |
| Crash recovery | PASS | expected batch ids at step 31 == resumed batch ids: ['mb-31-0', 'mb-31-1'] |
| Replay | PASS | steps [0,39) recomputed from scratch and hash-compared against the original consumption ledger; mismatched_steps=[] |
| Learning trace | PASS | 700 learning-ledger entries, each linked to a real manifest shard_id |
| Fork lineage | PASS | branch 'fork-1' full history reconstructed across ['main', 'fork-1'] (fork at step 112): 936 samples over steps [0,117); diverged from parent at 4/4 post-fork steps (different seed, same starting checkpoint) |
| Throughput | PASS | 10522.4 tokens/sec, 10385.9 useful tokens/sec over 165 steps |

**Overall: PASS**
