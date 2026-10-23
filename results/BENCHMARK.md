# Local benchmark: 18 languages, 360 queries + 90 negatives per cell
# Models: bge-m3-reranker-v2 (568M), jina-reranker-v2-base-multilingual (278M)
# Devices: CPU (llama.cpp, -ngl 0) vs GPU (GTX 1050 Ti 4GB, -ngl auto)
# GTR-multilingual excluded: GGUF arch 'new' unsupported by llama.cpp
# Compiled from results/*.jsonl (harness: tools/benchmark_rerank.py)

| model | mode | recall@5 | wrong | noise | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|
| baseline (off) | off | 357/360 | 1 | 0/90 | 4 | 7 |
| bge-fp16 [gpu] | gate_blend | 355/360 | 3 | 3/90 | 69 | 144 |
| bge-fp16 [gpu] | gate_only | 355/360 | 2 | 3/90 | 69 | 142 |
| bge-fp16 [gpu] | off | 357/360 | 1 | 0/90 | 4 | 9 |
| bge-fp16 [gpu] | reorder_blend | 357/360 | 1 | 0/90 | 69 | 141 |
| bge-fp16 [gpu] | reorder_only | 357/360 | 1 | 0/90 | 68 | 140 |
| bge-q4 [cpu] | gate_blend | 357/360 | 3 | 3/90 | 237 | 571 |
| bge-q4 [cpu] | gate_only | 357/360 | 2 | 3/90 | 230 | 563 |
| bge-q4 [cpu] | off | 357/360 | 1 | 0/90 | 4 | 7 |
| bge-q4 [cpu] | reorder_blend | 357/360 | 1 | 0/90 | 231 | 554 |
| bge-q4 [cpu] | reorder_only | 357/360 | 1 | 0/90 | 242 | 583 |
| bge-q4 [gpu] | gate_blend | 357/360 | 3 | 3/90 | 39 | 80 |
| bge-q4 [gpu] | gate_only | 357/360 | 2 | 3/90 | 39 | 80 |
| bge-q4 [gpu] | off | 357/360 | 1 | 0/90 | 4 | 6 |
| bge-q4 [gpu] | reorder_blend | 357/360 | 1 | 0/90 | 39 | 79 |
| bge-q4 [gpu] | reorder_only | 357/360 | 1 | 0/90 | 39 | 80 |
| bge-q8 [cpu] | gate_blend | 356/360 | 3 | 3/90 | 280 | 712 |
| bge-q8 [cpu] | gate_only | 356/360 | 2 | 3/90 | 281 | 700 |
| bge-q8 [cpu] | off | 357/360 | 1 | 0/90 | 4 | 7 |
| bge-q8 [cpu] | reorder_blend | 357/360 | 1 | 0/90 | 279 | 757 |
| bge-q8 [cpu] | reorder_only | 357/360 | 1 | 0/90 | 284 | 701 |
| jina-fp16 [gpu] | gate_blend | 357/360 | 3 | 3/90 | 26 | 51 |
| jina-fp16 [gpu] | gate_only | 357/360 | 1 | 3/90 | 26 | 53 |
| jina-fp16 [gpu] | off | 357/360 | 1 | 0/90 | 4 | 7 |
| jina-fp16 [gpu] | reorder_blend | 357/360 | 1 | 0/90 | 26 | 49 |
| jina-fp16 [gpu] | reorder_only | 357/360 | 1 | 0/90 | 27 | 51 |
| jina-q8 [cpu] | gate_blend | 357/360 | 3 | 3/90 | 85 | 202 |
| jina-q8 [cpu] | gate_only | 357/360 | 1 | 3/90 | 87 | 206 |
| jina-q8 [cpu] | off | 357/360 | 1 | 0/90 | 4 | 7 |
| jina-q8 [cpu] | reorder_blend | 357/360 | 1 | 0/90 | 88 | 220 |
| jina-q8 [cpu] | reorder_only | 357/360 | 1 | 0/90 | 88 | 214 |
