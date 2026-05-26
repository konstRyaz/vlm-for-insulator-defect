# Existing Qwen sweep summary

The already completed clean Qwen sweep suggests that simply scaling within the frozen Qwen family did not solve the main semantic bottleneck.

| model | coarse acc | macro-F1 | main pattern |
|---|---:|---:|---|
| Qwen2.5-VL-3B | 0.4828 | 0.2946 | balanced but still weak flashover/ok discrimination |
| Qwen2.5-VL-7B | 0.5000 | 0.1556 | strong normal bias; flashover and broken recall collapse |
| Qwen3-VL-4B | 0.5345 | 0.2748 | higher raw accuracy but poor flashover recall |
| Qwen2.5-VL-7B-AWQ | not completed | n/a | preflight failed in earlier run |
| Qwen3-VL-2B | not completed | n/a | schema validation failed in earlier run |

This makes the next comparison useful: try non-Qwen open VLMs as frozen backbones, but keep the same clean prompt contract and metrics.
