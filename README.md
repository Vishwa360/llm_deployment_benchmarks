
# Benchmark: vLLM vs Ollama on Lightning AI (A100 80GB) — Qwen2-VL-2B

This repository contains a reproducible benchmark I ran on **Lightning AI** using an **NVIDIA A100 (80GB VRAM)** GPU cluster to compare:

- **vLLM** (OpenAI-compatible server)
- **Ollama** (OpenAI-compatible server)

The benchmark measures:
- **Latency** (avg / p50 / p90 / p95 / p99)
- **Throughput** (requests/sec and output tokens/sec)
- Scaling across multiple **concurrency** levels

---

## Models used (this run)

### vLLM
- Model: `Qwen/Qwen2-VL-2B-Instruct`
- Endpoint: `http://localhost:8000/v1/chat/completions`

### Ollama
- Model: `aleSuglia/qwen2-vl-2b-instruct-q4_k_m` (GGUF quantized)
- Endpoint: `http://localhost:11434/v1/chat/completions`

> Note: This is not perfectly apples-to-apples at the kernel level because Ollama serves a quantized GGUF build, while vLLM serves HF weights (typically FP16/BF16). The intent is to compare serving stacks and concurrency behavior.

---

## Hardware / environment

- Platform: Lightning AI
- GPU: NVIDIA A100 80GB VRAM (cluster)
- OS: Lightning Studio runtime
- Benchmark client: `httpx` + asyncio workers


```bash
nvidia-smi
python -V
pip show vllm ollama httpx pandas matplotlib
```

⸻

##Server configuration

###vLLM server
```bash
source .venv/bin/activate

MODEL_ID="Qwen/Qwen2-VL-2B-Instruct"

vllm serve "$MODEL_ID" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9

Ollama server

# Terminal 1: start ollama server
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Terminal 2: pull model
OLLAMA_HOST=0.0.0.0:11434 ollama pull aleSuglia/qwen2-vl-2b-instruct-q4_k_m
```

⸻

##Benchmark script

Benchmark runner: bench_qwen_vllm_vs_ollama.py

Key parameters (from script)
	•	Duration per concurrency level: DURATION_SEC = 60
	•	Concurrency sweep: CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
	•	Temperature: TEMPERATURE = 0.1
	•	Max output tokens: MAX_TOKENS = 256
	•	Prompt set: currently a single prompt (PROMPTS = [PROMPT])

Run benchmark

source .venv/bin/activate
python bench_qwen_vllm_vs_ollama.py

Outputs:
	•	qwen_vllm_vs_ollama_results.csv (script writes a CSV at the end)

CSV includes:
	•	latency: avg / p50 / p90 / p95 / p99 (ms)
	•	throughput: requests/sec and tokens/sec
	•	total requests and wall time

⸻

##Plotting

Once you have the CSV, run the plotting script (example):

python plot_qwen_bench.py

Outputs:
	•	qwen_throughput_vs_concurrency.png
	•	qwen_p95_latency_vs_concurrency.png

⸻

##Methodology notes
	•	API: POST /v1/chat/completions
	•	Non-streaming requests (stream=false)
	•	Latency: end-to-end wall clock per request
	•	Throughput:
	•	requests_per_s = total_requests / wall_time_s
	•	tokens_per_s = total_output_tokens / wall_time_s (from usage when available)

⸻

##Capturing logs (recommended)

I initially ran the benchmark from terminal without saving logs. For reproducibility, capture stdout/stderr:

mkdir -p logs
python bench_qwen_vllm_vs_ollama.py 2>&1 | tee logs/bench_run_$(date +%Y%m%d_%H%M%S).log

Also capture server logs:

# vLLM server
vllm serve ... 2>&1 | tee logs/vllm_server_$(date +%Y%m%d_%H%M%S).log

# Ollama server
OLLAMA_HOST=0.0.0.0:11434 ollama serve 2>&1 | tee logs/ollama_server_$(date +%Y%m%d_%H%M%S).log


⸻

##Limitations / fairness
	•	vLLM uses HF weights; Ollama uses GGUF quantized model → performance will differ due to quantization and runtime.
	•	These are VL models, but this run uses text-only prompts (no images).
	•	Single-node serving per engine (in this setup).

⸻

