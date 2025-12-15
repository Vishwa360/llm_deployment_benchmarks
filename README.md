# LLM Serving Benchmark on Lightning AI: vLLM vs Ollama

This repo contains an end-to-end, reproducible benchmark I ran on **Lightning AI (Studio)** to compare:

- **vLLM** (OpenAI-compatible server)
- **Ollama** (OpenAI-compatible server)

I measure:
- **Latency** (avg / p50 / p90 / p95 / p99)
- **Throughput** (**requests/sec** and **tokens/sec**)
- Scaling across **concurrency** levels

---

## What I benchmarked

### Model (same family on both)
To keep things lightweight and easy to run on a single GPU:

- **vLLM:** `meta-llama/Llama-3.2-1B-Instruct` (HF weights)
- **Ollama:** `llama3.2:1b` (GGUF quantized variant)

> Note: Ollama typically serves a quantized build (e.g., Q8_0), while vLLM may use FP16/BF16. This can affect throughput/latency; I call it out in the results discussion.

---

## Repo structure

```text
.
├── bench_llama32_1b_vllm_vs_ollama.py       # benchmark runner (latency + throughput)
├── plot_llama32_1b_bench.py                 # plots (PNG) for article
├── llama32_1b_vllm_vs_ollama_results.csv    # generated results (after benchmark run)
└── README.md


⸻

Hardware / Environment
	•	Platform: Lightning AI Studio
	•	GPU: (Fill in: A10 / L4 / A100 / etc.)
	•	OS Image: Lightning Studio default
	•	Python: 3.x (recommend 3.10+)

Record the exact environment for your article:

nvidia-smi
python -V
pip show vllm ollama httpx pandas matplotlib


⸻

Setup on Lightning AI Studio

1) Create a Lightning Studio (GPU)
	•	Create a new AI Studio workspace
	•	Choose any GPU (A10/L4/A100). 1B models run comfortably on most.

2) Create a virtualenv + install deps

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install "vllm>=0.6.2" httpx pandas matplotlib


⸻

Start vLLM server

Open Terminal 1:

source .venv/bin/activate

MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"

vllm serve "$MODEL_ID" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9

Endpoint:
	•	http://localhost:8000/v1/chat/completions

⸻

Start Ollama server

Open Terminal 2:

curl -fsSL https://ollama.com/install.sh | sh

# Start server
OLLAMA_HOST=0.0.0.0:11434 ollama serve

Open Terminal 3:

# Pull the lightweight 1B model
OLLAMA_HOST=0.0.0.0:11434 ollama pull llama3.2:1b

Endpoint:
	•	http://localhost:11434/v1/chat/completions

⸻

Run the benchmark

Benchmark config

In bench_llama32_1b_vllm_vs_ollama.py you can edit:
	•	DURATION_SEC (default: 60s per concurrency)
	•	CONCURRENCY_LEVELS (default: [1,2,4,8,16,32])
	•	MAX_TOKENS (default: 256)
	•	Prompt set in PROMPTS

Execute

source .venv/bin/activate
python bench_llama32_1b_vllm_vs_ollama.py

Outputs:
	•	llama32_1b_vllm_vs_ollama_results.csv

CSV columns include:
	•	engine, model_name, concurrency
	•	avg_latency_ms, p50_latency_ms, p90_latency_ms, p95_latency_ms, p99_latency_ms
	•	requests_per_s, tokens_per_s
	•	num_requests, total_output_tokens, wall_time_s

⸻

Generate graphs (for article)

source .venv/bin/activate
python plot_llama32_1b_bench.py

Generated images:
	•	llama32_1b_throughput_vs_concurrency.png
	•	llama32_1b_p95_latency_vs_concurrency.png

⸻

Methodology notes (important for writing)

Request type
	•	Uses non-streaming chat.completions requests (stream=False)
	•	Measures end-to-end latency from request send → response received

Throughput
	•	requests_per_s = total requests / wall time
	•	tokens_per_s = total output tokens / wall time (uses usage when available)

If Ollama usage fields differ on your version, the script includes a fallback for alternate usage keys.

Fairness considerations
	•	Same model family and same general task prompts
	•	Quantization differences (Ollama GGUF vs vLLM FP16/BF16) are noted
	•	Concurrency scaling is the main focus

⸻

Troubleshooting

Port already in use

Change ports in the server commands:
	•	vLLM: --port 8001
	•	Ollama: set OLLAMA_HOST=0.0.0.0:11435

Update the base URLs inside the benchmark script accordingly.

Hugging Face model access

Some Meta models require HF authentication. If you get auth errors:

pip install -U huggingface_hub
huggingface-cli login

Or set HF_TOKEN in your environment.

Kernel / process crashes
	•	Reduce CONCURRENCY_LEVELS (e.g., stop at 16)
	•	Reduce MAX_TOKENS
	•	Use shorter prompts
	•	Ensure GPU memory is not exhausted (check nvidia-smi)

⸻

How to cite / reproduce in an article

Include:
	•	Lightning Studio GPU type + region (if relevant)
	•	vLLM version, Ollama version
	•	Model identifiers
	•	Benchmark script parameters (duration, concurrency, max_tokens)
	•	Graphs: throughput vs concurrency, p95/p99 latency vs concurrency

⸻

License

Choose one:
	•	MIT
	•	Apache-2.0
	•	Or leave unlicensed for now

⸻

Acknowledgements
	•	Lightning AI (Studio runtime)
	•	vLLM project
	•	Ollama project

If you want, paste your **actual GPU type** (A10/L4/A100) + the **CSV** you generated, and I’ll update the README with a “Results” section containing a clean table + the exact numbers you observed.
