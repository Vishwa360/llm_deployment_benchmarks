import asyncio
import time
import math
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import httpx
import pandas as pd


# ---------- CONFIG ----------
PROMPT = (
    "Explain self-attention in transformers to a senior ML engineer. "
    "Focus on the mathematical formulation and inference-time performance implications."
)

# you can duplicate this list or add more prompts
PROMPTS = [PROMPT]

DURATION_SEC = 60          # how long to run per concurrency level
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
TEMPERATURE = 0.1
MAX_TOKENS = 256


@dataclass
class RequestMetric:
    engine: str
    model_name: str
    concurrency: int
    start_ts: float
    end_ts: float
    latency_ms: float
    input_tokens: int
    output_tokens: int


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


async def run_worker(
    client: httpx.AsyncClient,
    engine: str,
    model_name: str,
    base_url: str,
    concurrency: int,
    metrics: List[RequestMetric],
    end_time: float,
) -> None:
    """Continuously sends requests until end_time, collects per-request metrics."""
    i = 0
    while time.time() < end_time:
        prompt = PROMPTS[i % len(PROMPTS)]
        i += 1

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }

        url = base_url.rstrip("/") + "/v1/chat/completions"

        start = time.time()
        try:
            resp = await client.post(url, json=payload, timeout=120.0)
            end = time.time()
            latency_ms = (end - start) * 1000.0
            data = resp.json()

            # vLLM: OpenAI-style usage
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # Ollama: may use a different usage schema; fallback
            if not input_tokens and not output_tokens:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

            metrics.append(
                RequestMetric(
                    engine=engine,
                    model_name=model_name,
                    concurrency=concurrency,
                    start_ts=start,
                    end_ts=end,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )

        except Exception as e:
            print(f"[{engine} c={concurrency}] error: {e}")
            # continue loop


async def run_benchmark_for_target(
    engine: str,
    model_name: str,
    base_url: str,
    concurrencies: List[int],
) -> pd.DataFrame:
    all_rows = []

    async with httpx.AsyncClient() as client:
        for c in concurrencies:
            print(f"\n=== {engine}: concurrency {c} ===")
            metrics: List[RequestMetric] = []

            end_time = time.time() + DURATION_SEC
            tasks = [
                asyncio.create_task(
                    run_worker(
                        client=client,
                        engine=engine,
                        model_name=model_name,
                        base_url=base_url,
                        concurrency=c,
                        metrics=metrics,
                        end_time=end_time,
                    )
                )
                for _ in range(c)
            ]

            await asyncio.gather(*tasks)

            if not metrics:
                print("No successful requests for this setting.")
                continue

            # aggregate
            latencies = [m.latency_ms for m in metrics]
            total_time = max(m.end_ts for m in metrics) - min(
                m.start_ts for m in metrics
            )
            total_requests = len(metrics)
            total_out_tokens = sum(m.output_tokens for m in metrics) or float("nan")

            row = {
                "engine": engine,
                "model_name": model_name,
                "concurrency": c,
                "num_requests": total_requests,
                "avg_latency_ms": sum(latencies) / len(latencies),
                "p50_latency_ms": percentile(latencies, 50),
                "p90_latency_ms": percentile(latencies, 90),
                "p95_latency_ms": percentile(latencies, 95),
                "p99_latency_ms": percentile(latencies, 99),
                "wall_time_s": total_time,
                "requests_per_s": total_requests / total_time if total_time > 0 else float("nan"),
                "tokens_per_s": total_out_tokens / total_time if total_time > 0 else float("nan"),
                "total_output_tokens": total_out_tokens,
            }
            print(json.dumps(row, indent=2))
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def main():
    # vLLM target
    vllm_engine = "vllm"
    vllm_model = "Qwen/Qwen2-VL-2B-Instruct"
    vllm_base = "http://localhost:8000"

    # Ollama target
    ollama_engine = "ollama"
    ollama_model = "aleSuglia/qwen2-vl-2b-instruct-q4_k_m"
    ollama_base = "http://localhost:11434"

    df_vllm = asyncio.run(
        run_benchmark_for_target(vllm_engine, vllm_model, vllm_base, CONCURRENCY_LEVELS)
    )

    df_ollama = asyncio.run(
        run_benchmark_for_target(
            ollama_engine, ollama_model, ollama_base, CONCURRENCY_LEVELS
        )
    )

    df_all = pd.concat([df_vllm, df_ollama], ignore_index=True)
    df_all.to_csv("qwen_vllm_vs_ollama_results.csv", index=False)
    print("\nSaved results to qwen_vllm_vs_ollama_results.csv")
    print(df_all)


if __name__ == "__main__":
    main()