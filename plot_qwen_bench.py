import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("qwen_vllm_vs_ollama_results.csv")

# make engine names nicer for plots
df["engine_label"] = df["engine"].map(
    {"vllm": "vLLM (Qwen2-VL-2B)", "ollama": "Ollama (Qwen2.5-VL-7B)"}
)

# --- Throughput vs concurrency ---
plt.figure()
for engine, group in df.groupby("engine_label"):
    group = group.sort_values("concurrency")
    plt.plot(group["concurrency"], group["tokens_per_s"], marker="o", label=engine)

plt.xlabel("Concurrency (number of parallel requests)")
plt.ylabel("Throughput (output tokens / second)")
plt.title("Qwen2-VL on vLLM vs Qwen2.5-VL on Ollama\nThroughput vs Concurrency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("qwen_throughput_vs_concurrency.png", dpi=200)

# --- P95 latency vs concurrency ---
plt.figure()
for engine, group in df.groupby("engine_label"):
    group = group.sort_values("concurrency")
    plt.plot(group["concurrency"], group["p95_latency_ms"], marker="o", label=engine)

plt.xlabel("Concurrency (number of parallel requests)")
plt.ylabel("P95 latency (ms)")
plt.title("Qwen2-VL on vLLM vs Qwen2.5-VL on Ollama\nP95 Latency vs Concurrency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("qwen_p95_latency_vs_concurrency.png", dpi=200)

print("Saved plots: qwen_throughput_vs_concurrency.png, qwen_p95_latency_vs_concurrency.png")