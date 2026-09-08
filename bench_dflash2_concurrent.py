import json, time, urllib.request, threading

PROMPTS = [
    "Write a short story about a robot who discovers a hidden talent for "
    "painting, and how it changes its relationship with the humans in the "
    "workshop.",
    "Write a short story about a lighthouse keeper who discovers something "
    "unusual washed ashore one stormy night.",
]

def run(prompt, max_tok, temp, results, idx):
    body = json.dumps({
        "model": "qwen3.8-27b-dflash2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": max_tok,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        d = json.load(resp)
    t1 = time.perf_counter()
    usage = d["usage"]
    results[idx] = {
        "completion_tokens": usage["completion_tokens"],
        "wall_s": t1 - t0,
        "finish_reason": d["choices"][0]["finish_reason"],
    }

def run_concurrent(max_tok, temp, label):
    results = [None, None]
    threads = [
        threading.Thread(target=run, args=(PROMPTS[i], max_tok, temp, results, i))
        for i in range(2)
    ]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_end = time.perf_counter()
    wall = t_end - t_start
    total_tokens = sum(r["completion_tokens"] for r in results)
    print(f"--- {label} ---")
    for i, r in enumerate(results):
        print(f"  stream {i}: completion_tokens={r['completion_tokens']} "
              f"own_wall_s={r['wall_s']:.2f} own_tok/s={r['completion_tokens']/r['wall_s']:.2f} "
              f"finish_reason={r['finish_reason']}")
    print(f"  batch wall_s={wall:.2f}  aggregate_tok/s={total_tokens/wall:.2f} "
          f"(sum of {total_tokens} tokens across 2 concurrent streams)")

# warmup (serial, small)
run(PROMPTS[0], 50, 0.3, [None, None], 0)

for trial in range(3):
    run_concurrent(400, 0.3, f"trial {trial+1}: 2 concurrent streams")
