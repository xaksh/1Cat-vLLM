import json, time, urllib.request

def run(prompt, max_tok, temp):
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
    dt = t1 - t0
    ct = usage["completion_tokens"]
    tps = ct / dt
    print("completion_tokens=%d wall_s=%.2f tok/s=%.2f finish_reason=%s" % (
        ct, dt, tps, d["choices"][0]["finish_reason"]))
    return d["choices"][0]["message"]["content"]

prompt = "Write a short story about a robot who discovers a hidden talent for painting, and how it changes its relationship with the humans in the workshop."
run(prompt, 50, 0.3)  # warmup
c = None
for _ in range(3):
    c = run(prompt, 400, 0.3)
print("\n--- last response (first 600 chars) ---")
print(c[:600])
