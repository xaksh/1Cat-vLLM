import json, time, urllib.request

def run(prompt, max_tok, temp=0.0):
    body = json.dumps({
        "model": "qwen3.8-27b-nvfp4",
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

prompt = "Write a Python implementation of quicksort with detailed comments explaining each step, then explain the time complexity in different cases."
run(prompt, 50)  # warmup
for _ in range(2):
    run(prompt, 512)
