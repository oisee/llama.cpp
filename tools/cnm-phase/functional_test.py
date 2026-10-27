#!/usr/bin/env python3
"""Functional test of runtime bias switching, MoE cache prefetch and the phase proxy (one llama-server launch).

usage: functional_test.py OUTDIR BIN_DIR MODEL PROFILES_DIR [CACHE_MIB] [HOST] [PORT]
Gates (each can fail):
  G1 selections share: the cold-selection share falls as the bias of the same profile goes ~0 -> -0.5 -> -30
  G2 profile switch changes routing: output at bias -30 differs from bias ~0
  G3 reset is exact: profile "none" (all-zero offsets) after other profiles reproduces the first request at server
     start (all-zero offsets, before any switch) and counts no cold selections
  G4 prefetch: after the same churn, misses on the same request are lower with prefetch than without
  G5 prefetch is invisible to results: the output with and without prefetch is identical (temp 0)
  G6 per-request timings carry cnm_moe_cache, /metrics carries cnm_* counters
  G7 proxy: a request whose last tool call is Edit switches the server to the edit profile
Refuses to start if another process uses the GPU.
"""
import hashlib, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

out = Path(sys.argv[1]); BIN = Path(sys.argv[2]); MODEL = sys.argv[3]; PROF = sys.argv[4]
CACHE = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
HOST = sys.argv[6] if len(sys.argv) > 6 else "127.0.0.1"
PORT = int(sys.argv[7]) if len(sys.argv) > 7 else 18491
out.mkdir(parents=True, exist_ok=True)
U = f"http://{HOST}:{PORT}"


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def save(name, obj):
    (out / name).write_text(json.dumps(obj, indent=1) + "\n")


def call(path, obj=None, timeout=1800):
    data = json.dumps(obj).encode() if obj is not None else None
    r = urllib.request.Request(U + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            raw = f.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    try:
        return 200, json.loads(raw)
    except ValueError:
        return 200, raw.decode(errors="replace")


apps = sh("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader")
if apps:
    print("GPU busy, refusing:", apps)
    sys.exit(2)

cmd = [str(BIN / "llama-server"), "-m", MODEL, "--host", HOST, "--port", str(PORT), "-c", "8192", "-np", "1",
       "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "999", "--n-cpu-moe", "48", "--moe-cache-mib", str(CACHE),
       "-t", "8", "-tb", "8", "--fit", "off", "--no-warmup", "--metrics", "--cnm-profiles", PROF, "--jinja"]
env = dict(os.environ, CNM_EXPERT_BIAS_RUNTIME="1", LD_LIBRARY_PATH=str(BIN))
save("command.json", {"argv": cmd, "env": {"CNM_EXPERT_BIAS_RUNTIME": "1"}, "pre": {
    "load": open("/proc/loadavg").read().strip(), "vram": sh("nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader")}})
log = open(out / "server.log", "w")
p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
res = {"gates": {}, "runs": [], "status": "RUNNING"}
proxy = None

PROMPT = ("You are reviewing an ABAP class. Explain step by step what the method below does, list possible bugs and "
          "propose a fixed version.\n\nMETHOD calculate_discount.\n  DATA lv_total TYPE p DECIMALS 2.\n"
          "  LOOP AT it_items INTO DATA(ls_item).\n    lv_total = lv_total + ls_item-price * ls_item-quantity.\n"
          "  ENDLOOP.\n  IF lv_total > 1000.\n    rv_discount = lv_total * '0.05'.\n  ELSEIF lv_total > 500.\n"
          "    rv_discount = lv_total * '0.02'.\n  ENDIF.\nENDMETHOD.\n")
CHURN = ("Write a long Python program that parses a CSV file of sensor readings, computes rolling averages, "
         "detects anomalies with a z-score and plots the result with matplotlib. Add docstrings and tests.\n")


def run(label, prompt, n=128):
    st, r = call("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "seed": 4242, "ignore_eos": True,
                                 "cache_prompt": False, "stream": False})
    assert st == 200, (st, r)
    t = r["timings"]
    c = t.get("cnm_moe_cache", {})
    rec = {"label": label, "content_sha256": hashlib.sha256(r["content"].encode()).hexdigest(),
           "predicted_n": t["predicted_n"], "tps": t["predicted_per_second"], "cnm": c,
           "cold_share": (c["selections_cold"] / c["selections"]) if c.get("selections") else None,
           "hit_rate": (c["hits"] / (c["hits"] + c["misses"])) if c.get("hits", 0) + c.get("misses", 0) else None}
    res["runs"].append(rec); save("result.json", res)
    print(label, json.dumps({k: rec[k] for k in ("content_sha256", "tps", "cold_share", "hit_rate")}),
          "misses", c.get("misses"), flush=True)
    return rec


def bias(profile, b=None, prefetch=False):
    obj = {"profile": profile, "prefetch": prefetch}
    if b is not None:
        obj["bias"] = b
    st, r = call("/cnm/bias", obj)
    assert st == 200, (st, r)
    print("bias", profile, b, prefetch, json.dumps(r), flush=True)
    return r


def gate(name, ok, detail):
    res["gates"][name] = {"ok": bool(ok), "detail": detail}
    save("result.json", res)
    print("GATE", name, "PASS" if ok else "FAIL", detail, flush=True)


try:
    t0 = time.time()
    while True:
        if p.poll() is not None:
            raise RuntimeError("server exited rc=%s" % p.returncode)
        try:
            if call("/health", timeout=2)[1].get("status") == "ok":
                break
        except Exception:
            pass
        if time.time() - t0 > 1800:
            raise RuntimeError("load timeout")
        time.sleep(2)
    res["load_s"] = round(time.time() - t0, 1)
    res["vram_after_load"] = sh("nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader")
    res["stats0"] = call("/cnm/stats")[1]
    print("stats0", json.dumps(res["stats0"]), flush=True)

    # G3 reference: all-zero offsets at start
    r_start = run("start-zero", PROMPT)

    # G1/G2 selections share on one profile
    bias("pb3", -1e-4); r0 = run("pb3-1e-4", PROMPT)
    bias("pb3", -0.5); r05 = run("pb3-0.5", PROMPT)
    bias("pb3", -30.0); r30 = run("pb3-30", PROMPT)
    bias("explore", -0.5); rex = run("explore-0.5", PROMPT)
    gate("G1_cold_share_monotone", r0["cold_share"] is not None and r30["cold_share"] < r05["cold_share"] < r0["cold_share"],
         {"~0": r0["cold_share"], "-0.5": r05["cold_share"], "-30": r30["cold_share"], "explore-0.5": rex["cold_share"]})
    gate("G2_switch_changes_output", r30["content_sha256"] != r0["content_sha256"],
         {"~0": r0["content_sha256"][:12], "-30": r30["content_sha256"][:12]})
    bias("none"); rn = run("none", PROMPT)
    gate("G3_none_equals_start", rn["content_sha256"] == r_start["content_sha256"] and rn["cnm"]["selections_cold"] == 0,
         {"start": r_start["content_sha256"][:12], "none": rn["content_sha256"][:12]})

    # G4/G5 prefetch after identical churn
    bias("pb3", -0.5); run("warm-pb3", PROMPT)
    bias("verify", -30.0); run("churn-1", CHURN, 192)
    bias("pb3", -0.5); a = run("after-churn-noPF", PROMPT)
    bias("verify", -30.0); run("churn-2", CHURN, 192)
    pf = bias("pb3", -0.5, prefetch=True); b = run("after-churn-PF", PROMPT)
    res["prefetch_result"] = pf
    gate("G4_prefetch_fewer_misses", b["cnm"]["misses"] < a["cnm"]["misses"],
         {"noPF_misses": a["cnm"]["misses"], "PF_misses": b["cnm"]["misses"], "noPF_hit": a["hit_rate"], "PF_hit": b["hit_rate"],
          "prefetch": pf.get("prefetch")})
    gate("G5_prefetch_same_output", a["content_sha256"] == b["content_sha256"],
         {"noPF": a["content_sha256"][:12], "PF": b["content_sha256"][:12]})

    st, metrics = call("/metrics")
    has = isinstance(metrics, str) and "llamacpp:cnm_moe_cache_hits_total" in metrics
    gate("G6_timings_and_metrics", has and "hits" in b["cnm"],
         {"metrics_lines": [l for l in metrics.splitlines() if l.startswith("llamacpp:cnm")] if isinstance(metrics, str) else metrics})

    # G7 proxy
    ppath = Path(__file__).with_name("phase_proxy.py")
    plog = out / "proxy.jsonl"
    proxy = subprocess.Popen([sys.executable, "-I", str(ppath), "--listen", f"{HOST}:{PORT + 1}", "--upstream", U,
                              "--bias", "-0.5", "--prefetch", "--log", str(plog)], stdout=open(out / "proxy.log", "w"),
                             stderr=subprocess.STDOUT)
    time.sleep(1.5)
    msgs = [{"role": "user", "content": "Rename the variable lv_total to lv_sum in zcl_a.abap."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "Edit", "arguments": json.dumps({"file_path": "zcl_a.abap", "old_string": "lv_total", "new_string": "lv_sum"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
    body = json.dumps({"messages": msgs, "max_tokens": 16, "temperature": 0}).encode()
    rq = urllib.request.Request(f"http://{HOST}:{PORT + 1}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=600) as f:
        pr = json.loads(f.read())
    rec = json.loads(plog.read_text().splitlines()[-1])
    s_after = call("/cnm/stats")[1]
    gate("G7_proxy_switches_profile", rec.get("phase") == "edit" and s_after.get("profile") == "edit" and "choices" in pr,
         {"proxy_record": rec, "server_profile": s_after.get("profile")})
    res["stats_end"] = s_after
    res["status"] = "DONE"
except Exception as e:
    res["status"] = "ERROR"; res["error"] = repr(e)
    print("ERROR", repr(e), flush=True)
finally:
    if proxy and proxy.poll() is None:
        proxy.terminate()
    if p.poll() is None:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(120)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()
    log.close()
    txt = (out / "server.log").read_text(errors="replace")
    res["log_lines"] = [l for l in txt.splitlines() if any(k in l for k in ("MoE cache", "slots (", "llama_moe_cache", "CNM_", "cnm:", "llama_set_expert_bias"))][:80]
    save("result.json", res)
    ok = res["status"] == "DONE" and all(g["ok"] for g in res["gates"].values())
    print("RESULT", "PASS" if ok else "FAIL", res["status"], flush=True)
