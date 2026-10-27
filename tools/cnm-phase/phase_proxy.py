#!/usr/bin/env python3
"""Phase proxy between a bench agent and llama-server (stdlib only).

For every generation request (Anthropic /v1/messages, OpenAI /v1/chat/completions, /completion) it finds the LAST
tool call in the request messages, classifies it as explore / edit / verify, and before forwarding calls
POST /cnm/bias {"profile": <profile of the phase>, "bias": B, "prefetch": P} on the server (llama-server started with
--cnm-profiles DIR and CNM_EXPERT_BIAS_RUNTIME=1). The server skips the switch when the profile is unchanged; with
--prefetch the profile's prefetch list is uploaded before every request (cheap when the experts are already cached).
Requests without a tool call yet (first turn) use --default-phase.

Modes:
  --mode phase   switch per phase (default)
  --mode fixed   always the --fixed-profile (control arm; still logs the detected phase)
  --mode off     forward only (logs the phase and the counters)

Every generation request is logged as one JSON line (--log): time, path, detected phase, last tool, applied
profile, switch/prefetch result, wall seconds, and the MoE cache counter delta from GET /cnm/stats around the request
(requests are serialized by default, so the delta belongs to the request; with -np 1 the server serializes anyway).

Classification of a tool call (name + Bash command):
  verify : osg-abap, abaplint, a word "test", go/npm test, pytest, make, cargo test|build|check (as phase_tomo.py)
  edit   : Write/Edit/MultiEdit/NotebookEdit/apply_patch, or sed -i, perl -i, redirect into *.abap, open(..,'w'),
           tee, patch, git apply
  explore: everything else (Read, Grep, Glob, LS, cat, ls, rg, find, git log/diff/show, WebFetch, ...)

Usage:
  phase_proxy.py --listen 127.0.0.1:8090 --upstream http://127.0.0.1:8080 --bias -0.5 --prefetch --log phase.jsonl
"""
import argparse, http.client, http.server, json, re, socketserver, sys, threading, time, urllib.parse

GEN_PATHS = ("/v1/messages", "/v1/chat/completions", "/chat/completions", "/completion", "/completions",
             "/v1/completions")

# same regexes as the phase tomography (phase_tomo.py turn_phase), plus tee / patch / git apply / cargo / pytest / make
VERIFY_RE = re.compile(r"osg-abap|abaplint|\btest\b|go test|npm (run )?test|\bpytest\b|\bmake\b|cargo (test|build|check)")
EDIT_RE = re.compile(r"sed -i|perl -[a-z]*i|>\s*[\w./-]+\.abap|open\([^)]*['\"]w|\btee\b|\bgit apply\b|\bpatch\b")
EDIT_TOOLS = {"write", "edit", "multiedit", "notebookedit", "apply_patch", "str_replace_editor", "create_file",
              "str_replace", "insert", "patch"}
VERIFY_TOOLS = {"build", "test", "run_tests", "check"}


def classify(name, args):
    """phase of one tool call"""
    n = (name or "").lower()
    if n in EDIT_TOOLS:
        return "edit"
    if n in VERIFY_TOOLS or n.startswith("osg") or "abap_check" in n or "syntax_check" in n:
        return "verify"
    cmd = ""
    if isinstance(args, dict):
        cmd = str(args.get("command") or args.get("cmd") or "")
    elif isinstance(args, str):
        cmd = args
    if cmd:
        if VERIFY_RE.search(cmd):
            return "verify"
        if EDIT_RE.search(cmd):
            return "edit"
    return "explore"


def last_tool_call(body):
    """(name, args) of the last tool call in the request messages, None if there is none"""
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return None
    for m in reversed(msgs):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        # OpenAI: tool_calls [{function: {name, arguments}}]
        calls = m.get("tool_calls") or []
        if calls:
            fn = calls[-1].get("function", {}) if isinstance(calls[-1], dict) else {}
            args = fn.get("arguments", "")
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                pass
            return fn.get("name", ""), args
        # Anthropic: content blocks of type tool_use
        content = m.get("content")
        if isinstance(content, list):
            uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
            if uses:
                return uses[-1].get("name", ""), uses[-1].get("input", {})
    return None


class Upstream:
    def __init__(self, url, timeout):
        u = urllib.parse.urlsplit(url)
        self.host, self.port = u.hostname, u.port or 80
        self.timeout = timeout

    def conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def call(self, method, path, obj=None):
        c = self.conn()
        try:
            data = json.dumps(obj).encode() if obj is not None else None
            c.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
            r = c.getresponse()
            raw = r.read()
            try:
                out = json.loads(raw)
            except ValueError:
                out = {"raw": raw.decode(errors="replace")[:500]}
            return r.status, out
        finally:
            c.close()


def delta(a, b):
    if not a or not b:
        return None
    return {k: b[k] - a[k] for k in b if isinstance(b.get(k), (int, float)) and isinstance(a.get(k), (int, float))}


class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    cfg = None
    up = None
    lock = threading.Lock()
    log_lock = threading.Lock()

    def log_message(self, fmt, *args):
        if self.cfg.verbose:
            sys.stderr.write("proxy: " + fmt % args + "\n")

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _forward(self, body):
        """forward the request, stream the response back unchanged"""
        c = self.up.conn()
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
        hdrs["Content-Length"] = str(len(body))
        c.request(self.command, self.path, body=body, headers=hdrs)
        r = c.getresponse()
        self.send_response(r.status)
        chunked = False
        for k, v in r.getheaders():
            kl = k.lower()
            if kl in ("connection", "keep-alive"):
                continue
            if kl == "transfer-encoding":
                chunked = True
            self.send_header(k, v)
        if r.getheader("Content-Length") is None and not chunked:
            self.send_header("Transfer-Encoding", "chunked")
            chunked = True
        self.end_headers()
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            if chunked:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            else:
                self.wfile.write(chunk)
            self.wfile.flush()
        if chunked:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        c.close()
        return r.status

    def _handle(self):
        body = self._read_body()
        path = self.path.split("?")[0]
        if self.command != "POST" or path not in GEN_PATHS:
            self._forward(body)
            return
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            req = {}
        call = last_tool_call(req) if isinstance(req, dict) else None
        phase = classify(*call) if call else self.cfg.default_phase
        rec = {"t": time.time(), "path": path, "phase": phase,
               "tool": None if call is None else call[0],
               "cmd": None if call is None or not isinstance(call[1], dict) else str(call[1].get("command", ""))[:120]}
        lock = self.lock if self.cfg.serialize else None
        if lock:
            lock.acquire()
        try:
            if self.cfg.mode != "off":
                profile = self.cfg.fixed_profile if self.cfg.mode == "fixed" else self.cfg.map.get(phase, phase)
                obj = {"profile": profile, "prefetch": bool(self.cfg.prefetch)}
                if self.cfg.bias is not None:
                    obj["bias"] = self.cfg.bias
                st, res = self.up.call("POST", "/cnm/bias", obj)
                rec["profile"] = profile
                rec["switch"] = res if st == 200 else {"status": st, "error": res}
            s0 = self.up.call("GET", "/cnm/stats")[1].get("moe_cache") if self.cfg.stats else None
            t0 = time.time()
            status = self._forward(body)
            rec["wall_s"] = round(time.time() - t0, 3)
            rec["status"] = status
            if self.cfg.stats:
                s1 = self.up.call("GET", "/cnm/stats")[1].get("moe_cache")
                rec["cache"] = delta(s0, s1)
        finally:
            if lock:
                lock.release()
        if self.cfg.log:
            with self.log_lock, open(self.cfg.log, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def do_POST(self):
        self._handle()

    def do_GET(self):
        self._handle()

    def do_DELETE(self):
        self._handle()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="127.0.0.1:8090")
    ap.add_argument("--upstream", default="http://127.0.0.1:8080")
    ap.add_argument("--mode", choices=("phase", "fixed", "off"), default="phase")
    ap.add_argument("--fixed-profile", default="pb3")
    ap.add_argument("--map", default="explore=explore,edit=edit,verify=verify",
                    help="phase=profile pairs")
    ap.add_argument("--default-phase", default="explore")
    ap.add_argument("--bias", type=float, default=None, help="override the profiles' scalar bias (<= 0)")
    ap.add_argument("--prefetch", action="store_true", help="prefetch the profile's experts before each request")
    ap.add_argument("--no-stats", dest="stats", action="store_false", help="do not read /cnm/stats around requests")
    ap.add_argument("--no-serialize", dest="serialize", action="store_false")
    ap.add_argument("--log", default=None)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--classify", nargs="+", metavar="CMD", help="print the phase of Bash commands and exit")
    cfg = ap.parse_args()
    if cfg.classify:
        for c in cfg.classify:
            print(classify("Bash", {"command": c}), c)
        return
    cfg.map = dict(kv.split("=", 1) for kv in cfg.map.split(",") if kv)
    host, port = cfg.listen.rsplit(":", 1)
    Proxy.cfg = cfg
    Proxy.up = Upstream(cfg.upstream, cfg.timeout)
    srv = Server((host, int(port)), Proxy)
    sys.stderr.write(f"phase_proxy: {cfg.listen} -> {cfg.upstream} mode={cfg.mode} prefetch={cfg.prefetch} bias={cfg.bias}\n")
    srv.serve_forever()


if __name__ == "__main__":
    main()
