#!/usr/bin/env python3
"""Build router-bias profiles and prefetch lists for llama-server --cnm-profiles.

Inputs: phase-signal.json (phase tomography: signal[variant][phase][layer] = 512 shares) and a fixed keep mask
(PB3-192.json: {layer: [kept IDs]}). For every phase the per-layer keep COUNT is the one of the fixed mask (same
budget, only the members change); the members are the brightest experts of that phase. Layers without a phase signal
(e.g. 47) keep the fixed mask's members.

Writes into OUTDIR:
  pb3.json, explore.json, edit.json, verify.json            CNM_EXPERT_BIAS format {"bias": B, "layers": {l: [cold IDs]}}
  <name>.prefetch.json                                       {"layers": {l: [kept IDs, brightest first]}}
  MANIFEST.txt                                               inputs, sha256, per-profile counts (not *.json: the
                                                             server loads every *.json in the dir as a profile)
Usage: make_profiles.py SIGNAL.json KEEP_MASK.json OUTDIR [--bias -0.5] [--variant soft|pure]
"""
import argparse, hashlib, json, os


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signal"); ap.add_argument("mask"); ap.add_argument("out")
    ap.add_argument("--bias", type=float, default=-0.5)
    ap.add_argument("--variant", default="soft", choices=("soft", "pure"))
    ap.add_argument("--n-expert", type=int, default=512)
    a = ap.parse_args()
    if a.bias > 0:
        raise SystemExit("bias must be <= 0")
    sig = json.load(open(a.signal))["signal"][a.variant]
    keep = {int(k): sorted(v) for k, v in json.load(open(a.mask)).items()}
    phases = ("explore", "edit", "verify")
    layers = sorted(keep)
    os.makedirs(a.out, exist_ok=True)

    def has_sig(l):
        return all(str(l) in sig[p] for p in phases)

    # fixed profile: members of the mask, prefetch order = phase-balanced brightness (mean of the three phases)
    profiles = {}
    pb = {}
    for l in layers:
        if has_sig(l):
            bright = [sum(sig[p][str(l)][e] for p in phases) / 3 for e in range(a.n_expert)]
            pb[l] = sorted(keep[l], key=lambda e: (-bright[e], e))
        else:
            pb[l] = list(keep[l])
    profiles["pb3"] = pb
    for p in phases:
        prof = {}
        for l in layers:
            if has_sig(l):
                v = sig[p][str(l)]
                prof[l] = sorted(range(a.n_expert), key=lambda e: (-v[e], e))[:len(keep[l])]
            else:
                prof[l] = list(keep[l])
        profiles[p] = prof

    lines = [f"signal {a.signal} sha256 {sha(a.signal)} variant {a.variant}",
             f"mask {a.mask} sha256 {sha(a.mask)}", f"bias {a.bias}"]
    for name, prof in profiles.items():
        cold = {str(l): sorted(set(range(a.n_expert)) - set(prof[l])) for l in layers}
        bias_path = os.path.join(a.out, f"{name}.json")
        json.dump({"bias": a.bias, "layers": cold}, open(bias_path, "w"), separators=(",", ":"))
        # layers without a phase signal have no brightness order: leave them out of the prefetch list
        pf = {str(l): prof[l] for l in layers if has_sig(l)}
        pf_path = os.path.join(a.out, f"{name}.prefetch.json")
        json.dump({"layers": pf}, open(pf_path, "w"), separators=(",", ":"))
        n_keep = sum(len(prof[l]) for l in layers)
        diff = sum(len(set(prof[l]) - set(keep[l])) for l in layers)
        lines.append(f"{name}: kept {n_keep} over {len(layers)} layers, not in mask {diff}, "
                     f"bias sha256 {sha(bias_path)}, prefetch sha256 {sha(pf_path)}")
    open(os.path.join(a.out, "MANIFEST.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
