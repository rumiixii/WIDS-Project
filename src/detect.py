"""
WIDS / WIPS detector.

Reads wireless traffic, slides the same feature windows used in training,
asks the trained classifier to label each (device, window), and reacts:

  WIDS mode (default)  -> raise an ALERT and log it
  WIPS mode (--wips)   -> also take an automatic response against the
                          offending source MAC. The default response is SAFE:
                          it records the MAC in a blocklist file and logs the
                          exact firewall/deauth-mitigation command that a real
                          deployment would run. A real enforcement hook is
                          provided but only runs with --enforce (so the demo
                          cannot accidentally touch a live system).

Input sources:
  --pcap FILE    read a capture file (works everywhere, used for the demo)
  --live IFACE   sniff a monitor-mode interface live (needs root + a card in
                 monitor mode; not available in the cloud sandbox). Same code
                 path either way.

Run:
  python src/detect.py --pcap data/demo/demo_attack.pcap --model models/wids.joblib
  python src/detect.py --pcap data/demo/demo_attack.pcap --wips
  sudo python src/detect.py --live wlan0mon --wips --enforce   # real lab only
"""
import argparse
import datetime as dt
import os
import subprocess
import sys

import joblib
import pandas as pd

from features import (FEATURE_COLUMNS, frames_to_dataframe, build_windows,
                      frame_info)

ALERT_LOG = "results/alerts.log"
BLOCKLIST = "results/blocklist.txt"


def log_line(msg, path=ALERT_LOG):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    with open(path, "a") as fh:
        fh.write(line + "\n")


def wips_respond(mac, detail, enforce=False, iface=None):
    """Automatic response for WIPS mode.

    Default (safe): add MAC to a blocklist file and log the command that a
    real deployment would run. With --enforce, actually run a firewall rule
    (only meaningful on a real gateway you control).
    """
    os.makedirs(os.path.dirname(BLOCKLIST), exist_ok=True)
    already = set()
    if os.path.exists(BLOCKLIST):
        already = set(open(BLOCKLIST).read().split())
    if mac not in already:
        with open(BLOCKLIST, "a") as fh:
            fh.write(mac + "\n")

    # The command a real WIPS/gateway would use to drop this station.
    # (Shown and logged; executed only with --enforce.)
    fw_cmd = ["iptables", "-A", "INPUT", "-m", "mac", "--mac-source", mac, "-j", "DROP"]
    log_line(f"WIPS ACTION: block source {mac} -> added to {BLOCKLIST}  ({detail})")
    log_line(f"WIPS would run: {' '.join(fw_cmd)}")

    if enforce:
        try:
            subprocess.run(fw_cmd, check=True)
            log_line(f"WIPS ENFORCED firewall drop for {mac}")
        except Exception as e:  # pragma: no cover - only on a real gateway
            log_line(f"WIPS enforce failed ({e}); MAC still recorded in blocklist")


def read_packets(args):
    from scapy.all import rdpcap, sniff
    if args.pcap:
        pkts = list(rdpcap(args.pcap))
        return pkts
    if args.live:
        print(f"Sniffing {args.live} (monitor mode). Ctrl-C to stop...")
        pkts = sniff(iface=args.live, timeout=args.live_seconds)
        return list(pkts)
    raise SystemExit("Provide --pcap FILE or --live IFACE")


def main():
    ap = argparse.ArgumentParser(description="WIDS/WIPS detector")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcap", help="read a capture file")
    src.add_argument("--live", help="monitor-mode interface to sniff (needs root)")
    ap.add_argument("--live-seconds", type=int, default=30)
    ap.add_argument("--model", default="models/wids.joblib")
    ap.add_argument("--wips", action="store_true", help="enable automatic response")
    ap.add_argument("--enforce", action="store_true",
                    help="actually run firewall command (real lab gateway only)")
    ap.add_argument("--iface", default=None)
    ap.add_argument("--quiet-normal", action="store_true",
                    help="don't print a line for every normal window")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    window_sec = bundle["window_sec"]
    step_sec = bundle["step_sec"]
    feat_cols = bundle["features"]

    mode = "WIPS (auto-respond)" if args.wips else "WIDS (alert-only)"
    print("=" * 64)
    print(f"  Wireless IDS/IPS   mode={mode}   model={bundle.get('model_name')}")
    print(f"  window={window_sec}s step={step_sec}s")
    print("=" * 64)

    pkts = read_packets(args)
    df = frames_to_dataframe(pkts)
    if df.empty:
        print("No 802.11 frames found.")
        return
    windows = build_windows(df, window_sec=window_sec, step_sec=step_sec)
    if windows.empty:
        print("No windows to score.")
        return

    X = windows[feat_cols].values
    windows = windows.copy()
    windows["pred"] = model.predict(X)

    n_alerts = 0
    alerted_macs = set()
    for _, row in windows.iterrows():
        if row["pred"] == 1:
            n_alerts += 1
            mac = row["src"]
            detail = (f"t={row['window_start']:.0f}s frames/s={row['frames_per_sec']:.0f} "
                      f"deauth/s={row['deauth_per_sec']:.0f} "
                      f"frac_deauth={row['frac_deauth']:.2f} "
                      f"rssi={row['rssi_mean']:.0f}dBm")
            log_line(f"ALERT  attack detected from {mac}  [{detail}]")
            if args.wips:
                wips_respond(mac, detail, enforce=args.enforce, iface=args.iface)
            alerted_macs.add(mac)
        elif not args.quiet_normal:
            print(f"  ok   t={row['window_start']:6.1f}s  {row['src']}  "
                  f"frames/s={row['frames_per_sec']:5.0f}  deauth/s={row['deauth_per_sec']:4.0f}")

    print("\n" + "-" * 64)
    print(f"Scored {len(windows)} device-windows.")
    if n_alerts:
        print(f"*** {n_alerts} ALERT window(s). Offending MAC(s): {sorted(alerted_macs)}")
        if args.wips:
            print(f"*** WIPS blocklist: {BLOCKLIST}")
    else:
        print("No attacks detected. Traffic looks normal.")
    print(f"Alert log: {ALERT_LOG}")


if __name__ == "__main__":
    main()
