"""
Real-time-style replay for the live demonstration.

Reads a capture and feeds it to the detector window-by-window in (optionally)
wall-clock time, so the audience sees normal traffic scrolling by and then the
ALERT firing the moment the deauth flood starts. This is the same detection
code as detect.py, just paced for a live demo.

Run:
  python src/demo_live.py --pcap data/demo/demo_attack.pcap --wips
  python src/demo_live.py --pcap data/demo/demo_attack.pcap --speed 4
"""
import argparse
import time

import joblib

from features import frames_to_dataframe, build_windows
from detect import wips_respond, log_line

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; DIM = "\033[2m"; RST = "\033[0m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--model", default="models/wids.joblib")
    ap.add_argument("--wips", action="store_true")
    ap.add_argument("--speed", type=float, default=2.0,
                    help="playback speed multiplier (2 = twice real time)")
    args = ap.parse_args()

    from scapy.all import rdpcap
    bundle = joblib.load(args.model)
    model = bundle["model"]; feat_cols = bundle["features"]
    window_sec = bundle["window_sec"]; step_sec = bundle["step_sec"]

    pkts = list(rdpcap(args.pcap))
    df = frames_to_dataframe(pkts)
    windows = build_windows(df, window_sec=window_sec, step_sec=step_sec)
    windows = windows.sort_values("window_start")

    print(f"{DIM}Replaying {args.pcap}  ({len(pkts)} frames, {args.speed}x speed){RST}")
    print(f"{DIM}mode = {'WIPS auto-respond' if args.wips else 'WIDS alert-only'}{RST}\n")
    print(f"{'time':>6}  {'device':<19} {'frm/s':>6} {'deauth/s':>9} {'signal':>7}  status")
    print("-" * 72)

    blocked = set()
    last_t = None
    for _, row in windows.iterrows():
        t = row["window_start"]
        if last_t is not None and t > last_t:
            time.sleep((t - last_t) / max(args.speed, 0.1))
        last_t = t

        X = [[row[c] for c in feat_cols]]
        pred = model.predict(X)[0]
        mac = row["src"]
        line = (f"{t:6.1f}  {mac:<19} {row['frames_per_sec']:6.0f} "
                f"{row['deauth_per_sec']:9.0f} {row['rssi_mean']:6.0f}dBm")
        if pred == 1:
            tag = f"{RED}*** ATTACK ***{RST}"
            print(f"{line}  {tag}")
            if args.wips and mac not in blocked:
                wips_respond(mac, f"t={t:.0f}s deauth/s={row['deauth_per_sec']:.0f}")
                print(f"        {YELLOW}-> WIPS blocked {mac}{RST}")
                blocked.add(mac)
        else:
            print(f"{line}  {GREEN}ok{RST}")

    print("\n" + "-" * 72)
    if blocked:
        print(f"{RED}Attack mitigated. Blocked: {sorted(blocked)}{RST}")
    else:
        print(f"{GREEN}No attack detected.{RST}")


if __name__ == "__main__":
    main()
