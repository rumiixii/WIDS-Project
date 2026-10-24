"""
Synthetic 802.11 lab-traffic generator for a Wireless IDS project.

This script does NOT touch a radio. It uses Scapy to construct 802.11
management/control/data frames in memory and saves them to a .pcap file,
plus a sidecar CSV giving a ground-truth label per frame (0 normal,
1 malicious). The pcap is later read back by the feature extractor and used
to train/evaluate the classifier. Nothing is ever transmitted over the air.

The point is to produce a labelled dataset that resembles what you would see
if you captured an authorised test lab in monitor mode, so the ML pipeline
can be built and demonstrated reproducibly without needing live hardware.

Run:
  python src/generate_lab_traffic.py --out data/train --scenarios 8 --seed 1
  python src/generate_lab_traffic.py --demo --out data/demo
"""
import argparse
import csv
import os
import random

from scapy.all import (RadioTap, Dot11, Dot11Beacon, Dot11Elt, Dot11ProbeReq,
                       Dot11ProbeResp, Dot11Auth, Dot11AssoReq, Dot11AssoResp,
                       Dot11Deauth, Dot11Disas, LLC, SNAP, Raw, PcapWriter)

BCAST = "ff:ff:ff:ff:ff:ff"
SSID = "LAB-TESTBED"


def rand_mac(rng, local=False):
    first = rng.randrange(0, 256) & 0xFC
    if local:
        first |= 0x02
    rest = [rng.randrange(256) for _ in range(5)]
    return ":".join(f"{b:02x}" for b in [first] + rest)


class FrameLog:
    """Collects (time, built_frame, label) tuples, sorts and writes them out."""

    def __init__(self):
        self.items = []  # list of [time, scapy_pkt, label]

    def add(self, t, pkt, label):
        self.items.append([float(t), pkt, int(label)])

    def write(self, pcap_path, csv_path):
        self.items.sort(key=lambda x: x[0])
        os.makedirs(os.path.dirname(pcap_path), exist_ok=True)
        pw = PcapWriter(pcap_path, sync=True)
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pkt_index", "time", "label"])
            for i, (t, pkt, label) in enumerate(self.items):
                pkt.time = t
                pw.write(pkt)
                w.writerow([i, f"{t:.6f}", label])
        pw.close()
        return len(self.items)


def rt(rssi):
    """RadioTap header carrying a signal-strength reading (dBm)."""
    return RadioTap(present="dBm_AntSignal", dBm_AntSignal=int(rssi))


def jitter(rng, base, spread=3):
    return max(-95, min(-20, int(base + rng.uniform(-spread, spread))))


def beacon(rng, bssid, rssi):
    p = rt(jitter(rng, rssi)) / Dot11(type=0, subtype=8, addr1=BCAST, addr2=bssid, addr3=bssid)
    p /= Dot11Beacon(cap="ESS+privacy") / Dot11Elt(ID="SSID", info=SSID)
    return p


def data_frame(rng, src, dst, bssid, rssi, to_ds=1):
    # type=2 data; small payload to look like real traffic
    p = rt(jitter(rng, rssi)) / Dot11(type=2, subtype=0, addr1=bssid if to_ds else dst,
                                      addr2=src, addr3=dst if to_ds else bssid,
                                      FCfield="to_DS" if to_ds else "from_DS")
    p /= LLC() / SNAP() / Raw(load=bytes(rng.randrange(256) for _ in range(rng.randint(40, 300))))
    return p


def null_frame(rng, src, bssid, rssi):
    # power-save null data (type=2 subtype=4)
    return rt(jitter(rng, rssi)) / Dot11(type=2, subtype=4, addr1=bssid, addr2=src, addr3=bssid)


def rts_frame(rng, src, dst, rssi):
    # control RTS (type=1 subtype=11)
    return rt(jitter(rng, rssi)) / Dot11(type=1, subtype=11, addr1=dst, addr2=src)


def probe_req(rng, src, rssi):
    return rt(jitter(rng, rssi)) / Dot11(type=0, subtype=4, addr1=BCAST, addr2=src, addr3=BCAST) / \
        Dot11ProbeReq() / Dot11Elt(ID="SSID", info=SSID)


def probe_resp(rng, bssid, dst, rssi):
    return rt(jitter(rng, rssi)) / Dot11(type=0, subtype=5, addr1=dst, addr2=bssid, addr3=bssid) / \
        Dot11ProbeResp() / Dot11Elt(ID="SSID", info=SSID)


def auth(rng, src, dst, rssi):
    return rt(jitter(rng, rssi)) / Dot11(type=0, subtype=11, addr1=dst, addr2=src, addr3=dst) / Dot11Auth(seqnum=1)


def assoc_req(rng, src, bssid, rssi):
    return rt(jitter(rng, rssi)) / Dot11(type=0, subtype=0, addr1=bssid, addr2=src, addr3=bssid) / \
        Dot11AssoReq() / Dot11Elt(ID="SSID", info=SSID)


def assoc_resp(rng, bssid, dst, rssi):
    return rt(jitter(rng, rssi)) / Dot11(type=0, subtype=1, addr1=dst, addr2=bssid, addr3=bssid) / Dot11AssoResp()


def deauth(rng, src, dst, bssid, rssi, reason=7):
    return rt(jitter(rng, rssi)) / Dot11(type=0, subtype=12, addr1=dst, addr2=src, addr3=bssid) / \
        Dot11Deauth(reason=reason)


def disas(rng, src, dst, bssid, rssi, reason=8):
    return rt(jitter(rng, rssi)) / Dot11(type=0, subtype=10, addr1=dst, addr2=src, addr3=bssid) / \
        Dot11Disas(reason=reason)


def build_scenario(seed, duration, attack):
    """Build one capture window. `attack` selects the malicious behaviour (or None)."""
    rng = random.Random(seed)
    log = FrameLog()

    bssid = rand_mac(rng)
    ap_rssi = rng.randint(-55, -40)   # AP is strong/close
    n_clients = rng.randint(2, 5)
    clients = []
    for _ in range(n_clients):
        clients.append({
            "mac": rand_mac(rng),
            "rssi": rng.randint(-75, -50),
            "profile": rng.choice(["idle", "web", "video"]),
        })

    # --- normal background traffic over the whole window ---
    t = 0.0
    # steady beacons ~10/s
    bt = 0.0
    while bt < duration:
        log.add(bt + rng.uniform(0, 0.02), beacon(rng, bssid, ap_rssi), 0)
        bt += 0.1

    # association handshakes near the start for each client
    for c in clients:
        start = rng.uniform(0.2, 2.0)
        log.add(start, probe_req(rng, c["mac"], c["rssi"]), 0)
        log.add(start + 0.01, probe_resp(rng, bssid, c["mac"], ap_rssi), 0)
        log.add(start + 0.03, auth(rng, c["mac"], bssid, c["rssi"]), 0)
        log.add(start + 0.04, auth(rng, bssid, c["mac"], ap_rssi), 0)
        log.add(start + 0.06, assoc_req(rng, c["mac"], bssid, c["rssi"]), 0)
        log.add(start + 0.07, assoc_resp(rng, bssid, c["mac"], ap_rssi), 0)

    # per-client data traffic
    rates = {"idle": 1.0, "web": 12.0, "video": 60.0}
    for c in clients:
        rate = rates[c["profile"]]
        ct = rng.uniform(2.0, 3.0)
        while ct < duration:
            # uplink + occasional downlink
            log.add(ct, data_frame(rng, c["mac"], bssid, bssid, c["rssi"], to_ds=1), 0)
            if rng.random() < 0.6:
                log.add(ct + 0.001, data_frame(rng, bssid, c["mac"], bssid, ap_rssi, to_ds=0), 0)
            if rng.random() < 0.15:
                log.add(ct + 0.002, rts_frame(rng, c["mac"], bssid, c["rssi"]), 0)
            if rng.random() < 0.1:
                log.add(ct + 0.003, null_frame(rng, c["mac"], bssid, c["rssi"]), 0)
            # exponential-ish inter-arrival
            ct += rng.expovariate(rate) if rate > 0 else 1.0

    # a few LEGITIMATE deauth/disassoc events (these stay labelled 0):
    #   client leaving the network, AP housekeeping. This stops the model
    #   from simply learning "deauth frame == attack".
    n_legit = rng.randint(1, 4)
    for _ in range(n_legit):
        c = rng.choice(clients)
        lt = rng.uniform(3.0, max(3.1, duration - 1))
        if rng.random() < 0.5:
            log.add(lt, deauth(rng, c["mac"], bssid, bssid, c["rssi"], reason=3), 0)  # client leaving
        else:
            log.add(lt, disas(rng, bssid, c["mac"], bssid, ap_rssi, reason=8), 0)     # AP disassoc

    # --- malicious burst, if any ---
    if attack:
        a_start = rng.uniform(duration * 0.35, duration * 0.6)
        a_dur = rng.uniform(2.5, min(6.0, duration - a_start - 0.5))
        a_dur = max(1.5, a_dur)
        victim = rng.choice(clients)
        atk_rssi = rng.randint(-80, -60)  # attacker often weaker/variable signal

        if attack == "broadcast_flood":
            rate = rng.uniform(120, 300)      # frames/sec, very high
            tt = a_start
            while tt < a_start + a_dur:
                log.add(tt, deauth(rng, bssid, BCAST, bssid, atk_rssi, reason=7), 1)
                tt += 1.0 / rate
        elif attack == "targeted_flood":
            rate = rng.uniform(80, 200)
            tt = a_start
            while tt < a_start + a_dur:
                # spoof both directions AP<->client
                log.add(tt, deauth(rng, bssid, victim["mac"], bssid, atk_rssi, reason=7), 1)
                log.add(tt + 0.0005, deauth(rng, victim["mac"], bssid, bssid, atk_rssi, reason=7), 1)
                tt += 1.0 / rate
        elif attack == "stealth_flood":
            rate = rng.uniform(25, 50)        # lower rate, harder to catch
            tt = a_start
            while tt < a_start + a_dur:
                log.add(tt, deauth(rng, bssid, victim["mac"], bssid, atk_rssi, reason=7), 1)
                tt += 1.0 / rate
        elif attack == "disassoc_flood":
            rate = rng.uniform(100, 220)
            tt = a_start
            while tt < a_start + a_dur:
                log.add(tt, disas(rng, bssid, BCAST, bssid, atk_rssi, reason=7), 1)
                tt += 1.0 / rate
        elif attack == "rogue_mac_flood":
            rogue = rand_mac(rng, local=True)
            rate = rng.uniform(100, 250)
            tt = a_start
            while tt < a_start + a_dur:
                log.add(tt, deauth(rng, rogue, BCAST, bssid, atk_rssi, reason=7), 1)
                tt += 1.0 / rate

    return log


ATTACKS = [None, "broadcast_flood", "targeted_flood", "stealth_flood",
           "disassoc_flood", "rogue_mac_flood"]


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic 802.11 lab pcaps for WIDS training.")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--scenarios", type=int, default=8)
    ap.add_argument("--duration", type=float, default=20.0, help="seconds per scenario")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--demo", action="store_true",
                    help="produce one clean capture then one with a broadcast flood (for the live demo)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.demo:
        plan = [("demo_normal", None), ("demo_attack", "broadcast_flood")]
        for i, (name, atk) in enumerate(plan):
            log = build_scenario(args.seed + i, args.duration, atk)
            n = log.write(os.path.join(args.out, f"{name}.pcap"),
                          os.path.join(args.out, f"{name}.labels.csv"))
            print(f"[demo] {name}: {n} frames  attack={atk}")
        return

    rng = random.Random(args.seed)
    for i in range(args.scenarios):
        # ~40% clean, rest spread across attack types
        if rng.random() < 0.4:
            atk = None
        else:
            atk = rng.choice(ATTACKS[1:])
        log = build_scenario(args.seed * 1000 + i, args.duration, atk)
        name = f"scenario_{i:02d}_{atk or 'normal'}"
        n = log.write(os.path.join(args.out, f"{name}.pcap"),
                      os.path.join(args.out, f"{name}.labels.csv"))
        print(f"[{i:02d}] {name}: {n} frames  attack={atk}")


if __name__ == "__main__":
    main()
