# Wi-Fi Intrusion Detection / Prevention System (WIDS / WIPS)

A small, explainable machine-learning system that reads 802.11 (Wi-Fi)
traffic, extracts a few simple features per device, and uses a scikit-learn
classifier (Random Forest / Decision Tree) to label traffic as **normal** or
**malicious**. It detects **deauthentication / disassociation flood attacks**
and raises an alert. In **WIPS** mode it also takes an automatic response:
blocklisting the offending MAC address.

> **Scope / ethics.** All traffic in this repo is *synthetic* — frames are
> built in memory with Scapy and written to `.pcap` files. **Nothing is ever
> transmitted over the air.** The detector is read-only. The optional live
> capture and any real attack generation must only ever be done on your own
> isolated lab network that you are authorised to test, as the brief requires.

---

## What it does

```
 pcap / live monitor-mode capture
            │
            ▼
   feature extraction  ── group frames into 1-second windows, per source MAC
            │              features: frames/sec, deauth/sec, disassoc/sec,
            │              frame-type mix, deauth fraction, signal strength
            │              (mean/std/min dBm), #destinations, broadcast frac, size
            ▼
   scikit-learn classifier  (Random Forest)
            │
            ├── normal   → log "ok"
            └── malicious → ALERT  (WIDS)
                              └── + block offending MAC  (WIPS)
```

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh            # generate data, train, evaluate, run demos
```

Or step by step:

```bash
# 1. make labelled synthetic captures (pcap + per-frame label CSV)
python3 src/generate_lab_traffic.py --out data/train --scenarios 20 --seed 7
python3 src/generate_lab_traffic.py --demo --out data/demo

# 2. train + evaluate (saves model, confusion matrix, feature importance)
python3 src/train.py --data data/train --model random_forest --out models/wids.joblib

# 3a. WIDS: detect on a capture, alert only
python3 src/detect.py --pcap data/demo/demo_attack.pcap

# 3b. WIPS: detect + auto-block the attacker's MAC
python3 src/detect.py --pcap data/demo/demo_attack.pcap --wips

# 4. live-style replay for the demonstration
python3 src/demo_live.py --pcap data/demo/demo_attack.pcap --wips --speed 2
```

## Files

| path | purpose |
|------|---------|
| `src/generate_lab_traffic.py` | builds synthetic 802.11 pcaps + ground-truth labels |
| `src/features.py` | shared feature extraction (used by training **and** detection) |
| `src/train.py` | trains the classifier, prints accuracy + confusion matrix, saves plots |
| `src/detect.py` | WIDS/WIPS detector (reads pcap or live interface) |
| `src/demo_live.py` | paced real-time replay for the live demo |
| `results/` | confusion matrix, feature importance, metrics, alert log, blocklist |
| `poster/poster.html` | A1 poster (open in a browser, "Print → Save as PDF", paper size A1) |

## Results (held-out attack scenarios)

Evaluated by holding out **whole capture scenarios** from training (a group
split), so the model is tested on attacks it never saw during training.

| model | accuracy | malicious precision | malicious recall |
|-------|:--------:|:-------------------:|:----------------:|
| Random Forest | **0.995** | 1.000 | 0.875 |
| Decision Tree | 0.991 | 1.000 | 0.750 |

Confusion matrix and feature-importance charts are in `results/`.
The strongest features are `deauth_per_sec`, `frac_deauth` and `frames_per_sec`
— which matches the intuition that a flood is an abnormal burst of management
frames from one device.

## Running on real hardware (authorised lab only)

The same `detect.py` works on a live capture. On a Linux box with a Wi-Fi
adapter that supports monitor mode:

```bash
sudo airmon-ng start wlan0           # creates wlan0mon in monitor mode
sudo python3 src/detect.py --live wlan0mon --wips
```

To *enforce* a real block on a gateway you control, add `--enforce` (it runs an
`iptables` MAC-drop rule). Without `--enforce` the WIPS response is safe: it
records the MAC in `results/blocklist.txt` and logs the command it *would* run.
Only ever point this at your own / an approved testbed.

## How AI assistants were used

See `AI_USE.md` (also summarised on the poster).
