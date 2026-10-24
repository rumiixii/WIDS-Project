"""
Feature extraction for the Wireless IDS.

The idea: wireless attacks like deauth floods show up as an *unusual pattern
of frames coming from one device over a short time*. So instead of looking at
single frames, we group frames into short sliding time WINDOWS, per source
device (transmitter MAC / addr2), and compute a handful of simple, explainable
features for each (device, window):

  rate features     - frames/sec, deauth/sec, disassoc/sec from this device
  frame-type mix    - fraction of mgmt / ctrl / data frames
  attack-ish mix    - fraction of frames that are deauth or disassoc
  signal strength   - mean / std / min RSSI (dBm) seen for this device
  spread            - number of distinct destinations, broadcast fraction
  size              - mean payload length

These are exactly the kinds of features the brief asks for (frame type, signal
strength, frames-per-second per device). The same function is used to build the
training set from labelled pcaps and to score live/offline traffic, so training
and detection never drift apart.

A window is labelled malicious (1) if it contains at least one frame that the
ground-truth marked malicious; otherwise normal (0).
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scapy.all import RadioTap, Dot11, Dot11Deauth, Dot11Disas, rdpcap

BCAST = "ff:ff:ff:ff:ff:ff"

# Columns the model is trained on, in a fixed order.
FEATURE_COLUMNS = [
    "frames_per_sec",
    "deauth_per_sec",
    "disassoc_per_sec",
    "frac_mgmt",
    "frac_ctrl",
    "frac_data",
    "frac_deauth",
    "frac_disassoc",
    "rssi_mean",
    "rssi_std",
    "rssi_min",
    "n_unique_dst",
    "frac_broadcast",
    "mean_size",
]


def frame_info(pkt):
    """Pull the raw fields we care about out of one 802.11 frame.

    Returns None for frames with no 802.11 layer (e.g. stray non-dot11).
    """
    if not pkt.haslayer(Dot11):
        return None
    d = pkt[Dot11]
    ftype = d.type          # 0 mgmt, 1 ctrl, 2 data
    subtype = d.subtype
    src = d.addr2            # transmitter address
    dst = d.addr1            # receiver address
    is_deauth = pkt.haslayer(Dot11Deauth) or (ftype == 0 and subtype == 12)
    is_disassoc = pkt.haslayer(Dot11Disas) or (ftype == 0 and subtype == 10)

    rssi = None
    if pkt.haslayer(RadioTap):
        rt = pkt[RadioTap]
        rssi = getattr(rt, "dBm_AntSignal", None)

    size = len(pkt)

    return {
        "time": float(pkt.time),
        "type": ftype,
        "subtype": subtype,
        "src": src,
        "dst": dst,
        "is_deauth": bool(is_deauth),
        "is_disassoc": bool(is_disassoc),
        "rssi": rssi,
        "size": size,
    }


def frames_to_dataframe(packets, labels=None):
    """Turn a list of Scapy packets into a per-frame DataFrame.

    `labels`, if given, is a list of 0/1 aligned with `packets`.
    """
    rows = []
    for i, pkt in enumerate(packets):
        fi = frame_info(pkt)
        if fi is None:
            continue
        if fi["src"] is None:
            continue  # control frames with no addr2 (e.g. CTS/ACK) -> skip
        fi["label"] = int(labels[i]) if labels is not None else 0
        rows.append(fi)
    df = pd.DataFrame(rows)
    return df


def _window_features(sub, window_sec):
    """Compute the feature dict for all frames of ONE device in ONE window."""
    n = len(sub)
    dur = window_sec
    types = sub["type"].values
    rssis = sub["rssi"].dropna().values
    if len(rssis) == 0:
        rssis = np.array([-70.0])  # neutral fallback if no radiotap present

    feats = {
        "frames_per_sec": n / dur,
        "deauth_per_sec": sub["is_deauth"].sum() / dur,
        "disassoc_per_sec": sub["is_disassoc"].sum() / dur,
        "frac_mgmt": float(np.mean(types == 0)),
        "frac_ctrl": float(np.mean(types == 1)),
        "frac_data": float(np.mean(types == 2)),
        "frac_deauth": float(sub["is_deauth"].mean()),
        "frac_disassoc": float(sub["is_disassoc"].mean()),
        "rssi_mean": float(np.mean(rssis)),
        "rssi_std": float(np.std(rssis)),
        "rssi_min": float(np.min(rssis)),
        "n_unique_dst": int(sub["dst"].nunique()),
        "frac_broadcast": float(np.mean(sub["dst"].values == BCAST)),
        "mean_size": float(sub["size"].mean()),
    }
    return feats


def build_windows(df, window_sec=1.0, step_sec=1.0):
    """Slide a window across time; for each (device, window) make one feature row.

    Returns a DataFrame with FEATURE_COLUMNS plus: window_start, src, label,
    n_frames. `label` = 1 if any frame in that (device,window) was malicious.
    Non-overlapping windows by default (step == window).
    """
    if df.empty:
        return pd.DataFrame(columns=["window_start", "src", "n_frames", "label"] + FEATURE_COLUMNS)

    t0 = df["time"].min()
    t_end = df["time"].max()
    rows = []
    w = t0
    while w <= t_end + 1e-9:
        win = df[(df["time"] >= w) & (df["time"] < w + window_sec)]
        if not win.empty:
            for src, sub in win.groupby("src"):
                feats = _window_features(sub, window_sec)
                feats["window_start"] = round(w - t0, 3)
                feats["src"] = src
                feats["n_frames"] = len(sub)
                feats["label"] = int(sub["label"].max()) if "label" in sub else 0
                rows.append(feats)
        w += step_sec

    cols = ["window_start", "src", "n_frames", "label"] + FEATURE_COLUMNS
    return pd.DataFrame(rows)[cols]


def pcap_to_windows(pcap_path, label_csv=None, window_sec=1.0, step_sec=1.0):
    """Read a pcap (+ optional label CSV) and return the per-window feature table."""
    packets = rdpcap(pcap_path)
    labels = None
    if label_csv:
        lab = pd.read_csv(label_csv)
        labels = [0] * len(packets)
        for _, r in lab.iterrows():
            idx = int(r["pkt_index"])
            if 0 <= idx < len(labels):
                labels[idx] = int(r["label"])
    df = frames_to_dataframe(packets, labels)
    return build_windows(df, window_sec=window_sec, step_sec=step_sec)
