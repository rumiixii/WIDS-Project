# Statement on use of AI coding assistants

This project was built with the help of an AI coding assistant (Claude Code).
In the spirit of the assignment, here is exactly how it was used and what I
understand about the result.

**How it was used**
- Scaffolding the Python modules (traffic generator, feature extraction,
  training, detection) and the overall project structure.
- Explaining Scapy's 802.11 frame fields (e.g. the `FCfield` flags, RadioTap
  `dBm_AntSignal` for signal strength) and the scikit-learn API.
- Drafting the poster layout and this documentation.

**What I did / checked myself**
- Chose the detection approach: per-device, per-second feature windows rather
  than per-frame, because a flood is a *pattern over time*, not a single frame.
- Decided to include **legitimate** deauth/disassoc frames in the normal class
  so the model cannot cheat by learning "any deauth = attack".
- Used a **group split** (holding out whole capture scenarios) after realising
  a random split leaks windows from the same attack into train and test and
  inflates the score.
- Verified every result by running the code and reading the confusion matrix.

**I can explain, during the demo**
- Why `deauth_per_sec` and `frac_deauth` dominate the feature importance.
- The meaning of each cell in the confusion matrix and why recall < 1.0 (the
  first/last window of a short attack has few malicious frames).
- The difference between WIDS (alert) and WIPS (auto-block) in `detect.py`.
- Why nothing is transmitted over the air and where real-hardware capture
  would plug in.
