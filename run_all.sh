#!/usr/bin/env bash
# One-command reproduction of the whole project:
#   generate data -> train model -> evaluate -> run WIDS + WIPS demos
set -e
cd "$(dirname "$0")"

echo "== 1/5  installing deps =="
pip install -q -r requirements.txt

echo "== 2/5  generating synthetic lab captures =="
python3 src/generate_lab_traffic.py --out data/train --scenarios 20 --seed 7
python3 src/generate_lab_traffic.py --demo --out data/demo

echo "== 3/5  training Random Forest =="
python3 src/train.py --data data/train --model random_forest --out models/wids.joblib

echo "== 4/5  training Decision Tree (comparison) =="
python3 src/train.py --data data/train --model decision_tree --out models/wids_tree.joblib --results results_tree

echo "== 5/5  running detector =="
echo "--- WIDS on a normal capture (expect: no alerts) ---"
python3 src/detect.py --pcap data/demo/demo_normal.pcap --quiet-normal
echo "--- WIPS on an attack capture (expect: alert + block) ---"
python3 src/detect.py --pcap data/demo/demo_attack.pcap --wips --quiet-normal

echo
echo "Done. See results/confusion_matrix.png, results/feature_importance.png,"
echo "results/metrics.txt, results/alerts.log, results/blocklist.txt"
