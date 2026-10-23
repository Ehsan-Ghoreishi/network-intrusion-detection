# Model card

Covers both models in this repository: the supervised classifier
(`models/classifier.keras`) and the autoencoder anomaly detector
(`models/autoencoder.keras`). Both are produced by `nids supervised` and
`nids autoencoder` and are not committed; retrain them to reproduce.

## Intended use

Research and teaching: comparing supervised and anomaly-based intrusion
detection on a standard benchmark, and showing how they fail differently.
**Not intended for production network defence.** The training data is from
1998 and does not represent modern protocols, encrypted traffic or current
attack techniques.

## Training data

NSL-KDD `KDDTrain+` (125,973 connection records, 41 features, 22 attack
types). The autoencoder uses only the 67,343 records labelled `normal`.
Files are SHA-256 verified on download (`src/nids/data.py`).

## Evaluation data

NSL-KDD `KDDTest+` (22,544 records, 37 attack types, 17 of them absent from
training). The test set is used only for final reporting. Model selection
(early stopping, autoencoder threshold) uses validation splits carved from
the training data.

## Metrics

Reported in `reports/*.json`, with multi-seed spread in
`reports/seed_variance.json`:

- detection rate (attack recall) and false-alarm rate (normal-traffic false
  positive rate), the two numbers a SOC tunes against each other;
- ROC-AUC and PR-AUC as threshold-free summaries;
- per-family detection, and detection split by attack types seen vs unseen
  in training.

## Known limitations

- **R2L attacks are hard for both models** (≈45% detection). Password
  guessing and SNMP abuse are indistinguishable from normal logins at the
  connection-record level.
- **Supervised results vary by ±2 points across seeds.** Single-run numbers
  should not be compared without their spread.
- **The autoencoder flags what is unusual, not what is malicious.** On real
  networks, legitimate but rare traffic (backups, new services) will raise
  alerts, so the 5% false-alarm target would need retuning against analyst
  capacity.
- **NSL-KDD artefacts.** The data comes from a simulated 1998 network, and
  its traffic statistics are known to differ from real networks, which makes
  attacks easier to separate than they would be in practice.

## Ethical considerations

The dataset contains no personal data. Intrusion detectors can be used for
surveillance; this project is limited to a public benchmark.
