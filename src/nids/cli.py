"""Command-line entry point: ``nids <command>``."""

from __future__ import annotations

import argparse

from nids.utils import setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nids", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="fetch and verify NSL-KDD")

    p = sub.add_parser("supervised", help="train the multi-class classifier")
    p.add_argument("--epochs", type=int, default=100)

    p = sub.add_parser("autoencoder", help="train the anomaly detector")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--target-fpr", type=float, default=0.05,
                   help="false-alarm rate the threshold is calibrated to")

    sub.add_parser("baselines", help="fit logistic regression and random forest")
    p = sub.add_parser("robustness", help="retrain both deep models across seeds")
    p.add_argument("--seeds", type=int, default=5)
    sub.add_parser("plots", help="render figures from saved results")
    sub.add_parser("all", help="download, train everything, plot")

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    # Imported lazily so `nids download` doesn't pay TensorFlow's start-up cost.
    if args.command in ("download", "all"):
        from nids import data
        data.download()
    if args.command == "supervised":
        from nids import supervised
        supervised.run(epochs=args.epochs)
    if args.command == "autoencoder":
        from nids import anomaly
        anomaly.run(epochs=args.epochs, target_fpr=args.target_fpr)
    if args.command == "baselines":
        from nids import baselines
        baselines.run()
    if args.command == "robustness":
        from nids import robustness
        robustness.run(n_seeds=args.seeds)
    if args.command == "all":
        from nids import anomaly, baselines, supervised
        supervised.run()
        anomaly.run()
        baselines.run()
    if args.command in ("plots", "all"):
        from nids import plots
        plots.run()


if __name__ == "__main__":
    main()
