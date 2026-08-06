#!/usr/bin/env python3
"""E04 entry point for E02-style KcELECTRA 5-fold fine-tuning.

The shared CV trainer uses the train 80% for five folds, calibrates class
thresholds only on dev 10%, and evaluates the frozen ensemble on test 10%.
It writes run artifacts and run_summary.json, but never edits the experiment
Markdown ledger automatically.
"""

from train_youtube_stance_roberta_cv import main


if __name__ == "__main__":
    main()
