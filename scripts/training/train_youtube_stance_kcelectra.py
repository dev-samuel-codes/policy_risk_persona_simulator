#!/usr/bin/env python3
"""E03 entry point for E01-style beomi/KcELECTRA-base fine-tuning.

The shared trainer owns the group-safe 80/10/10 split, weighted loss,
evaluation, predictions, and completed-run-only Markdown recording. The E03
launcher supplies the KcELECTRA model and E01 hyperparameters.
"""

from train_youtube_stance_roberta import main


if __name__ == "__main__":
    main()
