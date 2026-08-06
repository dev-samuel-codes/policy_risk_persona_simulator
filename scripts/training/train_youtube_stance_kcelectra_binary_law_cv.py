#!/usr/bin/env python3
"""Binary KcELECTRA CV training with original-law features.

The target is restricted to 긍정->찬성 and 부정->반대. Neutral rows are
excluded before the group-safe 80/10/10 split. Fold count and epochs are set
by each experiment launcher. The shared runner calibrates one binary decision
threshold on dev and evaluates test once. Experiment Markdown is never edited
automatically.
"""

import train_youtube_stance_roberta as core
import train_youtube_stance_roberta_cv as cv


BINARY_SOURCE_TO_STANCE = {"긍정": "찬성", "부정": "반대"}
BINARY_LABELS = ["찬성", "반대"]
BINARY_LABEL2ID = {label: index for index, label in enumerate(BINARY_LABELS)}
BINARY_ID2LABEL = {index: label for label, index in BINARY_LABEL2ID.items()}


def configure_binary_task() -> None:
    for module in (core, cv):
        module.SOURCE_TO_STANCE = BINARY_SOURCE_TO_STANCE
        module.LABELS = BINARY_LABELS
        module.LABEL2ID = BINARY_LABEL2ID
        module.ID2LABEL = BINARY_ID2LABEL


if __name__ == "__main__":
    configure_binary_task()
    cv.main()
