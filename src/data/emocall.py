# Lint as: python3
"""EmoCall dataset."""

import os
from typing import Union
import datasets
import pandas as pd

_DESCRIPTION = """\
    EmoCall is a data set of 329 telephone recordings from 10 actors.
    Actors spoke from a selection of 10 sentences for each emotion.
    The sentences were presented using one of six different emotions
    (Anger, Positive, Neutral, Sad and Other). 
"""

# _HOMEPAGE = "https://github.com/CheyneyComputerScience/CREMA-D"

DATA_DIR = {"train": "EmoCall"}
LABEL_MAP = {"ANG": "angry",
                 "NEU": "neutral",
                 "OTH": "other",
                 "HAP": "positive",
                 "SAD": "sad"}


class EmoCall(datasets.GeneratorBasedBuilder):
    """EmoCall dataset."""

    DEFAULT_WRITER_BATCH_SIZE = 32
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="clean", description="Train Set.")]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {"file": datasets.Value("string"), "label": datasets.Value("string")}
            ),
            supervised_keys=("file", "label"),
        )

    def _split_generators(
        self, dl_manager: datasets.utils.download_manager.DownloadManager
    ):
        data_dir = dl_manager.extract(self.config.data_dir)
        if self.config.name == "clean":
            train_splits = [
                datasets.SplitGenerator(
                    name="train", gen_kwargs={"files": data_dir, "name": "train"}
                )
            ]

        return train_splits

    def _generate_examples(self, files: Union[str, os.PathLike], name: str):
        """Generate examples from a EmoCall unzipped directory."""
        key = 0
        examples = list()

        audio_dir = os.path.join(files, DATA_DIR[name])

        if not os.path.exists(audio_dir):
            raise FileNotFoundError
        else:
            for file in os.listdir(audio_dir):
                res = dict()
                res["file"] = "{}".format(os.path.join(audio_dir, file))
                res["label"] = LABEL_MAP[file.split("_")[-1].split(".")[0]]
                examples.append(res)

        for example in examples:
            yield key, {**example}
            key += 1
        examples = []
