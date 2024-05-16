import sys
import os
sys.path.append('/home/alina/repos/Audio-Classification-HF/')

import wandb
import numpy as np
from packaging import version

import torch

from datasets import DatasetDict, load_dataset, load_metric, concatenate_datasets
from transformers import (
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
    Wav2Vec2Processor,
    WavLMForSequenceClassification,
    AutoFeatureExtractor,
)

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Tuple

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch

set_seed(42) #to make experiments reproducible

USER = "USERNAME"
WANDB_PROJECT = "PROJECTNAME"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
wandb.init(entity=USER, project=WANDB_PROJECT)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

dataset = load_dataset("xbgoose/dusha")

ds_ang = dataset['train'].filter(lambda example: example["emotion"] == "angry")
ds_hap = dataset['train'].filter(lambda example: example["emotion"] == "positive")
ds_sad = dataset['train'].filter(lambda example: example["emotion"] == "sad")
ds_neu = dataset['train'].filter(lambda example: example["emotion"] == "neutral")
ds_oth = dataset['train'].filter(lambda example: example["emotion"] == "other")

ds_a = ds_ang.shard(num_shards=4, index=0)
ds_h = ds_hap.shard(num_shards=4, index=0)
ds_s = ds_sad.shard(num_shards=6, index=0)
ds_n = ds_neu.shard(num_shards=30, index=0)
ds_o = ds_oth

ds = concatenate_datasets([ds_a, ds_h, ds_s, ds_n, ds_o])

train_testvalid = ds.train_test_split(shuffle=True, test_size=0.1, seed=42)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict(
    {
        "train": train_testvalid["train"],
        "test": test_valid["test"],
        "val": test_valid["train"],
    }
)

print(dataset)

dataset = dataset.class_encode_column("emotion")

labels = dataset["train"].features["emotion"].names

label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

print(id2label)

model_name_or_path = "microsoft/wavlm-large"

feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

INPUT_FIELD = "input_values"
LABEL_FIELD = "labels"

def prepare_dataset(examples, feature_extractor):
    audio_arrays = [x["array"] for x in examples["audio"]]
    input = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        padding=True,
        return_tensors="pt"
    )

    examples[INPUT_FIELD] = input.input_values.to(device)
    examples[LABEL_FIELD] = examples[ "emotion"]  # colname MUST be labels as Trainer will look for it by default

    return examples

encoded_dataset = dataset.map(prepare_dataset, 
                              fn_kwargs={"feature_extractor": feature_extractor}, 
                              remove_columns=["audio"],
                              batched=True, 
                              batch_size=1)

config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="audio-classification",
        trust_remote_code=True,
    )
config.mask_time_prob = 0.0

model = WavLMForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    ).to(device)


model.freeze_feature_extractor()

metric = load_metric("accuracy", trust_remote_code=True)

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

data_collator = DataCollatorCTCWithPadding(processor=feature_extractor, padding=True)

training_args = TrainingArguments(
    output_dir="/home/alina/repos/Audio-Classification-HF/results/wavlm/05_exp",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    num_train_epochs=3.0,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['val'],
    tokenizer=feature_extractor,
)

trainer.train()

model_dir = "models/wavlm_05_exp/audio-model"
test_results = trainer.predict(encoded_dataset["test"])

wandb.log({"test_accuracy": test_results.metrics["test_accuracy"]})

trainer.save_model(os.path.join(PROJECT_ROOT, model_dir))

# logging trained models to wandb
wandb.save(
    os.path.join(PROJECT_ROOT,  model_dir, "*"),
    base_path=os.path.dirname( model_dir),
    policy="end",
)
