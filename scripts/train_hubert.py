import sys
sys.path.append('/home/alina/repos/Audio-Classification-HF/')

import os
import logging
import torch

import wandb
import numpy as np

from datasets import DatasetDict, load_dataset, load_metric, concatenate_datasets
from transformers import (
    HubertForSequenceClassification,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    set_seed
)
from src.utils import collator

logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO
)

set_seed(42) #to make experiments reproducible
USER = "USERNAME"
WANDB_PROJECT = "PROJECTNAME"
wandb.init(entity=USER, project=WANDB_PROJECT)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

#loading Dusha dataset
dataset = load_dataset("xbgoose/dusha")

metric = load_metric("accuracy", trust_remote_code=True)

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

ds = DatasetDict(
    {
        "train": train_testvalid["train"],
        "test": test_valid["test"],
        "val": test_valid["train"],
    }
)

ds = ds.class_encode_column("emotion")

print(ds)

labels = ds["train"].features["emotion"].names
print(labels)

label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

model_checkpoint = "facebook/hubert-large-ls960-ft"
batch_size = 4
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)
print(f"Feature Extractor: {feature_extractor}")

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

encoded_dataset = ds.map(prepare_dataset, 
                              fn_kwargs={"feature_extractor": feature_extractor}, 
                              remove_columns=["audio"],
                              batched=True, 
                              batch_size=batch_size)
print(f"Encoded dataset: {encoded_dataset}")


num_labels = len(id2label)
print(f"Number of labels: {num_labels}")
model = HubertForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)

model = model.to(device)

model.freeze_feature_extractor()

training_args = TrainingArguments(
    output_dir="/home/alina/repos/Audio-Classification-HF/results/hubert/06_exp_dusha",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    num_train_epochs=1.0,
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


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

data_collator = collator.DataCollatorCTCWithPadding(
    processor=feature_extractor, padding=True
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["val"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

test_results = trainer.predict(encoded_dataset["test"])
# logging.info("Test Set Result: {}".format(test_results.metrics))
wandb.log({"test_accuracy": test_results.metrics["test_accuracy"]})

model_dir = "models/hubert_06_exp/audio-model"
trainer.save_model(os.path.join(PROJECT_ROOT, model_dir))

# logging trained models to wandb
wandb.save(
    os.path.join(PROJECT_ROOT, model_dir, "*"),
    base_path=os.path.dirname(model_dir),
    policy="end",
)

