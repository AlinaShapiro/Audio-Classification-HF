import sys
import os
sys.path.append('/home/alina/repos/Audio-Classification-HF/')

import wandb
import numpy as np
from packaging import version

import torch
from torch import nn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from datasets import DatasetDict, load_dataset, load_metric, concatenate_datasets
from transformers import (
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
    is_apex_available,
    Wav2Vec2Processor
)
from transformers.file_utils import ModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Tuple

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
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
    
class CTCTrainer(Trainer):

    # def __init__(self, use_cuda_amp: bool = False):
        # self.use_cuda_amp = use_cuda_amp

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_cpu_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_cpu_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

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

model_name_or_path = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
pooling_mode = "mean"

config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
    finetuning_task="wav2vec2_clf",
)
setattr(config, 'pooling_mode', pooling_mode)
config.mask_time_prob = 0.0

processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

INPUT_FIELD = "input_values"
LABEL_FIELD = "labels"

def prepare_dataset(examples, processor):
    audio_arrays = [x["array"] for x in examples["audio"]]
    input = processor(
        audio_arrays, 
        sampling_rate=16000, 
        padding=True,
        return_tensors="pt"
    )

    examples[INPUT_FIELD] = input.input_values.to(device)
    examples[LABEL_FIELD] = examples[ "emotion"]  # colname MUST be labels as Trainer will look for it by default

    return examples

encoded_dataset = dataset.map(prepare_dataset, 
                              fn_kwargs={"processor": processor}, 
                              remove_columns=["audio"],
                              batched=True, 
                              batch_size=1)


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

is_regression = False

metric = load_metric("accuracy", trust_remote_code=True)

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)

model.freeze_feature_extractor()

training_args = TrainingArguments(
    output_dir="/home/alina/repos/Audio-Classification-HF/results/wav2vec/04_exp",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    num_train_epochs=2.0,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=1e-3,
    lr_scheduler_type="linear",
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
    tokenizer=processor.feature_extractor,
)

trainer.train()

model_dir = "models/wav2vec_04_exp/audio-model"
test_results = trainer.predict(encoded_dataset["test"])

wandb.log({"test_accuracy": test_results.metrics["test_accuracy"]})

trainer.save_model(os.path.join(PROJECT_ROOT, model_dir))

# logging trained models to wandb
wandb.save(
    os.path.join(PROJECT_ROOT,  model_dir, "*"),
    base_path=os.path.dirname( model_dir),
    policy="end",
)
