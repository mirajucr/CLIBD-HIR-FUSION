import math

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig


def _build_sequence_pipeline(tokenizer):
    return lambda x: tokenizer(
        x,
        return_tensors="pt",
        padding="longest",
    )["input_ids"]


def load_pre_trained_dnabert2(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    config = BertConfig.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, config=config)
    return model, _build_sequence_pipeline(tokenizer)


def load_pre_trained_modified_dnabert2(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    config = BertConfig.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, config=config)
    return ModifiedBertModel(model, target_dim=512), _build_sequence_pipeline(tokenizer)


def load_pre_trained_Modified_dnabert2(checkpoint):
    return load_pre_trained_modified_dnabert2(checkpoint)


class ModifiedBertModel(nn.Module):
    def __init__(self, base_model, target_dim=512):
        super().__init__()
        self.base_model = base_model
        self.projection = nn.Linear(base_model.config.hidden_size, target_dim)

        nn.init.kaiming_uniform_(self.projection.weight, a=math.sqrt(5))
        if self.projection.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.projection.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.projection.bias, -bound, bound)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_state = outputs[0]
        projected_output = self.projection(last_hidden_state)
        return projected_output, last_hidden_state
