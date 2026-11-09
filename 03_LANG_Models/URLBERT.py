from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM


class URLBERT(nn.Module):
    """URLBERT classifier backed by the original local pretrained weights."""

    def __init__(
        self,
        output_size: int = 2,
        drop_prob: float = 0.2,
        pretrained_path: str = "urlbert_model",
        num_classes: int = None,
        **kwargs,
    ):
        super().__init__()
        model_dir = Path(pretrained_path).expanduser().resolve()
        weights_path = model_dir / "urlBERT.pt"
        if not weights_path.is_file():
            raise FileNotFoundError(f"URLBERT weights not found: {weights_path}")

        config = AutoConfig.from_pretrained(
            model_dir,
            hidden_dropout_prob=0.2,
            vocab_size=5000,
        )
        self.pretrained_path = str(model_dir)
        self.bert_model = AutoModelForMaskedLM.from_config(config)
        self.bert_model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        self.out_dim = config.hidden_size
        self.dropout = nn.Dropout(drop_prob)
        self.classifier = nn.Linear(
            self.out_dim,
            num_classes if num_classes is not None else output_size,
        )

    def extract_features(self, input_ids=None, attention_mask=None, **kwargs):
        if isinstance(input_ids, dict):
            attention_mask = input_ids.get("attention_mask")
            input_ids = input_ids.get("input_ids")
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return self.dropout(outputs.hidden_states[-1][:, 0, :])

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return self.classifier(
            self.extract_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        )


URLBertForSequenceClassification = URLBERT
