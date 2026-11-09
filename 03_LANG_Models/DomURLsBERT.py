import torch
import torch.nn as nn
from transformers import AutoModel


class DomURLsBERT(nn.Module):
    """DomURLs_BERT sequence classification model (pure text-only baseline)."""

    def __init__(
        self,
        output_size: int = 2,
        drop_prob: float = 0.2,
        pretrained_path: str = "amahdaouy/DomURLs_BERT",
        num_classes: int = None,
        **kwargs,
    ):
        super().__init__()
        # ponytail: kwargs absorbs unused architecture params; drop when caller
        # configs stabilize.
        final_output_size = (
            num_classes if num_classes is not None else output_size
        )
        self.pretrained_path = pretrained_path
        self.transformer = AutoModel.from_pretrained(pretrained_path)
        self.out_dim = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(self.out_dim, final_output_size)

    def extract_features(self, input_ids=None, attention_mask=None, **kwargs):
        if isinstance(input_ids, dict):
            attention_mask = input_ids.get("attention_mask")
            input_ids = input_ids.get("input_ids")

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = (
            outputs.pooler_output
            if hasattr(outputs, "pooler_output")
            and outputs.pooler_output is not None
            else outputs[1]
        )
        return self.dropout(pooled)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return self.fc(
            self.extract_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        )


# Alias matching the original repository naming
PLMEncoder = DomURLsBERT


if __name__ == "__main__":
    model = DomURLsBERT(output_size=2)
    dummy_ids = torch.randint(0, 1000, (2, 32))
    dummy_mask = torch.ones((2, 32), dtype=torch.long)
    logits = model(input_ids=dummy_ids, attention_mask=dummy_mask)
    assert logits.shape == (2, 2), f"Expected shape (2, 2), got {logits.shape}"
    print("DomURLsBERT test passed:", logits.shape)
