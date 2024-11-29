import torch
import torch.nn as nn

from transformers import AutoModel

BASE_MODEL = "microsoft/graphcodebert-base"
NUM_CLASSES = 2

# Acquired directly from GraphCodeBERT's clonedetection model from GitHub
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, NUM_CLASSES)
        self.config = config

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class SideEffectClassificationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # pretrained is included in the model (not-frozen),
        # because some weights are not obtained from the model checkpoint
        # and thus re-initialized, so they need to be retrained
        self.pretrained = AutoModel.from_pretrained(BASE_MODEL)
        self.classifier = RobertaClassificationHead(config)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, attn_masks):
        x = self.pretrained(x, attention_mask=attn_masks)[0]
        logits = self.classifier(x)
        y_pred = self.softmax(logits)
        return y_pred