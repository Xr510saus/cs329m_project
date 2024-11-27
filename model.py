import torch
import torch.nn as nn

from transformers import AutoModel

# Acquired directly from GraphCodeBERT's clonedetection model
# https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/clonedetection/model.py

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class SideEffectClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # pretrained is included in the model (not-frozen),
        # because some weights are not obtained from the model checkpoint
        # and thus re-initialized, so they need to be retrained
        self.pretrained = AutoModel.from_pretrained("microsoft/graphcodebert-base")
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pretrained(x)
        logits = RobertaClassificationHead(x)
        y_pred = self.softmax(logits)
        return y_pred