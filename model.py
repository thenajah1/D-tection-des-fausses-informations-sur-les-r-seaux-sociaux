import torch
import torch.nn as nn
from transformers import BertModel

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 2)  # Modifiez la taille de la couche de sortie en fonction de vos besoins

    def forward(self, input_ids):
        _, cls_hs = self.bert(input_ids)
        x = self.dropout(cls_hs)
        logits = self.fc(x)
        return logits

# Chargez le modèle BERT pré-entraîné
bert = BertModel.from_pretrained('bert-base-uncased')

# Créez une instance du modèle BERT_Arch
model = BERT_Arch(bert)
