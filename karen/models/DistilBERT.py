import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from ..base_model import BaseModel
from ..register_model import RegisterModel


@RegisterModel("DistilBERT")
class DistilBERT(BaseModel):
    """
    BERT sentiment classification
    """

    def __init__(
        self, 
        num_labels, 
        cased, 
        device
    ):
        super(DistilBERT, self).__init__()
        self.cased = "distilbert-base-cased" if cased else "distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.cased)
        self.bert = DistilBertForSequenceClassification.from_pretrained(self.cased, num_labels=num_labels)
        self.device = device

    def forward(self, data):
        inputs = self.tokenizer(data["text"], padding=True, return_tensors="pt").to(self.device)
        outputs = self.bert(**inputs)
        del inputs  # this code uses too much memory so it's better this way
        logits = outputs.logits
        return logits

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument("--distilbert-cased", type=bool, default=False, help="whether to use cased BERT")

    @staticmethod
    def make_model(args):
        return DistilBERT(args.out_feat, args.distilbert_cased, args.device)

    @staticmethod
    def data_requirements():
        return ["text"]
