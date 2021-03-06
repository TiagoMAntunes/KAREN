import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from ..base_model import BaseModel
from ..register_model import RegisterModel


@RegisterModel("BERT")
class BERT(BaseModel):
    """
    BERT sentiment classification
    """

    def __init__(
        self,
        num_labels,
        cased,
        device,
    ):
        super(BERT, self).__init__()
        self.cased = "bert-base-cased" if cased else "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.cased)
        self.bert = BertForSequenceClassification.from_pretrained(self.cased, num_labels=num_labels)
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

        group.add_argument("--bert-cased", type=bool, default=False, help="whether to use cased BERT")

    @staticmethod
    def make_model(args):
        return BERT(args.out_feat, args.bert_cased, args.device)

    @staticmethod
    def data_requirements():
        return ["text"]
