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
        cased
    ):
        super(BERT, self).__init__()
        self.cased = 'bert-base-cased' if cased else 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.cased)
        self.bert = BertForSequenceClassification.from_pretrained(self.cased, num_labels=num_labels)

    def forward(self, data):
        inputs = self.tokenizer(data, return_tensors="pt") # here data is raw text
        outputs = self.bert(**inputs)
        logits = outputs.logits
        return logits

    @staticmethod
    def add_required_arguments(parser):
        group = parser.add_argument_group()

        group.add_argument("--bert_cased", type=bool, default=False, help="whether to use cased BERT")
        group.add_argument("--bert_num_labels", type=int, default=2, help="number of output label classes for BERT")


    @staticmethod
    def make_model(args):
        if args.embeddings is not None:
            embeddings = nn.Embedding.from_pretrained(torch.tensor(args.embeddings))
        else:
            embeddings = nn.Embedding(args.vocab_size, args.embedding_dim)

        return BERT(
            args.bert_num_labels,
            args.bert_cased,
        )

    @staticmethod
    def data_requirements():
        return ["tokens", "mask"]
