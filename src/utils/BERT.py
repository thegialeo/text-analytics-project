from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from utils import gpu


class BERT:
    
    def __init__(self, X_train, y_train, X_test, y_test):
        """BERT wrapper class

           Written by Leo Nguyen. Contact Xenovortex, if problems arises.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.lr = 1e-4
        self.device = gpu.check_gpu()

        # Load BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

        # Load pretrained BERT and move to GPU (if available)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=6, output_attentions=False, output_hidden_states=False)
        self.model = self.model.to(self.device)

        # init optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-8)