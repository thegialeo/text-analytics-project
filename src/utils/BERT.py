from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig


class BERT:
    
    def __init__(self, X_train, y_train, X_test, y_test, device):
        """BERT wrapper class

           Written by Leo Nguyen. Contact Xenovortex, if problems arises.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = device


        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=6, output_attentions=False, output_hidden_states=False)