from transformers import BertTokenizer


class BERT:
    
    def __init__(self):
        """BERT wrapper class

           Written by Leo Nguyen. Contact Xenovortex, if problems arises.
        """

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=True)