from transformers import BertTokenizer, BertModel, AdamW, BertConfig, get_linear_schedule_with_warmup
#from utils import gpu


class BERT:
    
    def __init__(self):
        """BERT wrapper class

           Written by Leo Nguyen. Contact Xenovortex, if problems arises.
        """
        self.device = gpu.check_gpu()

        # Load BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=True)

        # Load pretrained BERT and move to GPU (if available)
        self.model = BertModel.from_pretrained('bert-base-german-cased', output_hidden_states=True)
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocessing(self, sentences):
        """Prepare sentences for to conform with BERT input (tokenize, add special tokens, create Segment ID)

        Args:
            sentences (array-like): sentences to prepare for BERT input
        
        Return:
            tokens (pytorch tensor): token tensor of sentences conforming with BERT input
            segments (pytorch tensor): segment IDs of sentences (needed as BERT input)
        """
        pass

