from transformers import BertTokenizer, BertModel, AdamW, BertConfig, get_linear_schedule_with_warmup
from utils import gpu


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
            input_ids (pytorch tensor): BERT vocabulary indices of sentences 
            segments (pytorch tensor): segment IDs of sentences (needed as BERT input)
        """
        
        token_lst = []

        for sentence in sentences:
            
            # add special tokens
            sentence = "[CLS]" + sentence + "[SEP]"
            
            # tokenize
            tokens = self.tokenizer.tokenize(sentence)

            # vocabulary indices as pytorch tensor
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([input_ids])

            # segment ID as pytorch tensor
            segments = [1] * len(tokens)
            segments = torch.tensor([segments])





