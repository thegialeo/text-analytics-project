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
        self.model = BertModel.from_pretrained('bert-base-german-cased')
        self.model = self.model.to(self.device)
        self.model.eval()


    def preprocessing(self, sentences):
        """Prepare sentences for to conform with BERT input (tokenize, add special tokens, create Segment ID)

        Args:
            sentences (array-like): sentences to prepare for BERT input
        
        Return:
            input_tensor (pytorch tensor): BERT vocabulary indices of sentences 
            segment_tensor (pytorch tensor): segment IDs of sentences (needed as BERT input)
        """
        
        input_lst = []
        segment_lst = []

        for sentence in sentences:
            
            # add special tokens
            sentence = "[CLS]" + sentence + "[SEP]"
            
            # tokenize
            tokens = self.tokenizer.tokenize(sentence)

            # vocabulary indices as pytorch tensor
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_lst.append(input_ids)

            # segment ID as pytorch tensor
            segments = [1] * len(tokens)
            segment_lst.append(segments)

        # type cast to pytorch tensor
        input_tensor = torch.tensor(input_lst)
        segment_tensor = torch.tensor(segment_lst)
        
        return input_tensor, segment_tensor

    
    def get_features(self, input_tensor, segment_tensor):
        """Get features from input and segment tensor using BERT

        Args:
            input_tensor (array-like): BERT vocabulary indices of sentences
            segment_tensor (array-like): segment IDs of sentences (needed as BERT input)
        
        Return:
            features (array-like): feature array (num_sentence, num_features=768)
        """
        outputs = self.model(input_tensor, segment_tensor)

        # last hidden layer 
        last_hidden_states = outputs.last_hidden_state

        # average over all tokens
        features = torch.mean(last_hidden_states, dim = 1)
        
        return features





