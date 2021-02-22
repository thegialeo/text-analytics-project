from transformers import BertTokenizer, BertModel, AdamW, BertConfig, get_linear_schedule_with_warmup
from utils import gpu
import torch
import numpy as np


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

        # add special tokens + padding/truncated to token length of 512 + create segment ID + cast to PyTorch tensor
        for sentence in sentences:
            encoding = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
            input_lst.append(encoding['input_ids'])
            segment_lst.append(encoding['attention_mask'])

        # cast list to PyTorch tensor
        input_tensor = torch.cat(input_lst, dim=0)
        segment_tensor = torch.cat(segment_lst, dim=0)

        return input_tensor, segment_tensor

    
    def get_features(self, input_tensor, segment_tensor):
        """Get features from input and segment tensor using BERT

        Args:
            input_tensor (array-like): BERT vocabulary indices of sentences
            segment_tensor (array-like): segment IDs of sentences (needed as BERT input)
        
        Return:
            features (array-like): feature array (num_sentence, num_features=768)
        """

        with torch.no_grad():
            # move inputs to device 
            input_tensor = input_tensor.to(self.device)
            segment_tensor = segment_tensor.to(self.device)

            # get BERT features
            outputs = self.model(input_tensor, segment_tensor)

        # last hidden layer 
        last_hidden_states = outputs.last_hidden_state

        # average over all tokens
        features = torch.mean(last_hidden_states, dim = 1)
        
        return features





