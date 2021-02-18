from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from utils import gpu


class BERT:
    
    def __init__(self, epoch, lr, trainloader, testloader):
        """BERT wrapper class

           Written by Leo Nguyen. Contact Xenovortex, if problems arises.
        """
        self.epoch = epoch
        self.lr = lr
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = gpu.check_gpu()

        # Load BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

        # Load pretrained BERT and move to GPU (if available)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=6, output_attentions=False, output_hidden_states=False)
        self.model = self.model.to(self.device)

        # init optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-8)

        # init scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=len(trainloader)*self.epoch)