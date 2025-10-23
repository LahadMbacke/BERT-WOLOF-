from bert.optimizer import ScheduledOptim
from transformers import AdamW  # Import AdamW optimizer

import torch.nn as nn
from tqdm import tqdm


class BertTrainer:
    
    def __init__(self,model,train_dataloader,test_dataloader,lr=2e-5,weight_decay=0.01,betas=(0.9,0.999),warmup_steps=10000,log_frequency=10,device="cuda"):
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.lr = lr
        self.device = device


        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        # pass d_model explicitly and warmup steps to the scheduler wrapper
        self.optimizer_scheduler = ScheduledOptim(self.optimizer, d_model=self.model.bert.d_model, n_warmup_steps=warmup_steps)

        # Use CrossEntropyLoss (expects raw logits) and ignore padding index 0 for MLM
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.log_frequency = log_frequency
        print("Total Parameters:", sum([p.numel() for p in self.model.parameters()]))


    def train(self,epoch):
        self.iteration(epoch, self.train_data, train=True)
    
    def test(self,epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self,epoch,dataloader,train=True):

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        if train:
            self.model.train()
        else:
            self.model.eval()

        # progress bar for visualizing the training progress
        t = tqdm(enumerate(dataloader), desc="Epoch %d" % (epoch), total=len(dataloader), bar_format="{l_bar}{r_bar}")  

        for i, data in t:

            # Move tensors to chosen device
            data = {key: value.to(self.device) for key, value in data.items()}

            # Forward the next_sentence_prediction model
            next_sentence_prediction_logits, masked_lm_logits = self.model(data["input_ids"], data["segment_labels"])

            # NLL Loss of is_next_sentence result
            nsp_loss = self.criterion(next_sentence_prediction_logits, data["is_next"])

            # NLL Loss of masked_lm result
            masked_lm_loss = self.criterion(masked_lm_logits.view(-1, masked_lm_logits.size(-1)), data["masked_lm_labels"].view(-1))
            # Total loss
            loss = nsp_loss + masked_lm_loss

            # Backward and optimize
            if train:
                self.optimizer_scheduler.zero_grad()
                loss.backward()
                self.optimizer_scheduler.step_and_update_lr()

            # next sentence prediction accuracy
            correct = next_sentence_prediction_logits.max(dim=1)[1].eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].numel()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "accuracy": total_correct / total_element * 100,
            }

            if i % self.log_frequency == 0:
                t.set_postfix(post_fix)
            
        print("Epoch %d %s Loss = %.4f, Accuracy = %.4f" % (epoch, "Train" if train else "Test", avg_loss / len(dataloader), total_correct / total_element * 100))



