from aihwkit.optim.analog_optimizer import AnalogOptimizer
import torch
from zhw323.DP.dp_tile import DPRPUConfig, DPSimulatorTile, DPTile
from zhw323.DP.dp_sgd import DP_SGD
from torch.nn.utils import clip_grad_norm_

class DPTrainer():
    def __init__(self, model, rpu_config, train_set_size=0,batch_size=0,noise_multiplier=1.0, max_grad_norm=1.0, lr=0.1,delta=1e-5, moment_order=range(2, 33)):
        assert isinstance(rpu_config, DPRPUConfig), "Error: rpu_config must not be an instance of class DPRPUConfig"
        assert batch_size != 0, "Error: batch_size must be none zero"
        assert train_set_size != 0, "Error: train_set_size must be none zero"
        self.model = model
        self.rpu_config = rpu_config
        self.train_set_size = train_set_size
        self.batch_size = batch_size
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.delta = delta
        self.moment_order = moment_order
    
    def create_DP_optimizer(self):
        self.optimizer = AnalogOptimizer(DP_SGD, self.model.parameters(), lr=self.lr, batch_size=self.batch_size, noise_multiplier=self.noise_multiplier, max_grad_norm=self.max_grad_norm)
        self.optimizer.createPrivacyAccount(total_size=self.train_set_size, delta=self.delta, moment_order=self.moment_order)
        self.optimizer.regroup_param_groups(self.model)

    def train_step(self,train_data, criterion):
        assert isinstance(train_data, torch.utils.data.DataLoader), "Error: train_data has to be an instance of class DataLoader"
        
        # For each batches
        for batch in train_data:
            for param in self.model.parameters():
                param.accumulated_grads = []
            
            #Foward, Backward for each sample
            for sample in batch:
                print(sample)
                x,y = sample
                y_hat = self.model(x)
                loss = criterion(y_hat, y)
                loss.backward()
            
                #Clip each parameter's per-sample gradient
                for param in self.model.parameters():
                    if param.grad is not None:
                        per_sample_grad = param.grad.detach().clone()
                        clip_grad_norm_(per_sample_grad, max_norm=self.max_grad_norm, norm_type=2)
                        param.accumulated_grads.append(per_sample_grad)

            for param in self.model.parameters():
                if param.accumulated_grads:
                    param.grad = torch.stack(param.accumulated_grads, dim=0)

            self.optimizer.step()        
            


