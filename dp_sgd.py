import aihwkit
import dp_accounting
import math
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from aihwkit.optim import AnalogOptimizer
from aihwkit.optim.context import AnalogContext
from zhw323.DP.privacy_accountant import PrivacyAccountant


class DP_SGD(Optimizer):
    def __init__(self, params, batch_size,lr=0.1, noise_multiplier=1.0, max_grad_norm=1.0):
        defaults = dict(lr=lr, batch_size=batch_size, noise_multiplier=noise_multiplier,
                        max_grad_norm=max_grad_norm)
        super(DP_SGD, self).__init__(params, defaults)
        self.epsilon = []
        self.privacyAccount = None

    def createPrivacyAccount(self,total_size, delta, moment_order):
        #self.privacyAccount = PrivacyAccountant((self.defaults['batch_size']/total_size, delta, self.defaults['noise_multiplier'], moment_order))
        sample_probability = self.defaults['batch_size'] / total_size
        noise_multiplier = self.defaults['noise_multiplier']
        self.privacyAccount = PrivacyAccountant(sample_probability, delta, noise_multiplier, moment_order)
        
    def get_epsilon(self, epoch):
        return self.privacyAccount.compute_epsilon(epoch)
    
    """def privacyAccount(self, sample_probability, epoch, delta, moment_order=range(2, 33)):
        if epoch <= 0:
            raise ValueError(
                f'Number of epochs must be positive. Found{epoch}')
        if not 0 <= delta <= 1:
            raise ValueError(f'delta must between 0 and 1. Found{delta}')
        event_ = dp_accounting.GaussianDpEvent(
            self.defaults['noise_multiplier'])
        event_ = dp_accounting.PoissonSampledDpEvent(
            sampling_probability=sample_probability, event=event_)
        count = int(math.ceil(epoch / sample_probability))
        event_ = dp_accounting.SelfComposedDpEvent(count=count, event=event_)

        accountant = dp_accounting.rdp.RdpAccountant(
            orders=moment_order).compose(event=event_)
        epsilon = accountant.get_epsilon(delta)
        self.epsilon.append(epsilon)
        return epsilon"""

    def sanitizer(self, grad_list, noise_multiplier, max_grad_norm):
        # Flatten each tensor in the list and keep track of the original shapes
        flat_grad_list = [grad.view(-1) for grad in grad_list]
        original_shapes = [grad.size() for grad in grad_list]

        # Concatenate the flattened tensors
        concat_grad = torch.cat(flat_grad_list, dim=0)

        # Clip the concatenated gradients & Add noise
        clip_grad_norm_(concat_grad, max_norm=max_grad_norm, norm_type=2)
        noisy_grads = (1/self.defaults['batch_size'])*(concat_grad + torch.normal(mean=0, std=noise_multiplier * max_grad_norm, size=concat_grad.size(), device=concat_grad.device))

        # Split the concatenated gradients back into separate tensors
        split_sizes = [flat_grad.numel() for flat_grad in flat_grad_list]
        split_grads = torch.split(noisy_grads, split_sizes)

        # Reshape each tensor back to its original shape
        clipped_grad_list = [split_grad.view(
            original_shape) for split_grad, original_shape in zip(split_grads, original_shapes)]
        return clipped_grad_list

    def step(self, closure=None):
        for group in self.param_groups:
            grads = []
            use_Analog = False
            for p in group['params']:
                if isinstance(p, AnalogContext):
                    use_Analog = True
                    continue
                if p.grad is None:
                    continue
                grad = p.grad.data
                grads.append(grad)

            if not use_Analog:
                # If param_groups is not Analog Context
                # Sanitize the gradients by 1.Clip, 2.Add Noise
                """ noisy_grads = self.sanitizer(
                    grads, group['noise_multiplier'], group['max_grad_norm'])"""
                for p, grad in zip(group['params'], grads):
                    grad = (1/self.batch_size) * torch.normal(mean=0, std=self.noise_multiplier * self.max_grad_norm, size=grad.size(), device=grad.device)
                    p.data.add_(-group['lr'], grad)
