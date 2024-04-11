import dp_accounting
import math


class PrivacyAccountant():
    def __init__(self,sample_probability=0.01,delta=1e-5, noise_multiplier=4.0, moment_order=range(2, 33)):
        self.sample_probability = sample_probability
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.moment_order = moment_order
        self.epsilon = []

    def setDelta(self,delta):
        self.delta = delta
    
    def setNoise(self, noise_multiplier):
        self.noise_multiplier = noise_multiplier

    def _privacyAccount(self, epoch):
        if epoch <= 0:
            raise ValueError(
                f'Number of epochs must be positive. Found{epoch}')
        if not 0 <= self.delta <= 1:
            raise ValueError(f'delta must between 0 and 1. Found{self.delta}')
        event_ = dp_accounting.GaussianDpEvent(
            self.noise_multiplier)
        event_ = dp_accounting.PoissonSampledDpEvent(
            sampling_probability=self.sample_probability, event=event_)
        count = int(math.ceil(epoch / self.sample_probability))
        event_ = dp_accounting.SelfComposedDpEvent(count=count, event=event_)

        accountant = dp_accounting.rdp.RdpAccountant(
            orders=self.moment_order).compose(event=event_)
        epsilon = accountant.get_epsilon(self.delta)
        self.epsilon.append(epsilon)

    def compute_epsilon(self, epoch):
        self._privacyAccount(epoch)
        return self.epsilon[len(self.epsilon)-1]
