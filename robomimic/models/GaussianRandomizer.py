from robomimic.models.base_nets import Randomizer
from torch import normal,randn_like

class GaussianRandomizer(Randomizer):
    def __init__(self, mean=0.0, std=0.1):
        super(GaussianRandomizer, self).__init__()
        self.mean = mean
        self.std = std


    def output_shape_in(self, input_shape=None):
        return input_shape
    def output_shape_out(self, input_shape=None):
        return input_shape

    def forward_in(self, x):
        #return x + normal(self.mean, self.std, size=x.shape) #std-way, below from autopilot:
        return x + self.std * randn_like(x) + self.mean

    def forward_out(self, inputs):
        """
        Post-processing network outputs - not necessary...
        """
        return inputs

#no need for group or single encoder, nor the encoder factory...