import copy
import torch
from torch import nn

class DifferentiableBatchNorm2d(nn.BatchNorm2d):

    def fresh_statistics(self):
        self.running_mean = torch.zeros_like(self.running_mean)
        self.running_var = torch.ones_like(self.running_var)
        self.num_batches_tracked = torch.zeros_like(self.num_batches_tracked)
        self._calibration_count = 0

    def forward(self, x):
        axes = (0, 2, 3)
        if self.training:
            count = x.numel() // x.shape[1]
            if count <= 1:
                raise ValueError('BN calibration needs at least two values per channel')
            variance, mean = torch.var_mean(x, dim=axes, unbiased=False)
            self._calibration_count += 1
            fraction = 1.0 / self._calibration_count
            self.running_mean = (1 - fraction) * self.running_mean + fraction * mean
            self.running_var = (1 - fraction) * self.running_var + fraction * variance * (count / (count - 1))
        else:
            mean, variance = (self.running_mean, self.running_var)
        result = (x - mean[None, :, None, None]) * torch.rsqrt(variance[None, :, None, None] + self.eps)
        if self.affine:
            result = result * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return result

class CalibratedObjective:

    def __init__(self, model, start, calibration_batches, attack_inputs, attack_labels):
        self.model = copy.deepcopy(model)
        self.names = [(name, param.shape, param.numel()) for name, param in model.named_parameters()]
        self.start = start.detach()
        self.calibration_batches = tuple(calibration_batches)
        self.inputs, self.labels = (attack_inputs, attack_labels)
        self.layers = []
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if not module.track_running_stats:
                    raise ValueError('calibration requires tracked BN statistics')
                module.__class__ = DifferentiableBatchNorm2d
                self.layers.append(module)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                raise ValueError('calibration supports BatchNorm2d only')
        if self.layers and (not self.calibration_batches):
            raise ValueError('calibration reserve is required for BN models')

    def logits(self, delta):
        candidate = self.start + delta.to(self.start.dtype)
        params = {}
        offset = 0
        for name, shape, count in self.names:
            params[name] = candidate[offset:offset + count].view(shape)
            offset += count
        if candidate.numel() != offset:
            raise ValueError('parameter vector length mismatch')
        for layer in self.layers:
            layer.fresh_statistics()
        if self.layers:
            self.model.train()
            for batch in self.calibration_batches:
                torch.func.functional_call(self.model, params, (batch,))
        self.model.eval()
        return torch.func.functional_call(self.model, params, (self.inputs,))

    def __call__(self, delta):
        return nn.functional.cross_entropy(self.logits(delta), self.labels)
