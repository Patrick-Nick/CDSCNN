
""" Complex-valued Operation Units
    e.g.    Complex-valued Convolution (CC),
            Complex-valued Fully Connection (CFC),
            Complex-valued Batch Normalization (CBN),
            Complex-valued Maximum Pooling (CMP),
            Complex-Valued Average Pooling (CAP),
            and Complex-valued ReLU (CReLU).
@FileName: utils.py
@Author: Chenghong Xiao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _retrieve_elements_from_indices(tensor, indices):
    output = tensor.gather(dim=-1, index=indices).view_as(indices)
    return output

class ComplexMaxPool1d(nn.Module):
    """
    Complex-valued Maximum Pooling (CMP)
    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False, complex_axis=1):
        super(ComplexMaxPool1d, self).__init__()
        self.complex_axis = complex_axis
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        
    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_axis)
        absolute_value = torch.abs(real + 1j*imag)
        absolute_value, indices = F.max_pool1d(absolute_value, kernel_size = self.kernel_size, stride = self.stride, 
                               padding = self.padding, dilation = self.dilation, ceil_mode = self.ceil_mode, return_indices = True)
        real = _retrieve_elements_from_indices(real, indices)
        imag = _retrieve_elements_from_indices(imag, indices)
        
        return torch.cat([real, imag], self.complex_axis)


class ComplexLinear(torch.nn.Module):
    """
    Complex-valued Fully Connection (CFC)
    """
    def __init__(self, in_features, out_features, complex_axis=1):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features // 2
        self.out_features = out_features
        self.complex_axis = complex_axis
        self.real_linear = nn.Linear(self.in_features, self.out_features)
        self.imag_linear = nn.Linear(self.in_features, self.out_features)

    def forward(self, inputs):
        xr, xi = torch.chunk(inputs, 2, self.complex_axis)
        xr = xr.view(xr.size(0), -1)
        xi = xi.view(xi.size(0), -1)

        yrr = self.real_linear(xr)
        yri = self.imag_linear(xr)
        yir = self.real_linear(xi)
        yii = self.imag_linear(xi)

        yr = yrr - yii
        yi = yri + yir

        return torch.sqrt(torch.pow(yr, 2) + torch.pow(yi, 2))


class ComplexAvgPool2d(nn.Module):
    """
    Complex-Valued Average Pooling (CAP)
    """
    def __init__(self, output_size, complex_axis=1):
        super(ComplexAvgPool2d,self).__init__()
        self.output_size = output_size
        self.complex_axis = complex_axis

    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_axis)
        real = F.adaptive_avg_pool2d(real, self.output_size)
        imag = F.adaptive_avg_pool2d(imag, self.output_size)
        return torch.cat([real,imag], self.complex_axis)


class CReLU(nn.Module):
    """
    Complex-valued ReLU (CReLU)
    """
    def __init__(self, complex_axis=1, inplace=False):
        super(CReLU,self).__init__()
        self.r_relu = nn.ReLU(inplace=inplace)
        self.i_relu = nn.ReLU(inplace=inplace)
        self.complex_axis = complex_axis


    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_axis)
        real = self.r_relu(real)
        imag = self.i_relu(imag)
        return torch.cat([real,imag], self.complex_axis)

class ComplexConv1d(nn.Module):
    """
    Complex-valued Convolution (CC)
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, complex_axis=1):

        super(ComplexConv1d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if groups == in_channels:
            self.groups = groups // 2
        else:
            self.groups = 1
        self.dilation = dilation
        self.bias = bias
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                   self.padding, self.dilation, groups=self.groups, bias=self.bias)
        self.imag_conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                   self.padding, self.dilation, groups=self.groups, bias=self.bias)
        
        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        

    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_axis)

        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)
        
        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real

        return torch.cat([real, imag], self.complex_axis)


# Source: https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch 
# from https://github.com/IMLHF/SE_DCUNet/blob/f28bf1661121c8901ad38149ea827693f1830715/models/layers/complexnn.py#L55

class ComplexBatchNorm(torch.nn.Module):
    """
    Complex-valued Batch Normalization (CBN)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, complex_axis=1):
        super(ComplexBatchNorm, self).__init__()
        self.num_features        = num_features // 2
        self.eps                 = eps
        self.momentum            = momentum
        self.affine              = affine
        self.track_running_stats = track_running_stats
        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br  = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi  = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br',  None)
            self.register_parameter('Bi',  None)
        
        if self.track_running_stats:
            self.register_buffer('RMr',  torch.zeros(self.num_features))
            self.register_buffer('RMi',  torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones (self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones (self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr',                 None)
            self.register_parameter('RMi',                 None)
            self.register_parameter('RVrr',                None)
            self.register_parameter('RVri',                None)
            self.register_parameter('RVii',                None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9) # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert(xr.shape == xi.shape)
        assert(xr.size(1) == self.num_features)

    def forward(self, inputs):
        #self._check_input_dim(xr, xi)
        
        xr, xi = torch.chunk(inputs, 2, self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i!=1]
        vdim  = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)

        xr, xi = xr-Mr, xi-Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr   = Vrr + self.eps
        Vri   = Vri
        Vii   = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau   = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, Vri, Vri, value=-1)
        s     = delta.sqrt()
        t     = (tau + 2*s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst   = (s * t).reciprocal()
        Urr   = (s + Vii) * rst
        Uii   = (s + Vrr) * rst
        Uri   = (  - Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}'.format(**self.__dict__) 



        

