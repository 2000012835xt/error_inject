import torch
import torch.nn as nn

from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn
from .quantizer import Quantizer
import torch.nn.functional as F
import pdb

def inject_error(result, prob, bw=32, bw_hardware=24):
    # err_prob: result.shape + [bw]

    prob = (
        torch.Tensor([prob] * bw).to(result.device) if not isinstance(prob, list)
        else torch.Tensor(prob).to(result.device)
        )
    
    err_prob = prob.repeat(list(result.shape) + [1])
    # generate error bit mask for each bit
    err_bit = torch.bernoulli(err_prob).to(result.device)
    weight = torch.Tensor([2 ** i for i in range(len(prob))]).to(result.device)
    err = torch.sum(err_bit * weight, dim=-1, keepdim=False).to(result.device)
    err = err.type(result.dtype).to(result.device)

    err_comp = err + 2 ** (len(prob) - 1)
    result = torch.where(
        err >= 0,
        torch.bitwise_xor(result, err),
        torch.where(
            result >= 0,
            torch.bitwise_xor(result, err_comp) - 2 ** (bw_hardware - 1),
            torch.bitwise_xor(result, err_comp) + 2 ** (bw_hardware - 1),
        )
    )
    return result


class quant_ConvReLU2d_error(nn.Module):
    def __init__(self, in_channels, out_channels, weight, bias, scale0, scale1, scale2, prob, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = weight.cuda()
        self.bias = bias.cuda()
        self.scale0 = scale0
        self.scale1 = scale1
        self.scale2 = scale2

        self.quan0 = Quantizer(bit=8, scale=scale0, all_positive=False)
        self.quan1 = Quantizer(bit=8, scale=scale1, all_positive=False)
        self.quan2 = Quantizer(bit=8, scale=scale2, all_positive=True)
        self.relu = nn.ReLU(inplace=True)
        self.prob = prob

    def forward(self, x):
        # q_input = self.quan0(x)
        q_input = x / self.scale0
        q_weight = self.quan1(self.weight) / self.scale1
        # pdb.set_trace()
        qresult = F.conv2d(q_input, q_weight, None, self.stride, self.padding)
        # qresult = qresult / (self.scale0 * self.scale1)
        qresult = qresult.to(torch.int32)
        # pdb.set_trace()
        qresult = inject_error(qresult, self.prob)
        bias = self.bias[None, :, None, None]
        y = qresult * self.scale0 * self.scale1 + bias

        # y = F.conv2d(x, q_weight, self.bias, padding=self.padding)
        y = self.relu(y)
        y = self.quan2(y)

        return y
    


class quant_ConvReLU2d(nn.Module):
    def __init__(self, in_channels, out_channels, weight, bias, scale0, scale1, scale2, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = weight.cuda()
        self.bias = bias.cuda()
        self.scale0 = scale0
        self.scale1 = scale1
        self.scale2 = scale2
        self.quan1 = Quantizer(bit=8, scale=scale1, all_positive=False)
        self.quan2 = Quantizer(bit=8, scale=scale2, all_positive=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        q_weight = self.quan1(self.weight)
        y = F.conv2d(x, q_weight, self.bias, padding=self.padding)
        y = self.relu(y)
        y = self.quan2(y)

        return y
    

class quant_LinearReLU(nn.Module):
    def __init__(self, in_channels, out_channels, weight, bias, scale1, scale2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.shape = (out_channels, in_channels)
        # self.weight = nn.Parameter((torch.rand(self.shape)-0.5) * 0.001, requires_grad=True)
        # self.bias = nn.Parameter((torch.rand(self.out_channels)-0.5) * 0.001, requires_grad=True)
        self.weight = weight.cuda()
        self.bias = bias.cuda()
        self.scale1 = scale1
        self.scale2 = scale2
        self.quan1 = Quantizer(bit=8, scale=scale1, all_positive=False)
        self.quan2 = Quantizer(bit=8, scale=scale2, all_positive=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        q_weight = self.quan1(self.weight)
        y = F.linear(x, q_weight, self.bias)
        y = self.relu(y)
        y = self.quan2(y)

        return y
    

class quant_Linear(nn.Module):
    # 用在vgg最后一层，没有ReLU
    def __init__(self, in_channels, out_channels, weight, bias, scale1, scale2, zero_point):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.shape = (out_channels, in_channels)
        # self.weight = nn.Parameter((torch.rand(self.shape)-0.5) * 0.001, requires_grad=True)
        # self.bias = nn.Parameter((torch.rand(self.out_channels)-0.5) * 0.001, requires_grad=True)
        self.weight = weight.cuda()
        self.bias = bias.cuda()
        self.scale1 = scale1
        self.scale2 = scale2
        self.quan1 = Quantizer(bit=8, scale=scale1, all_positive=False)
        self.quan2 = Quantizer(bit=8, scale=scale2, zero_point=zero_point, all_positive=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        q_weight = self.quan1(self.weight)
        y = F.linear(x, q_weight, self.bias)
        y = self.quan2(y)

        return y


stage = [64, 64, 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]
layer_injection = [2, 3, 5, 7, 9, 10, 13, 15]


class quant_VGG16_error(nn.Module):
    def __init__(
        self, num_classes: int = 10, 
        input_scale=None, input_zero_point=None,
        conv_weights=None, conv_bias=None, 
        linear_weights=None, linear_bias=None,
        conv_in_scale=None,
        conv_w_scale=None, conv_a_scale=None, 
        linear_w_scale=None, linear_a_scale=None, zero_point=None, 
        error_prob_injection=None,
        init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)

        self.input_quant = Quantizer(bit=8, scale=input_scale, zero_point=input_zero_point, all_positive=False)
        self.features = nn.ModuleList()

        i = 0
        j = 0
        in_channels = 3
        for v in stage:
            if v == "M":
                self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                v = cast(int, v) # 将v转换成整数
                if i in layer_injection:
                    QuantizedConvReLU2d = quant_ConvReLU2d_error(in_channels, v, conv_weights[i], conv_bias[i], conv_in_scale[i],
                                                       conv_w_scale[i], conv_a_scale[i], error_prob_injection[j], kernel_size=3, padding=1)
                    self.features.append(QuantizedConvReLU2d)
                    in_channels = v
                    j += 1

                else:
                    QuantizedConvReLU2d = quant_ConvReLU2d(in_channels, v, conv_weights[i], conv_bias[i], conv_in_scale[i],
                                                        conv_w_scale[i], conv_a_scale[i], kernel_size=3, padding=1)
                    self.features.append(QuantizedConvReLU2d)
                    in_channels = v

                i += 1
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.ReLU(True),
            # nn.Dropout(p=dropout),
            # nn.Linear(512, 512),
            # nn.ReLU(True),
            # nn.Dropout(p=dropout),
            # nn.Linear(512, num_classes),
            quant_LinearReLU(512, 512, linear_weights[0], linear_bias[0], linear_w_scale[0], linear_a_scale[0]),
            quant_LinearReLU(512, 512, linear_weights[1], linear_bias[1], linear_w_scale[1], linear_a_scale[1]),
            quant_Linear(512, num_classes, linear_weights[2], linear_bias[2], linear_w_scale[2], linear_a_scale[2], zero_point)
        )
        
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        # pdb.set_trace()
        for block in self.features:
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

