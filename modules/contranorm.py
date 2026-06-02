import torch
import torch.nn as nn

"""ContraNorm：对比学习视角下的过度平滑及其超越
过度平滑是各种图神经网络 (GNN) 和 Transformer 中的常见现象，其性能会随着层数的增加而下降。我们不是从表示收敛到单个点的完全崩溃的角度来描述过度平滑，而是深入研究维度崩溃的更一般视角，其中表示位于一个狭窄的锥体中。
因此，受到对比学习在防止维度崩溃方面的有效性的启发，我们提出了一种称为 ContraNorm 的新型规范化层。直观地说，ContraNorm 隐式地破坏了嵌入空间中的表示，从而导致更均匀的分布和更轻微的维度崩溃。
在理论分析中，我们证明了 ContraNorm 在某些条件下可以缓解完全崩溃和维度崩溃。我们提出的规范化层可以轻松集成到 GNN 和 Transformer 中，并且参数开销可以忽略不计。
在各种真实数据集上的实验证明了我们提出的 ContraNorm 的有效性。
"""

class ContraNorm(nn.Module):
    def __init__(self, dim, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False):
        super().__init__()
        if learnable and scale > 0:
            import math
            if positive:
                scale_init = math.log(scale)
            else:
                scale_init = scale
            self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        self.dual_norm = dual_norm
        self.scale = scale
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity

        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        if self.scale > 0.0:
            xn = nn.functional.normalize(x, dim=2)
            if self.pre_norm:
                x = xn
            sim = torch.bmm(xn, xn.transpose(1,2)) / self.temp
            if self.dual_norm:
                sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
            else:
                sim = nn.functional.softmax(sim, dim=2)
            x_neg = torch.bmm(sim, x)
            if not self.learnable:
                if self.identity:
                    x = (1+self.scale) * x - self.scale * x_neg
                else:
                    x = x - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    x = scale * x - scale * x_neg
                else:
                    x = x - scale * x_neg
        x = self.layernorm(x)
        return x


if __name__ == '__main__':
    block = ContraNorm(dim=128, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False)
    input = torch.rand(32, 784, 128)
    output = block(input)
    print("Input size:", input.size())
    print("Output size:", output.size())





#         输入张量的最后一个维度的大小。这决定了 LayerNorm 的维度和可学习的缩放参数（如果启用）。
#
#     scale (float, 默认值 0.1):
#         控制对比学习过程中负样本对正样本的影响程度。如果 learnable 设置为 False，则使用固定的缩放因子。如果 learnable 设置为 True，则此值用于初始化可学习参数。
#
#     dual_norm (bool, 默认值 False):
#         如果设置为 True，则使用双重归一化（Softmax 在两个维度上分别进行归一化的和）。否则，使用单一归一化（Softmax 在一个维度上进行归一化）。
#
#     pre_norm (bool, 默认值 False):
#         如果设置为 True，则在对比学习之前进行归一化（使用 nn.functional.normalize）。这意味着归一化后的张量将用于计算相似度矩阵。
#
#     temp (float, 默认值 1.0):
#         温度参数，用于缩放相似度矩阵。较小的温度会使得相似度矩阵的值更加分散，较大的温度会使得相似度矩阵的值更加集中。
#
#     learnable (bool, 默认值 False):
#         如果设置为 True，则缩放参数是可学习的（即，它会作为模型的参数进行训练）。否则，使用固定的缩放因子。
#
#     positive (bool, 默认值 False):
#         如果设置为 True，则可学习的缩放参数使用对数初始化，以确保其值为正。
#
#     identity (bool, 默认值 False):
#         如果设置为 True，则输出是输入的缩放版本加上负样本的负缩放版本。否则，输出是输入减去负样本的缩放版本。
#
# forward 方法
#
#     x (torch.Tensor):
#         输入张量，形状为 (batch_size, sequence_length, dim)。
#
# 具体操作
#
#     归一化：
#         xn = nn.functional.normalize(x, dim=2): 对输入张量在最后一个维度上进行归一化。
#
#     相似度矩阵：
#         sim = torch.bmm(xn, xn.transpose(1, 2)) / self.temp: 计算归一化后张量的相似度矩阵，并用温度参数进行缩放。
#         根据 dual_norm 参数，决定是否在两个维度上进行 Softmax 归一化。
#
#     负样本：
#         x_neg = torch.bmm(sim, x): 计算负样本，通过相似度矩阵与原始输入张量相乘得到。
#
#     更新输入张量：
#         根据 learnable 和 identity 参数，决定如何更新输入张量：
#             如果 learnable 为 False，使用固定的缩放因子。
#             如果 learnable 为 True，使用可学习的缩放参数。
#
#     LayerNorm：
#         x = self.layernorm(x): 最后对张量应用 LayerNorm 进行归一化