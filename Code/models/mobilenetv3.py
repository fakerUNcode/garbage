'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from itertools import chain


# h-swish 激活函数
class hswish(nn.Module):
    def forward(self, x):
        # 实现 h-swish 激活函数: x * ReLU6(x + 3) / 6
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out




# h-sigmoid 激活函数
class hsigmoid(nn.Module):
    def forward(self, x):
        # 实现 h-sigmoid 激活函数: ReLU6(x + 3) / 6
        out = F.relu6(x + 3, inplace=True) / 6
        return out




# SE模块，用于通道注意力机制
class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        """
        参数:
        - in_size: 输入通道数
        - reduction: 通道数压缩比例
        """
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),  # 压缩通道数
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),  # 恢复通道数
            nn.BatchNorm2d(in_size),
            hsigmoid()  # h-sigmoid 激活函数
        )

    def forward(self, x):
        # 输入与 SE 模块的输出相乘
        return x * self.se(x)


# Inverted Residual Block
class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        """
        参数:
        - kernel_size: 卷积核大小
        - in_size: 输入通道数
        - expand_size: 扩展后的通道数
        - out_size: 输出通道数
        - nolinear: 激活函数
        - semodule: SE模块
        - stride: 步幅
        """
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        # 扩展通道数的1x1卷积
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear

        # 深度可分离卷积
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear

        # 压缩通道数的1x1卷积
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        # Shortcut连接，如果 stride=1 且输入输出通道数不一致，需要通过1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        # 前向传播
        out = self.nolinear1(self.bn1(self.conv1(x)))  # 扩展通道数
        out = self.nolinear2(self.bn2(self.conv2(out)))  # 深度卷积
        out = self.bn3(self.conv3(out))  # 压缩通道数
        if self.se is not None:  # 如果存在SE模块，则应用
            out = self.se(out)
        # 如果 stride=1，则加上 Shortcut 的结果
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out



# MobileNetV3 大模型
class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        """
        初始化 MobileNetV3_Large 模型。

        参数:
        - num_classes: 分类的类别数，默认值为1000（ImageNet分类）。
        """
        super(MobileNetV3_Large, self).__init__()

        # 输入阶段
        # 首个卷积层：将输入特征从3通道（RGB图像）提升到16通道，卷积核大小为3x3，步幅为2，使用 h-swish 激活函数
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)  # 批量归一化
        self.hs1 = hswish()  # h-swish 激活函数

        # 主干网络：由一系列 Inverted Residual Block 组成
        self.bneck = nn.Sequential(
            # 每个 Block 定义一个 Inverted Residual Block，包含卷积、激活函数、SE模块等
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),          # 第一个 Block，无 SE 模块
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),          # 第二个 Block，无 SE 模块
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),          # 第三个 Block，无 SE 模块
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),  # 第四个 Block，使用 SE 模块
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1), # 第五个 Block，使用 SE 模块
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1), # 第六个 Block，使用 SE 模块
            Block(3, 40, 240, 80, hswish(), None, 2),                      # 第七个 Block，无 SE 模块
            Block(3, 80, 200, 80, hswish(), None, 1),                      # 第八个 Block，无 SE 模块
            Block(3, 80, 184, 80, hswish(), None, 1),                      # 第九个 Block，无 SE 模块
            Block(3, 80, 184, 80, hswish(), None, 1),                      # 第十个 Block，无 SE 模块
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),            # 第十一个 Block，使用 SE 模块
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),           # 第十二个 Block，使用 SE 模块
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),           # 第十三个 Block，使用 SE 模块
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),           # 第十四个 Block，使用 SE 模块
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),           # 第十五个 Block，使用 SE 模块
        )

        # 输出阶段
        # 卷积层：将特征维度从 160 提升到 960，卷积核大小为 1x1
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)  # 批量归一化
        self.hs2 = hswish()  # h-swish 激活函数

        # 全连接层：将特征从 960 维转换为 1280 维
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)  # 批量归一化
        self.hs3 = hswish()  # h-swish 激活函数

        # 输出层：将特征从 1280 维转换为类别数（num_classes）
        self.linear4 = nn.Linear(1280, num_classes)

        # 初始化模型参数
        self.init_params()

    def init_params(self):
        """
        初始化网络的参数：
        - Conv2d 层：使用 He 初始化
        - BatchNorm 层：权重初始化为 1，偏置初始化为 0
        - Linear 层：权重使用正态分布，偏置初始化为 0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """
        定义模型的前向传播过程：
        - 输入 x: 形状为 (batch_size, 3, H, W)，即 RGB 图像数据
        """
        # 输入阶段：卷积 + BatchNorm + h-swish
        out = self.hs1(self.bn1(self.conv1(x)))
        # 主干网络阶段：一系列 Block 的堆叠
        out = self.bneck(out)
        # 输出阶段：卷积 + BatchNorm + h-swish
        out = self.hs2(self.bn2(self.conv2(out)))
        # 全局平均池化，将 (C, H, W) 的特征转换为 (C) 的全局表示
        out = F.avg_pool2d(out, 7)
        # 展平，将多维特征转换为一维向量
        out = out.view(out.size(0), -1)
        # 全连接层：将特征从 960 维提升到 1280 维
        out = self.hs3(self.bn3(self.linear3(out)))
        # 最终分类层：输出类别数（num_classes）
        out = self.linear4(out)
        return out



class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        """
        初始化 MobileNetV3_Small 模型。

        参数:
        - num_classes: 分类的类别数，默认值为1000（ImageNet分类）。
        """
        super(MobileNetV3_Small, self).__init__()
        # 第一个卷积层，输入为3通道（RGB图像），输出为16通道，使用 h-swish 激活函数
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        # 主干网络，由多个 Inverted Residual Block 组成
        self.bneck = nn.Sequential(
            # 第一个 Block，使用 SE 模块
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            # 第二个 Block，无 SE 模块
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            # 第三个 Block，无 SE 模块
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            # 第四个 Block，使用 SE 模块
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            # 第五到第七个 Block，均使用 SE 模块
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            # 第八个 Block，使用 SE 模块
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            # 第九到第十一个 Block，均使用 SE 模块
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )

        # 卷积层，输出特征数从 96 提升到 576
        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()

        # 全连接层，将特征从 576 维转换为 1280 维
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()

        # 输出层，分类到指定类别数
        self.linear5 = nn.Linear(1280, num_classes)

        # 初始化网络参数
        self.init_params()

        # 冻结的层：不需要训练的部分
        self.freeze = [self.conv1, self.bneck, self.conv2]

        # 微调的层：需要进行微调的部分
        self.fine_tune = [self.linear3, self.linear5]

    def fine_tune_params(self):
        """
        返回需要微调的层的参数。
        """
        return chain(*[f.parameters() for f in self.fine_tune])

    def freeze_params(self):
        """
        返回需要冻结的层的参数。
        """
        return chain(*[f.parameters() for f in self.freeze])

    def init_params(self):
        """
        初始化网络的参数。
        - Conv2d 层使用 He 初始化
        - BatchNorm 层的权重初始化为 1，偏置初始化为 0
        - Linear 层权重使用正态分布，偏置初始化为 0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """
        定义模型的前向传播过程。
        - x: 输入图像张量，形状为 (batch_size, 3, H, W)
        """
        # 输入阶段：卷积 + BatchNorm + h-swish
        out = self.hs1(self.bn1(self.conv1(x)))
        # 主干网络阶段
        out = self.bneck(out)
        # 卷积阶段：卷积 + BatchNorm + h-swish
        out = self.hs2(self.bn2(self.conv2(out)))
        # 全局平均池化，将 (C, H, W) 转换为 (C)
        out = out.mean([2, 3])
        # 全连接层，逐步转换到最终分类的维度
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear5(out)
        return out



def test():
    net = MobileNetV3_Small()
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()
