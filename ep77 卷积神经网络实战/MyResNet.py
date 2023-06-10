import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):  # 残差模块实现
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """
        
        :param ch_in: 
        :param ch_out: 
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        # [64, 128,
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)  # 方便train起来更稳定

        self.extra_module = nn.Sequential()
        if ch_in != ch_out:
            self.extra_module = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        # short_cut.
        # extra_module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add
        # print('extra_out:', self.extra_modul)
        out = self.extra_module(x) + out  # F(x) = x + H(x)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h, w]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(128, 256, stride=2)
        # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)
        # [b, 512, h, w] => [b, 512, h, w]
        self.blk4 = ResBlk(512, 512, stride=2)  # 注：每一块的h和w都不同

        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print('after_conv:', x.shape)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # [b, 512, h, w] => [b, 512, 1, 1]
        # print('after_pool:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


def main():
    blk = ResBlk(64, 128, stride=4)  # 通过stride降低图片维度以减小数据量
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print('block:', out.shape)

    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)
    print('resnet:', out.shape)


if __name__ == '__main__':
    main()
