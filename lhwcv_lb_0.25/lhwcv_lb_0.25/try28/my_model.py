# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class Spec2D(nn.Module):
    def __init__(self, in_c=1, out_c=1, expansion=8, t_dilation=1):
        super(Spec2D, self).__init__()
        kernels_t = [1, 3, 5, 7, 9, 11]

        kernel_f = 3
        self.convs =nn.ModuleList()
        for k in kernels_t:
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=expansion,
                    kernel_size=(kernel_f, k),
                    padding="same",
                    dilation=(1, t_dilation)),)
        cs = len(kernels_t) * expansion
        if out_c !=cs:
            self.proj = nn.Conv2d(cs, out_c, kernel_size=1)
        else:
            self.proj = None
    def forward(self, x):
        """

        :param x: B, C, F, T
        :return:
        """
        x_convs = []
        for conv in self.convs:
            x_conv = conv(x)
            x_convs.append(x_conv)
        h = torch.cat(x_convs, dim=1)
        if self.proj is not None:
            h = self.proj(h)
        return h

class GatedSpec2D(nn.Module):
    def __init__(self, in_c=1, out_c=1, expansion=8, t_dilation=1, dropout=None):
        super().__init__()

        self.filt = nn.Sequential(
            Spec2D(in_c=in_c, out_c=6 * expansion, t_dilation=t_dilation),
            nn.BatchNorm2d(6*expansion),
            nn.LeakyReLU(),
        )
        # self.gate = nn.Sequential(
        #     Spec2D(in_c=in_c, out_c=6 * expansion, t_dilation=t_dilation),
        #     nn.BatchNorm2d(6 * expansion),
        #     nn.Sigmoid(),
        # )
        #self.gate = Spec2D(in_c=in_c, out_c=6*expansion, t_dilation=t_dilation)
        self.proj = nn.Conv2d(6*expansion, out_c, 1)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        """
        :param x: B, C, F, T
        :return:
        """
        x_filt = self.filt(x)
        #x_gate = self.gate(x)
        h = x_filt# * x_gate
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.proj(h)
        return h

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(1,1), dims=[48, 96, 144]):
        super(BasicBlock, self).__init__()
        self.in_conv = nn.Conv2d(in_c, dims[0], kernel_size=(1, 1))
        self.gated1 = nn.ModuleList()
        #self.gated2 = nn.ModuleList()
        #self.skip_convs = nn.ModuleList()
        for i,  in_dim in enumerate(dims[:-1]):
            expansion = in_dim // 6
            dim = dims[i+1]
            self.gated1.append(GatedSpec2D(in_dim, dim, expansion=expansion, t_dilation=1))
            #self.skip_convs.append(nn.Conv2d(in_dim, dim, kernel_size=(1, 1)))
            #self.gated2.append(GatedSpec2D(in_dim, dim, expansion=expansion, t_dilation=2))
        #exit(0)
        # Initialize parameters
        nn.init.xavier_uniform_(self.in_conv.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.in_conv.bias)
        # for i in range(len(self.skip_convs)):
        #     nn.init.xavier_uniform_(self.skip_convs[i].weight, gain=nn.init.calculate_gain("relu"))
        #     nn.init.zeros_(self.skip_convs[i].bias)

        self.n_layers = len(dims) - 1
        h_dim = dims[-1]
        self.out_conv = nn.Conv2d(h_dim, out_c, kernel_size=(3, 3), padding=(1, 1), stride=stride)

    def forward(self, x):
        """
        :param x: B, C, F, T
        :return:
        """
        x = self.in_conv(x)
        x_skip = x
        for layer in range(self.n_layers):
            x = self.gated1[layer](x)
            #x_skip = self.skip_convs[layer](x_skip)
            #x = x_skip + x
            #x = self.gated2[layer](x)
        x = self.out_conv(x)
        return x

class MyModel2(nn.Module):
    def __init__(self, fdim=128):
        super(MyModel2, self).__init__()
        self.block1 = BasicBlock(12,  48,    stride=(1, 2),  dims=[48,  48,  48])
        self.block2 = BasicBlock(48,  96,    stride=(2, 2),  dims=[96,  96,  96])
        self.block3 = BasicBlock(96,  192,   stride=(2, 2),  dims=[96,  144, 192])
        self.block4 = BasicBlock(192, 384,   stride=(2, 2),  dims=[192, 192, 384])
        #fea_dim = 128
        self.block5 = BasicBlock(384, 384,   stride=(2, 2),  dims=[384, 384, 384])

        #self.fea_dim = fea_dim * fdim // 16
        self.out = nn.Linear(384, 6)


    def forward(self, x):
        """
        :param x: B, C, F, T
        :return:
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        #print(x.shape)
        x = x.mean(dim=-1)
        x = x.mean(dim=-1)
        #b, c, f = x.shape
        #x = x.reshape(b, -1)
        x = self.out(x)
        return  x


if __name__ == '__main__':

    x = torch.zeros(1, 12, 128, 300)
    model = MyModel2()
    #print(model)
    y = model(x)
    print('y shape: ', y.shape)
    torch.save(model.state_dict(), '/home/lhw/model.pth')