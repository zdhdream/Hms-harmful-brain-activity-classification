import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp
import timm
from cfg import CFG
import numpy as np
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()
        
class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super(AttBlockV2, self).__init__()
        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (16, 1280, 16)
        # attention weights
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        # nonlinear transform
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)

# v3
class HMSHBACSpecModelSED(nn.Module):

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
            stage='finetune',
    ):
        super().__init__()
        self.cfg = CFG
        self.in_chans = in_channels
        self.num_classes = num_classes
        self.stage = stage
        pretrained_cfg = timm.create_model(model_name=model_name, pretrained=False).default_cfg
        print(pretrained_cfg)
        pretrained_cfg['file'] = f"{model_name}.pth"
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            drop_rate=CFG.backbone_dropout,
            drop_path_rate=CFG.backbone_droppath,
            in_chans=in_channels,
            # global_pool="",
            # num_classes=0,
            pretrained_cfg=pretrained_cfg
        )
        layers = list(self.model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if "efficientnet" in self.cfg.model_name:
            in_features = self.model.classifier.in_features
        elif "eca" in self.cfg.model_name:
            in_features = self.model.head.fc.in_features
        elif "res" in self.cfg.model_name:
            in_features = self.model.fc.in_features
        self.bn0 = nn.BatchNorm2d(1536)
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, self.num_classes, activation="sigmoid")
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)

    def freeze(self):
        if self.stage == "finetune":
            self.encoder.eval()
            self.fc1.eval()
            self.bn0.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.fc1.parameters():
                param.requires_grad = False
            for param in self.bn0.parameters():
                param.requires_grad = False
        return

    def gem_pooling(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def avg_pooling(self, x, p=1, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def operate(self, x):
        # kaggle spectrograms
        # print(x.shape)
        x1 = [x[:, i:i + 1, :, :] for i in range(3)]  # x: [bs,8,256,512]
        x1 = torch.concatenate(x1, dim=2)  # (bs, 1, 512, 1536)
        # eeg spectrograms
        # x2 = [x[:, i + 4:i + 5, :, :] for i in range(4)]
        # x2 = torch.concatenate(x2, dim=2)  # (bs, 1, 512, 256)
        # x = torch.concatenate([x1, x2], dim=3)  # (bs,1,512,512)
        return x1

    def extract_features(self, x):
        """
        :param x: (bs, n_channles, n_mels, n_frames)
        :return:
        """
        x = x.permute((0, 1, 3, 2))  # (bs, n_channles, n_frames, n_mels)
        frames_num = x.shape[2]
        x = x.transpose(1, 3)  # (bs, n_mels, n_frames, n_channels)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (bs, n_channels, n_frames, n_mels)
        x = x.transpose(2, 3)  # (bs, channles, n_mels, n_frames)
        x = self.encoder(x)

        # (bs, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2 # (16,1280, 16)
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2) # (16, 16, 1280)
        x = F.relu_(self.fc1(x)) # (16, 16, 1280)
        x = x.transpose(1, 2) # (16,1280,16)
        x = F.dropout(x, p=0.5, training=self.training) # (16,1280, 16)
        return x, frames_num

    def forward(self, x):
         # (bs,3,512,512)
        # if self.in_chans == 3:
        #     x = image_delta(x)
        x, frames_num = self.extract_features(x) # (16, 1280, 16)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=-1) 
        return logit
    
class HMSHBACSpecModel(nn.Module):

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
            path=None,
    ):
        super().__init__()
        pretrained_cfg = timm.create_model(model_name=model_name, pretrained=False).default_cfg
        
        pretrained_cfg['file'] = f"{model_name}.pth"
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            drop_rate=CFG.backbone_dropout,
            drop_path_rate=CFG.backbone_droppath,
            in_chans=in_channels,
            global_pool="",
            num_classes=0,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay = {'pool_size':CFG.pretrained_cfg_overlay}
        )
        print(self.model.pretrained_cfg)
        if path is not None:
            self.model.load_state_dict(torch.load(path))
        in_features = self.model.num_features
        self.fc = nn.Linear(2 * in_features, num_classes)
        init_layer(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1)
        )

        self.dropout = nn.Dropout(p=CFG.head_dropout)

    def gem_pooling(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
    
    def operate(self, x):
        # kaggle spectrograms
        #print(x.shape)
        x1 = [x[:, i:i + 1, :, :] for i in range(3)]  # x: [bs,8,256,512]
        x1 = torch.cat(x1, dim=2)  # (bs, 1, 1280, 512)
        # eeg spectrograms
        #x2 = [x[:, i + 4:i + 5, :, :] for i in range(4)]
        #x2 = torch.concatenate(x2, dim=2)  # (bs, 1, 512, 256)
        #x = torch.concatenate([x1, x2], dim=3)  # (bs,1,512,512)
        return x1

    def forward(self, x):
        batch_size = x.shape[0]
        #x  (bs,3, 1536, 512)
        x = self.model(x)
        xgem = self.gem_pooling(x)[:, :, 0, 0]
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        xatt = (x * attn_weights).sum(dim=1)
        x = torch.cat([xgem, xatt], dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
if __name__ == '__main__':
    print(1)