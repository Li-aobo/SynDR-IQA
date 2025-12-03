from torchvision.models import resnet50
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, pretrain=True, output_f=False):
        super(Baseline, self).__init__()
        self.output_f = output_f

        model = resnet50(pretrained=pretrain)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        in_features = 2048

        self.classify = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1),
        )

        weight_init(self.classify)

    def forward(self, x):
        x = self.backbone(x)
        f = x.reshape(x.size(0), -1)
        x = self.classify(f)

        if self.output_f:
            return x, f
        else:
            return x


def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
