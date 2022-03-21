import torch
import torch.nn as nn
import ResNet
import dynamic_factor
import time

#DeepMEDA_alexnet version
class DeepMEDA(nn.Module):

    def __init__(self, dropout_keep=None, num_classes=5,bottle_neck=True):
        """Init classifier."""
        super(DeepMEDA, self).__init__()
        self.mmd_loss = dynamic_factor.MMD_loss()
        self.dropout_keep = dropout_keep
        #self.bottle_neck = bottle_neck
        self.classifier = nn.Sequential(
            nn.Linear(256 * 10, 256 * 5),
            # nn.Linear(256 *5, 256 * 1),
            # nn.BatchNorm1d(256 * 5),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_keep),  # 0.5
            nn.Linear(256 * 5, 256)
        )
        self.cls_fc = nn.Linear(256, num_classes)
        # if bottle_neck:
        #     self.bottle = nn.Linear(2560, 256)
        #     self.cls_fc = nn.Linear(256, num_classes)
        # else:
        #     self.cls_fc = nn.Linear(2048, num_classes)

#初始化训练start=True则直接前向传播
#否则使用伪标签  且source target为提取过的特征   DeepMEDA只是一个分类器
    def forward(self, source, target, s_label,prob=torch.tensor([], dtype=torch.float32),start=True):
        out = self.classifier(source)
        if start:
            return out
        else:
            # 前向传播顺便打标签
            if prob==torch.Size([]):
                t_label = self.classifier(target)
                prob = self.cls_fc(t_label)
            #检查看是不是prob的问题
            t_label = self.classifier(target)
            prob = self.cls_fc(t_label)
            loss_c = self.mmd_loss.conditional(source, target, s_label, torch.nn.functional.softmax(prob, dim=1))
            loss_m = self.mmd_loss.marginal(source, target)
            mu= 0.4
            # mu = dynamic_factor.estimate_mu(source.detach().cpu().numpy(), s_label.detach().cpu().numpy(),
            #                                 target.detach().cpu().numpy(),
            #                                 torch.max(prob, 1)[1].detach().cpu().numpy())


            return out,loss_c, loss_m, mu











        source = self.feature_layers(source)
        if self.bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)
        target = self.feature_layers(target)
        if self.bottle_neck:
            target = self.bottle(target)
        #前向传播顺便打标签
        loss_c = self.mmd_loss.conditional(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
        loss_m = self.mmd_loss.marginal(source, target)
        mu = dynamic_factor.estimate_mu(source.detach().cpu().numpy(), s_label.detach().cpu().numpy(), target.detach().cpu().numpy(), torch.max(t_label, 1)[1].detach().cpu().numpy())
        return s_pred, loss_c, loss_m, mu

    def predict(self, x):
        x = self.feature_layers(x)
        if self.bottle_neck:
            x = self.bottle(x)
        return self.cls_fc(x)























# class DeepMEDA(nn.Module):
#
#     def __init__(self, num_classes=31, bottle_neck=True):
#         super(DeepMEDA, self).__init__()
#         self.feature_layers = ResNet.resnet50(True)
#         self.mmd_loss = dynamic_factor.MMD_loss()
#         #1*1卷积层
#         self.bottle_neck = bottle_neck
#         if bottle_neck:
#             self.bottle = nn.Linear(2048, 256)
#             self.cls_fc = nn.Linear(256, num_classes)
#         else:
#             self.cls_fc = nn.Linear(2048, num_classes)
#
#
#     def forward(self, source, target, s_label,t_label):
#         source = self.feature_layers(source)
#         if self.bottle_neck:
#             source = self.bottle(source)
#         s_pred = self.cls_fc(source)
#         target = self.feature_layers(target)
#         if self.bottle_neck:
#             target = self.bottle(target)
#         #前向传播顺便打标签
#         loss_c = self.mmd_loss.conditional(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
#         loss_m = self.mmd_loss.marginal(source, target)
#         mu = dynamic_factor.estimate_mu(source.detach().cpu().numpy(), s_label.detach().cpu().numpy(), target.detach().cpu().numpy(), torch.max(t_label, 1)[1].detach().cpu().numpy())
#         return s_pred, loss_c, loss_m, mu
#
#     def predict(self, x):
#         x = self.feature_layers(x)
#         if self.bottle_neck:
#             x = self.bottle(x)
#         return self.cls_fc(x)
