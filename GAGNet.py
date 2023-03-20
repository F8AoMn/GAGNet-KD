import torch
import torch.nn as nn
import math
from torch.nn import Parameter
from GAGNet-KD.mobilenetv2 import mobilenet_v2
from GAGNet-KD.GCN_11_4 import EAGCN
import torch.nn.functional as F


class SEM(nn.Module):
    def __init__(self, in_channel):
        super(SEM, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.conv1_1(x) + self.conv1_2(x)
        conv3 = self.conv3_1(conv1) + self.conv3_2(conv1)
        return conv3


class FIOM(nn.Module):
    def __init__(self, in_channel):
        super(FIOM, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        # self.conv_f = nn.Conv2d(in_channel, 1, 1, 1, 0)

    def forward(self, fi):
        B, C, H, W = fi.size()
        gap = self.gap(fi).view(B, C, -1)
        gmp = self.gmp(fi).view(B, C, -1)
        gam = torch.bmm(gap, gmp.permute(0, 2, 1).contiguous())
        g1 = self.softmax(gam)
        g2 = fi.view(B, C, -1)
        out = self.softmax(torch.bmm(g1, g2)).view(B, C, H, W)
        return out+fi


class propagation(nn.Module):
    def __init__(self, in_channel):
        super(propagation, self).__init__()
        self.bn_relu1 = nn.Sequential(nn.BatchNorm2d(in_channel), nn.ReLU())
        self.bn_relu2 = nn.Sequential(nn.BatchNorm2d(in_channel), nn.ReLU())
        self.softmax_conv = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, 1, 1, 0), nn.Softmax(dim=1))

    def forward(self, f1, f2):
        propa1 = self.bn_relu1(f1 * f2 + f1)
        propa2 = self.bn_relu2((f1 + f2) * f2)
        propa_fuse = self.softmax_conv(torch.cat([propa1, propa2], dim=1))
        return propa_fuse


class ccde_w1(nn.Module):
    def __init__(self, in_channel):
        super(ccde_w1, self).__init__()
        self.rela1_Q = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.rela2_K = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.rela3_V = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.convrelu = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, 1), nn.ReLU())

    def forward(self, prim, rela1, rela2, rela3, G):
        query = self.rela1_Q(rela1) * prim
        key = self.rela2_K(rela2) * prim
        value = self.rela3_V(rela3) * prim
        stage1 = self.bn1(query + key + value)
        # print('stage1:', stage1.shape)  # [2, 64, 32, 32]
        gra = G + stage1 * G
        sta = stage1 + stage1 * G
        concate = torch.cat([gra, sta], dim=1)

        gap = self.gap(concate)
        d1 = self.convrelu(concate * gap)
        # print('d1:', d1.shape)  # [2, 256, 32, 32]
        return d1


class ccde_w2(nn.Module):
    def __init__(self, in_channel):
        super(ccde_w2, self).__init__()
        self.rela1_Q = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.rela2_K = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.rela3_V = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        # self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.convrelu = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, 1), nn.ReLU())

    def forward(self, prim, rela1, rela2, rela3, d1, G):
        query = self.rela1_Q(rela1) * prim
        key = self.rela2_K(rela2) * prim
        value = self.rela3_V(rela3) * prim
        stage1 = query + key + value  # delete bn1

        stage2 = self.bn2(d1 * stage1)  # modify + to *

        gra = G + stage2 * G
        sta = stage2 + stage2 * G
        concate = torch.cat([gra, sta], dim=1)

        gap = self.gap(concate)
        d2 = self.convrelu(concate * gap)
        # print('d2:', d2.shape)  # [2, 256, 32, 32]
        return d2


class ccde_w3(nn.Module):
    def __init__(self, in_channel):
        super(ccde_w3, self).__init__()
        self.rela1_Q = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.rela2_K = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.rela3_V = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        # self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.convrelu = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, 1), nn.ReLU())

    def forward(self, prim, rela1, rela2, rela3, d1, d2, G):
        query = self.rela1_Q(rela1) * prim
        key = self.rela2_K(rela2) * prim
        value = self.rela3_V(rela3) * prim
        stage1 = query + key + value  # delete bn1

        stage2 = self.bn2(d1 * stage1 + d2 * stage1)

        gra = G + stage2 * G
        sta = stage2 + stage2 * G
        concate = torch.cat([gra, sta], dim=1)

        gap = self.gap(concate)
        d3 = self.convrelu(concate * gap)
        # print('d3:', d3.shape)  # [2, 256, 32, 32]
        return d3


class ccde_w4(nn.Module):
    def __init__(self, in_channel):
        super(ccde_w4, self).__init__()
        self.rela1_Q = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.rela2_K = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.rela3_V = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        # self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.convrelu = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, 1), nn.ReLU())

    def forward(self, prim, rela1, rela2, rela3, d1, d2, d3, G):
        query = self.rela1_Q(rela1) * prim
        key = self.rela2_K(rela2) * prim
        value = self.rela3_V(rela3) * prim
        stage1 = query + key + value  # delete bn1

        stage2 = self.bn2(d1 * stage1 + d2 * stage1 + d3 *stage1)

        gra = G + stage2 * G
        sta = stage2 + stage2 * G
        concate = torch.cat([gra, sta], dim=1)

        gap = self.gap(concate)
        d4 = self.convrelu(concate * gap)
        # print('d4:', d4.shape)  # [2, 256, 32, 32]
        return d4


class FFNet(nn.Module):
    def __init__(self, ind=50):
        super(FFNet, self).__init__()
        self.layer_rgb = mobilenet_v2(pretrained=True)
        self.layer_dsm = mobilenet_v2(pretrained=True)

        self.rgb_sem1 = SEM(in_channel=24)
        self.rgb_sem2 = SEM(in_channel=32)
        self.rgb_sem3 = SEM(in_channel=160)
        self.rgb_sem4 = SEM(in_channel=320)
        self.dsm_sem1 = SEM(in_channel=24)
        self.dsm_sem2 = SEM(in_channel=32)
        self.dsm_sem3 = SEM(in_channel=160)
        self.dsm_sem4 = SEM(in_channel=320)

        self.fiom11 = FIOM(in_channel=24)
        self.fiom2 = FIOM(in_channel=32)
        self.fiom3 = FIOM(in_channel=160)
        self.fiom4 = FIOM(in_channel=320)

        self.bi1 = nn.Sequential(nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True),
                                 nn.Conv2d(48, 64, kernel_size=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.bi2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(320, 64, kernel_size=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.bi3 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                 nn.Conv2d(640, 64, kernel_size=1),
                                 nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.gate = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Sigmoid())

        self.decoder_w1 = ccde_w1(64)
        self.decoder_w2 = ccde_w2(64)
        self.decoder_w3 = ccde_w3(64)
        self.decoder_w4 = ccde_w4(64)

        self.propagation1 = propagation(in_channel=64)
        self.propagation2 = propagation(in_channel=64)
        self.propagation3 = propagation(in_channel=64)
        self.propagation4 = propagation(in_channel=64)

        self.R1_conv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                     nn.Tanh())
        self.R2_conv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                     nn.Tanh())
        self.gcn = EAGCN(64, 6, (32, 32))

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.predict1 = nn.Conv2d(64, 6, kernel_size=3, padding=1)
        self.predict2 = nn.Conv2d(64, 6, kernel_size=3, padding=1)
        self.predict3 = nn.Conv2d(64, 6, kernel_size=3, padding=1)
        self.predict4 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 6, kernel_size=3, padding=1),
        )

    def forward(self, rgb, dsm):
        # d = torch.chunk(image, 6, dim=1)
        # image = torch.cat([d[0], d[1], d[2]], dim=1)
        # depth = torch.cat([d[3], d[4], d[5]], dim=1)
        rgb_0 = self.layer_rgb.features[0:2](rgb)
        rgb_1 = self.layer_rgb.features[2:4](rgb_0)
        rgb_2 = self.layer_rgb.features[4:7](rgb_1)
        rgb_3 = self.layer_rgb.features[7:17](rgb_2)
        rgb_4 = self.layer_rgb.features[17:18](rgb_3)
        dsm_0 = self.layer_dsm.features[0:2](dsm)
        dsm_1 = self.layer_dsm.features[2:4](dsm_0)
        dsm_2 = self.layer_dsm.features[4:7](dsm_1)
        dsm_3 = self.layer_dsm.features[7:17](dsm_2)
        dsm_4 = self.layer_dsm.features[17:18](dsm_3)

        # print('rgb_0:', rgb_0.shape)
        # print('rgb_1:', rgb_1.shape)
        # print('rgb_2:', rgb_2.shape)
        # print('rgb_3:', rgb_3.shape)
        # print('rgb_4:', rgb_4.shape)
        # print('dsm_0:', dsm_0.shape)
        # print('dsm_1:', dsm_1.shape)
        # print('dsm_2:', dsm_2.shape)
        # print('dsm_3:', dsm_3.shape)
        # print('dsm_4:', dsm_4.shape)
        # rgb_0: torch.Size([2, 16, 128, 128])
        # rgb_1: torch.Size([2, 24, 64, 64])
        # rgb_2: torch.Size([2, 32, 32, 32])
        # rgb_3: torch.Size([2, 160, 16, 16])
        # rgb_4: torch.Size([2, 320, 8, 8])

        f1 = torch.cat((self.rgb_sem1(rgb_1), self.dsm_sem1(dsm_1)), dim=1)
        f2 = torch.cat((self.rgb_sem2(rgb_2), self.dsm_sem2(dsm_2)), dim=1)
        f3 = torch.cat((self.rgb_sem3(rgb_3), self.dsm_sem3(dsm_3)), dim=1)
        f4 = torch.cat((self.rgb_sem4(rgb_4), self.dsm_sem4(dsm_4)), dim=1)

        # print('f1:', f1.shape)
        # print('f2:', f2.shape)
        # print('f3:', f3.shape)
        # print('f4:', f4.shape)
        # f1: torch.Size([2, 48, 64, 64])
        # f2: torch.Size([2, 64, 32, 32])
        # f3: torch.Size([2, 320, 16, 16])
        # f4: torch.Size([2, 640, 8, 8])

        f1_opti = self.fiom1(f1)
        f2_opti = self.fiom2(f2)
        f3_opti = self.fiom3(f3)
        f4_opti = self.fiom4(f4)

        # print('f1_opti:', f1_opti.shape)
        # print('f2_opti:', f2_opti.shape)
        # print('f3_opti:', f3_opti.shape)
        # print('f4_opti:', f4_opti.shape)
        # f1: torch.Size([2, 48, 64, 64])
        # f2: torch.Size([2, 64, 32, 32])
        # f3: torch.Size([2, 320, 16, 16])
        # f4: torch.Size([2, 640, 8, 8])

        f1_add = self.bi1(f1 + f1_opti)
        f2_add = f2 + f2_opti
        f3_add = self.bi2(f3 + f3_opti)
        f4_add = self.bi3(f4 + f4_opti)
        # print('f1_add:', f1_add.shape)
        # print('f2_add:', f2_add.shape)
        # print('f3_add:', f3_add.shape)
        # print('f4_add:', f4_add.shape)
        # f1_add: torch.Size([2, 64, 32, 32])
        # f2_add: torch.Size([2, 64, 32, 32])
        # f3_add: torch.Size([2, 64, 32, 32])
        # f4_add: torch.Size([2, 64, 32, 32])

        # decoder
        f_cat = torch.cat([f1_add, f2_add, f3_add, f4_add], 1)
        f_gate = self.gate(f_cat)  # [2, 64, 32, 32]
        # print('f_gate:', f_gate.shape)
        w1 = f1_add * f_gate
        w2 = f2_add * f_gate
        w3 = f3_add * f_gate
        w4 = f4_add * f_gate
        # print('w1:', w1.shape)  # [2, 64, 32, 32]
        # print('w2:', w2.shape)  # [2, 64, 32, 32]
        # print('w3:', w3.shape)  # [2, 64, 32, 32]
        # print('w4:', w4.shape)  # [2, 64, 32, 32]

        # GCN decoder
        F1 = self.propagation1(f1_add, f2_add)
        F2 = self.propagation2(F1, f3_add)
        F3 = self.propagation4(self.propagation3(f2_add, f3_add), f4_add)
        # print('F1:', F1.shape)  # [2, 64, 32, 32]
        # print('F2:', F2.shape)  # [2, 64, 32, 32]
        # print('F3:', F3.shape)  # [2, 64, 32, 32]

        R1 = self.R1_conv(torch.cat([F1, F2], 1))
        R2 = self.R2_conv(torch.cat([R1, F3], 1))
        # print('R2:', R2.shape)  # [2, 64, 32, 32]
        G = self.gcn(R2)
        # print('G:', G.shape)  # [2, 64, 32, 32]

        d1 = self.decoder_w1(w1, w2, w3, w4, G)
        d2 = self.decoder_w2(w1, w2, w3, w4, d1, G)
        d3 = self.decoder_w3(w1, w2, w3, w4, d1, d2, G)
        d4 = self.decoder_w4(w1, w2, w3, w4, d1, d2, d3, G)
        # print('d1:', d1.shape)  # [2, 64, 32, 32]
        # print('d2:', d2.shape)  # [2, 64, 32, 32]
        # print('d3:', d3.shape)  # [2, 64, 32, 32]
        # print('d4:', d4.shape)  # [2, 64, 32, 32]

        out1 = self.predict1(d1)
        out2 = self.predict2(d2)
        out3 = self.predict3(d3)
        out4 = self.predict4(d4)

        return out1, out2, out3, out4

    # def load_pre(self, pre_model):
    #     self.swin_image.load_state_dict(torch.load(pre_model)['model'], strict=False)
    #     print(f"RGB SwinTransformer loading pre_model ${pre_model}")
    #     self.swin_thermal.load_state_dict(torch.load(pre_model)['model'], strict=False)
    #     print(f"Thermal SwinTransformer loading pre_model ${pre_model}")

    # def initialize(self):
    #     weight_init(self)


if __name__ == "__main__":
    image = torch.randn(2, 3, 256, 256).cuda()
    depth = torch.randn(2, 3, 256, 256).cuda()
    model = FFNet().cuda()
    # from ptflops import get_model_complexity_info
    #
    # flops, params = get_model_complexity_info(model, (6, 224, 224), as_strings=True, print_per_layer_stat=True,
    #                                           verbose=True)
    # print(flops)
    # print(params)
    # model.load_pre(bu
    #     '/home/map/PycharmProjects/pytorch_segementation_Remote_Sensing/toolbox/models/SwinMCNet/swin_base_patch4_window12_384_22k.pth')
    out = model(image, depth)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
