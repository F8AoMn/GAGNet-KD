import torch
import torch.nn as nn
class GCN(nn.Module):
    def __init__(self, num_state, num_node): #num_in: 41 num_node: HW
        super(GCN, self).__init__()
        self.num_state = num_state
        self.num_node = num_node
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1)

    def forward(self, seg, aj): # seg: n, 41, h, w, aj: hw, hw
        n, c, h, w = seg.size()
        seg = seg.view(n, self.num_state, -1).contiguous()
        seg_similar = torch.bmm(seg, aj)
        out = self.relu(self.conv2(seg_similar))
        output = out + seg

        return output


class EAGCN(nn.Module):
    def __init__(self, num_in, plane_mid, mids, normalize=False):
        """

        :param num_in: input channel
        :param plane_mid: 1
        :param mids: input size h , w
        :param normalize:
        """
        super(EAGCN, self).__init__()
        self.num_in = num_in
        self.mids = mids
        self.normalzie = normalize
        self.num_s = int(plane_mid)
        self.pool_avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.pool_max = nn.AdaptiveMaxPool2d(output_size=1)
        self.tan = nn.Tanh()
        self.conv_c1 = nn.Conv2d(num_in * 2, self.num_s, kernel_size=1)
        self.conv_s1 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s2 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        self.gcn = GCN(num_state=num_in, num_node=self.num_in)

    def forward(self, seg_ori):
        seg = seg_ori
        n, c, h, w = seg.size()  # [2, 64, 32, 32]
        th = torch.cat([self.pool_avg(seg), self.pool_max(seg)], dim=1)
        # print('self.tan(th):', self.tan(th).shape)  # [2, 128, 1, 1]
        theta = self.conv_c1(self.tan(th)).squeeze(3)
        theta_T = theta.view(n, -1, self.num_s)
        # print('theta_T:', theta_T.shape)  # [2, 1, 6]
        channel_diag = torch.bmm(theta, theta_T)

        spaital1 = self.conv_s1(seg).view(n, self.num_s, -1)
        spaital2 = self.conv_s2(seg).view(n, -1, self.num_s)

        atten1 = torch.bmm(channel_diag, spaital1)
        atten2 = torch.bmm(spaital2, channel_diag)
        # print('spaital1:', spaital1.shape)  # [2, 6, 1024]
        # print('spaital2:', spaital2.shape)  # [2, 1024, 6]
        # print('atten1:', atten1.shape)  # [2, 6, 1024]
        # print('atten2:', atten2.shape)  # [2, 1024, 6]

        adj_att = torch.bmm(atten2, atten1)
        adj_s = self.softmax(adj_att)

        seg_gcn = self.gcn(seg, adj_s).view(n, self.num_in, self.mids[0], self.mids[1])

        ext_up_seg_gcn = seg_gcn + seg_ori
        return ext_up_seg_gcn


if __name__ == '__main__':
    x = torch.randn(2, 32, 32, 32)
    net = EAGCN(32, 6, (32, 32))
    out = net(x)
    print(out.shape)
