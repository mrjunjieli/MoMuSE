import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from apex import amp
import copy
from MuSE.pretrain_networks.visual_frontend import VisualFrontend
import pdb 

EPS = 1e-8

def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class muse(nn.Module):
    def __init__(self, N, L, B, H, P, X, R, C, M,causal=False):
        super(muse, self).__init__()
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = N, L, B, H, P, X, R, C
        
        self.encoder = Encoder(L, N)
        self.separator = TemporalConvNet(N, B, H, P, X, R, C, M)
        self.decoder = Decoder(N, L)
        self.visual_frontend = VisualFrontend()
        pretrained_model = torch.load('./MuSE/pretrain_networks/visual_frontend.pt', map_location='cpu')
        state = self.visual_frontend.state_dict()
        for key in state.keys():
            pretrain_key = key
            if pretrain_key in pretrained_model.keys():
                state[key] = pretrained_model[pretrain_key]
            elif 'module.'+pretrain_key in pretrained_model.keys():
                state[key] = pretrained_model['module.'+pretrain_key]
            else:
                print(key +' is not loaded!!') 
        self.visual_frontend.load_state_dict(state)
        for key,param in self.visual_frontend.named_parameters():
            param.requires_grad = False

    def forward(self, mixture, visual):
        visual = visual.transpose(0,1)
        visual = visual.unsqueeze(2)
        with torch.no_grad():
            visual = self.visual_frontend(visual)
        visual = visual.transpose(0,1)

        mixture_w = self.encoder(mixture)
        est_a_emb, est_mask = self.separator(mixture_w, visual)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_a_emb, est_source

class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w

class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        est_source = mixture_w * est_mask  # [M,  N, K]
        est_source = torch.transpose(est_source, 2, 1) # [M,  K, N]
        est_source = self.basis_signals(est_source)  # [M,  K, L]
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T
        return est_source


class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, M,causal=False):
        super(TemporalConvNet, self).__init__()
        self.C = C
        self.layer_norm = ChannelWiseLayerNorm(N)
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        self.causal = causal 

        # Audio TCN
        tcn_blocks = []
        tcn_blocks += [nn.Conv1d(B*3, B, 1, bias=False)]
        for x in range(X):
            dilation = 2**x
            padding = (P - 1) * dilation // 2 if not causal else (P - 1) * dilation
            tcn_blocks += [TemporalBlock(B, H, P, stride=1,
                                     padding=padding,
                                     dilation=dilation,causal=causal)]
        self.tcn = _clones(nn.Sequential(*tcn_blocks), R)
        
        # visual blocks
        ve_blocks = []
        for x in range(5):
            ve_blocks +=[VisualConv1D()]
        self.visual_conv = nn.Sequential(*ve_blocks)

        # Audio and visual seprated layers before concatenation
        self.ve_conv1x1 = _clones(nn.Conv1d(512, B, 1, bias=False),R)
        self.ve_conv1x1_SE = _clones(nn.Conv1d(512, B, 1, bias=False),R)

        # speaker embedding extraction and classification
        self.se_net=_clones(SpeakerEmbedding(B), R)
        self.audio_linear=_clones(nn.Linear(B, M),R)

        # Mask generation layer
        self.mask_conv1x1 = nn.Conv1d(B, N, 1, bias=False)


    def forward(self, x, visual):
        visual = visual.transpose(1,2)
        visual = self.visual_conv(visual)

        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)

        mixture = x

        batch, B, K = x.size()

        est_a_emb=[]

        for i in range(len(self.tcn)):
            v = self.ve_conv1x1[i](visual)
            v = F.interpolate(v, (32*v.size()[-1]), mode='linear')
            v = F.pad(v,(0,K-v.size()[-1]))
            v_2 = self.ve_conv1x1_SE[i](visual)
            v_2 = F.interpolate(v_2, (32*v_2.size()[-1]), mode='linear')
            v_2 = F.pad(v_2,(0,K-v_2.size()[-1]))
            a = mixture*F.relu(x)
            a = self.se_net[i](torch.cat((a,v_2),1))
            est_a_emb.append(self.audio_linear[i](a.squeeze()))
            a = torch.repeat_interleave(a, repeats=K, dim=2)
            x = torch.cat((a, x, v),1)
            x = self.tcn[i](x)
            
        x = self.mask_conv1x1(x)
        x = F.relu(x)
        est_a_emb = torch.stack(est_a_emb)
        return est_a_emb, x

class SpeakerEmbedding(nn.Module):
    def __init__(self, B, R=3, H=256):
        super(SpeakerEmbedding, self).__init__()
        self.conv_proj = nn.Conv1d(B*2, B, 1, bias=False)
        Conv_1=nn.Conv1d(B, H, 1, bias=False)
        norm_1=nn.BatchNorm1d(H)
        prelu_1=nn.PReLU()
        Conv_2=nn.Conv1d(H, B, 1, bias=False)
        norm_2=nn.BatchNorm1d(B)
        self.resnet=_clones(nn.Sequential(Conv_1, norm_1,\
            prelu_1, Conv_2, norm_2), R)
        self.prelu=_clones(nn.PReLU(),R)
        self.maxPool=_clones(nn.AvgPool1d(3),R)

        self.conv=nn.Conv1d(B,B,1)
        self.avgPool=nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv_proj(x)
        for i in range(len(self.resnet)):
            residual = x
            x = self.resnet[i](x)
            x = self.prelu[i](x+residual)
            x = self.maxPool[i](x)

        x = self.conv(x)
        x = self.avgPool(x)

        return x



class VisualConv1D(nn.Module):
    def __init__(self):
        super(VisualConv1D, self).__init__()
        relu = nn.ReLU()
        norm_1 = nn.BatchNorm1d(512)
        dsconv = nn.Conv1d(512, 512, 3, stride=1, padding=1,dilation=1, groups=512, bias=False)
        prelu = nn.PReLU()
        norm_2 = nn.BatchNorm1d(512)
        pw_conv = nn.Conv1d(512, 512, 1, bias=False)

        self.net = nn.Sequential(relu, norm_1 ,dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation,causal=False):
        super(TemporalBlock, self).__init__()
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = GlobalLayerNorm(out_channels)
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation,causal=causal)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)


    def forward(self, x):
        residual = x
        out = self.net(x)
        return out + residual  # look like w/o F.relu is better than w/ F.relu


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation,causal=False):
        super(DepthwiseSeparableConv, self).__init__()
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)

        prelu = nn.PReLU()
        norm = GlobalLayerNorm(in_channels)
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.net = nn.Sequential(depthwise_conv, prelu, norm,
                                 pointwise_conv)

    def forward(self, x):
        return self.net(x)

class ChannelWiseLayerNorm(nn.LayerNorm):
    @amp.float_function
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    @amp.float_function
    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    @amp.float_function
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    @amp.float_function
    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    @amp.float_function
    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y

@amp.float_function
def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long().cuda()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result
