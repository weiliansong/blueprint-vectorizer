import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.utils.spectral_norm as spectral_norm

from torch.autograd import Variable


def add_pool(x, nd_to_sample):
    dtype, device = x.dtype, x.device
    batch_size = torch.max(nd_to_sample) + 1
    pooled_x = torch.zeros(batch_size, *x.shape[1:]).float().to(device)
    pool_to = nd_to_sample.view(-1, 1, 1, 1).expand_as(x).to(device)
    pooled_x = pooled_x.scatter_add(0, pool_to, x)
    return pooled_x

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def compute_gradient_penalty(D, x, x_fake, given_y=None, given_i=None, given_w=None, \
                             nd_to_sample=None, data_parallel=None, \
                             ed_to_sample=None, topo_vecs=None, given_b=None):
    indices = nd_to_sample, ed_to_sample
    batch_size = torch.max(nd_to_sample) + 1
    dtype, device = x.dtype, x.device
    u = torch.FloatTensor(x.shape[0], 1, 1).to(device)
    u.data.resize_(x.shape[0], 1, 1)
    u.uniform_(0, 1)
    x_both = x.data*u + x_fake.data*(1-u)
    x_both = x_both.to(device)
    x_both = Variable(x_both, requires_grad=True)
    grad_outputs = torch.ones(batch_size, 1).to(device)
    if data_parallel:
        _output = data_parallel(D, (x_both, given_y, given_w, nd_to_sample), indices)
    else:
        _output, _ = D(x_both, given_i, given_y, given_w, nd_to_sample, topo_vecs=topo_vecs, given_b=given_b)
    grad = torch.autograd.grad(outputs=_output, inputs=x_both, grad_outputs=grad_outputs, \
                               retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradient_penalty = ((grad.norm(2, 1).norm(2, 1) - 1) ** 2).mean()
    return gradient_penalty

def conv_block(in_channels, out_channels, k, s, p, act=None, upsample=False, spec_norm=False, batch_norm=True):
    block = []

    if upsample:
        if spec_norm:
            block.append(spectral_norm(torch.nn.ConvTranspose2d(in_channels, out_channels, \
                                                   kernel_size=k, stride=s, \
                                                   padding=p, bias=True)))
        else:
            block.append(torch.nn.ConvTranspose2d(in_channels, out_channels, \
                                                   kernel_size=k, stride=s, \
                                                   padding=p, bias=True))
    else:
        if spec_norm:
            block.append(spectral_norm(torch.nn.Conv2d(in_channels, out_channels, \
                                                       kernel_size=k, stride=s, \
                                                       padding=p, bias=True)))
        else:
            block.append(torch.nn.Conv2d(in_channels, out_channels, \
                                                       kernel_size=k, stride=s, \
                                                       padding=p, bias=True))
    if batch_norm:
        # block.append(nn.InstanceNorm2d(out_channels))
        block.append(nn.GroupNorm(1, out_channels))
    if "leaky" in act:
        block.append(torch.nn.LeakyReLU(0.1, inplace=True))
    elif "relu" in act:
        block.append(torch.nn.ReLU(inplace=True))
    elif "tanh":
        block.append(torch.nn.Tanh())
    return block

class CMP(nn.Module):
    def __init__(self, in_channels, out_channels, \
                 extra_channels=0, batch_norm=True):
        super(CMP, self).__init__()

        self.encoder = nn.Sequential(
            *conv_block(2*in_channels+extra_channels, 2*in_channels, 3, 1, 1, act="leaky", batch_norm=batch_norm),
            *conv_block(2*in_channels, in_channels, 3, 1, 1, act="leaky", batch_norm=batch_norm),
            *conv_block(in_channels, out_channels, 3, 1, 1, act="leaky", batch_norm=batch_norm))

        # self.encoder = UNet(in_channels*2, in_channels)

    def forward(self, feats, edges=None, extra_feats=None):

        # allocate memory
        dtype, device = feats.dtype, feats.device
        edges = edges.view(-1, 3)
        V, E = feats.size(0), edges.size(0)
        pooled_v_pos = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        # pooled_v_neg = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)

        neg_inds = torch.where(edges[:, 1] < 0)
        assert not len(neg_inds[0])

        # pool positive edges
        pos_inds = torch.where(edges[:, 1] > 0)
        pos_v_src = torch.cat([edges[pos_inds[0], 0], edges[pos_inds[0], 2]]).long()
        pos_v_dst = torch.cat([edges[pos_inds[0], 2], edges[pos_inds[0], 0]]).long()
        pos_vecs_src = feats[pos_v_src.contiguous()]
        pos_v_dst = pos_v_dst.view(-1, 1, 1, 1).expand_as(pos_vecs_src).to(device)
        pooled_v_pos = pooled_v_pos.scatter_add(0, pos_v_dst, pos_vecs_src)

        # pool negative edges
        # neg_v_src = torch.cat([edges[neg_inds[0], 0], edges[neg_inds[0], 2]]).long()
        # neg_v_dst = torch.cat([edges[neg_inds[0], 2], edges[neg_inds[0], 0]]).long()
        # neg_vecs_src = feats[neg_v_src.contiguous()]
        # neg_v_dst = neg_v_dst.view(-1, 1, 1, 1).expand_as(neg_vecs_src).to(device)
        # pooled_v_neg = pooled_v_neg.scatter_add(0, neg_v_dst, neg_vecs_src)

        pooled_v_pos = pooled_v_pos / max(len(pos_inds[0]), 1)
        # pooled_v_neg = pooled_v_neg / max(len(neg_inds[0]), 1)

        # update nodes features
        enc_in = torch.cat([feats, pooled_v_pos, extra_feats], 1)
        out = self.encoder(enc_in)

        return out


class Up(nn.Module):

  def __init__(self, in_channels, out_channels, extra_channels, shortcut):
    super(Up, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.extra_channels = extra_channels
    self.shortcut = shortcut

    self.cmp = CMP(in_channels=in_channels,
                   out_channels=out_channels,
                   extra_channels=extra_channels)
    self.up = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1, bias=False)

    if shortcut:
      self.conv = nn.Sequential(*conv_block(out_channels * 2,
                                            out_channels,
                                            3, 1, 1, act="leaky"))

  def forward(self, low_res_mask, high_res_mask, edges, extra_feat):
    x = self.cmp(low_res_mask, edges, extra_feat)
    x = self.up(x)

    if self.shortcut:
      x = torch.cat([x, high_res_mask], axis=1)
      x = self.conv(x)

    return x


class Generator(nn.Module):
    def __init__(self, image_input, topo_input, ind_input):
        super(Generator, self).__init__()

        self.init_size = 8
        self.image_input = image_input
        self.topo_input = topo_input
        assert image_input

        if topo_input:
          self.l1 = nn.Sequential(nn.Linear(168, 64))
        else:
          self.l1 = nn.Sequential(nn.Linear(136, 32))

        # an indicator mask might be concatenated
        if ind_input:
          self.enc_1 = nn.Sequential(
              *conv_block(2, 16, 3, 2, 1, act="leaky"),
              *conv_block(16, 32, 3, 2, 1, act="leaky"),
              *conv_block(32, 64, 3, 2, 1, act="leaky"))
        else:
          self.enc_1 = nn.Sequential(
              *conv_block(1, 16, 3, 2, 1, act="leaky"),
              *conv_block(16, 32, 3, 2, 1, act="leaky"),
              *conv_block(32, 64, 3, 2, 1, act="leaky"))

        if topo_input:
          self.enc_2 = nn.Sequential(
              *conv_block(128, 64, 3, 1, 1, act="leaky"),
              *conv_block(64, 64, 3, 1, 1, act="leaky"))
        else:
          self.enc_2 = nn.Sequential(
              *conv_block(96, 64, 3, 1, 1, act="leaky"),
              *conv_block(64, 64, 3, 1, 1, act="leaky"))

        self.upsample_1 = nn.Sequential(*conv_block(64, 64, 4, 2, 1, act="leaky", upsample=True))
        self.upsample_2 = nn.Sequential(*conv_block(64, 64, 4, 2, 1, act="leaky", upsample=True))
        self.upsample_3 = nn.Sequential(*conv_block(64, 64, 4, 2, 1, act="leaky", upsample=True))

        if image_input:
          self.enc_same = nn.Sequential(
              *conv_block(1, 16, 3, 1, 1, act="leaky"),
              *conv_block(16, 16, 3, 1, 1, act="leaky"))

          self.enc_down1 = nn.Sequential(
              *conv_block(16, 16, 3, 2, 1, act="leaky"),
              *conv_block(16, 16, 3, 1, 1, act="leaky"))

          self.enc_down2 = nn.Sequential(
              *conv_block(16, 16, 3, 2, 1, act="leaky"),
              *conv_block(16, 16, 3, 1, 1, act="leaky"))

          self.enc_down3 = nn.Sequential(
              *conv_block(16, 16, 3, 2, 1, act="leaky"),
              *conv_block(16, 16, 3, 1, 1, act="leaky"))

          self.cmp_1 = CMP(n_updown=3, in_channels=64, extra_channels=16)
          self.cmp_2 = CMP(n_updown=4, in_channels=64, extra_channels=16)
          self.cmp_3 = CMP(n_updown=5, in_channels=64, extra_channels=16)
          self.cmp_4 = CMP(n_updown=5, in_channels=64, extra_channels=16)

          self.decoder = nn.Sequential(
              *conv_block(80, 256, 3, 1, 1, act="leaky"),
              *conv_block(256, 128, 3, 1, 1, act="leaky"),
              *conv_block(128, 1, 3, 1, 1, act='none', batch_norm=False))

        else:
          self.cmp_1 = CMP(n_updown=3, in_channels=64)
          self.cmp_2 = CMP(n_updown=4, in_channels=64)
          self.cmp_3 = CMP(n_updown=5, in_channels=64)
          self.cmp_4 = CMP(n_updown=5, in_channels=64)

          self.decoder = nn.Sequential(
              *conv_block(64, 256, 3, 1, 1, act="leaky"),
              *conv_block(256, 128, 3, 1, 1, act="leaky"),
              *conv_block(128, 1, 3, 1, 1, act='none', batch_norm=False))


    def forward(self, z, given_masks, semantics, edges, topo_vecs, image=None):
        # z = z.view(-1, 128)
        # semantics = semantics.view(-1, 8) # num_classes
        # topo_vecs = topo_vecs.view(-1, 32)

        if self.topo_input:
          x = torch.cat([z, semantics, topo_vecs], 1)
        else:
          x = torch.cat([z, semantics], 1)

        x = self.l1(x)
        x = x.unsqueeze(2).unsqueeze(2)
        x = x.repeat([1, 1, self.init_size, self.init_size])

        hidden = {}

        m = self.enc_1(given_masks)
        hidden['enc_1'] = m.data

        x = torch.cat([x, m], 1)
        x = self.enc_2(x)
        hidden['enc_2'] = x.data

        if self.image_input:
          image = image.unsqueeze(0).unsqueeze(0)
          image = image.repeat(len(x), 1, 1, 1)
          img_64 = self.enc_same(image)
          img_32 = self.enc_down1(img_64)
          img_16 = self.enc_down2(img_32)
          img_8 = self.enc_down3(img_16)

        # lvl 1
        if self.image_input:
          x = self.cmp_1(x, edges, img_8)
        else:
          x = self.cmp_1(x, edges)
        hidden['cmp_1'] = x.data

        x = self.upsample_1(x)
        hidden['up_1'] = x.data

        # lvl 2
        if self.image_input:
          x = self.cmp_2(x, edges, img_16)
        else:
          x = self.cmp_2(x, edges)
        hidden['cmp_2'] = x.data

        x = self.upsample_2(x)
        hidden['up_2'] = x.data

        # lvl 3
        if self.image_input:
          x = self.cmp_3(x, edges, img_32)
        else:
          x = self.cmp_3(x, edges)
        hidden['cmp_3'] = x.data

        x = self.upsample_3(x)
        hidden['up_3'] = x.data

        # lvl 4
        if self.image_input:
          x = self.cmp_4(x, edges, img_64)
        else:
          x = self.cmp_4(x, edges)
        hidden['cmp_4'] = x.data

        # top lvl
        if self.image_input:
          x = torch.cat([x, img_64], axis=1)
        x = self.decoder(x)
        hidden['decoder'] = x.data

        x = x.view(-1, *x.shape[2:])

        return x, hidden

class Discriminator(nn.Module):
    def __init__(self, image_input, topo_input):
        super(Discriminator, self).__init__()
        self.image_input = image_input
        self.topo_input = topo_input
        self.init_size = 64
        assert image_input

        if topo_input:
          self.l1 = nn.Sequential(nn.Linear(40, 32))
          self.encoder = nn.Sequential(
              *conv_block(33, 16, 3, 1, 1, act="leaky", batch_norm=False),
              *conv_block(16, 16, 3, 1, 1, act="leaky", batch_norm=True),
              *conv_block(16, 16, 3, 1, 1, act="leaky", batch_norm=True),
              *conv_block(16, 16, 3, 1, 1, act="leaky", batch_norm=True))

        else:
          self.l1 = nn.Sequential(nn.Linear(8, 8))
          self.encoder = nn.Sequential(
              *conv_block(9, 16, 3, 1, 1, act="leaky", batch_norm=False),
              *conv_block(16, 16, 3, 1, 1, act="leaky", batch_norm=True),
              *conv_block(16, 16, 3, 1, 1, act="leaky", batch_norm=True),
              *conv_block(16, 16, 3, 1, 1, act="leaky", batch_norm=True))

        self.cmp_down_1 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky", batch_norm=True))
        self.cmp_down_2 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky", batch_norm=True))
        self.cmp_down_3 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky", batch_norm=True))

        self.enc_same = nn.Sequential(
            *conv_block(1, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"))

        self.enc_down1 = nn.Sequential(
            *conv_block(16, 16, 3, 2, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"))

        self.enc_down2 = nn.Sequential(
            *conv_block(16, 16, 3, 2, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"))

        self.enc_down3 = nn.Sequential(
            *conv_block(16, 16, 3, 2, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"))

        self.cmp_1 = CMP(in_channels=16, out_channels=16, extra_channels=16, batch_norm=True)
        self.cmp_2 = CMP(in_channels=16, out_channels=16, extra_channels=16, batch_norm=True)
        self.cmp_3 = CMP(in_channels=16, out_channels=16, extra_channels=16, batch_norm=True)

        self.global_decoder = nn.Sequential(
            *conv_block(16+16, 64, 3, 2, 1, act="leaky", batch_norm=True),
            *conv_block(64, 128, 3, 2, 1, act="leaky", batch_norm=True),
            *conv_block(128, 128, 3, 2, 1, act="leaky", batch_norm=False))
        self.fc_layer_global = nn.Sequential(nn.Linear(128, 1))

        self.local_decoder = nn.Sequential(
            *conv_block(16+16, 64, 3, 2, 1, act="leaky", batch_norm=True),
            *conv_block(64, 128, 3, 2, 1, act="leaky", batch_norm=True),
            *conv_block(128, 128, 3, 2, 1, act="leaky", batch_norm=False))
        self.fc_layer_local = nn.Sequential(nn.Linear(128, 1))


    def forward(self, x, given_i=None, given_y=None, given_w=None, nd_to_sample=None, topo_vecs=None, given_b=None):
        if self.topo_input:
          y = torch.cat([given_y, topo_vecs], 1)
        else:
          y = given_y

        y = self.l1(y)
        y = y.unsqueeze(2).unsqueeze(2)
        y = y.repeat([1, 1, self.init_size, self.init_size])

        x = x.view(-1, 1, 64, 64)
        x = torch.cat([x, y], 1)

        hidden = {}

        x = self.encoder(x)
        hidden['encoder'] = x.data

        given_b = given_b.unsqueeze(0).unsqueeze(0)
        given_b = given_b.repeat(len(x), 1, 1, 1)
        img_64 = self.enc_same(given_b)
        img_32 = self.enc_down1(img_64)
        img_16 = self.enc_down2(img_32)
        img_8 = self.enc_down3(img_16)

        x = self.cmp_1(x, given_w, img_64)
        hidden['cmp_1'] = x.data

        x = self.cmp_down_1(x)
        hidden['cmp_down_1'] = x.data

        x = self.cmp_2(x, given_w, img_32)
        hidden['cmp_2'] = x.data

        x = self.cmp_down_2(x)
        hidden['cmp_down_2'] = x.data

        x = self.cmp_3(x, given_w, img_16)
        hidden['cmp_3'] = x.data

        x = self.cmp_down_3(x)
        hidden['cmp_down_3'] = x.data

        x = torch.cat([x, img_8], axis=1)

        # global loss
        x_g = add_pool(x, nd_to_sample)
        x_g = self.global_decoder(x_g)
        x_g = x_g.view(-1, x_g.shape[1])
        validity_global = self.fc_layer_global(x_g)

        # local loss
        if True:
            x_l = self.local_decoder(x)
            x_l = add_pool(x_l, nd_to_sample)
            x_l = x_l.view(-1, x_l.shape[1])
            validity_local = self.fc_layer_local(x_l)

            # print('(gl: {}) (lc: {})'.format(validity_global, validity_local))
            validity = validity_global + validity_local
            return validity, hidden
        else:
            return validity_global, hidden

class ResGen(nn.Module):

  def __init__(self, hparams):
    super(ResGen, self).__init__()

    self.init_size = 8
    self.hparams = hparams

    self.l1 = nn.Sequential(nn.Linear(136, 32))

    # initial projection of mask and image
    self.mask_proj = nn.Sequential(*conv_block(1, 16, 3, 1, 1, act="leaky"))
    self.img_proj = nn.Sequential(*conv_block(1, 16, 3, 1, 1, act="leaky"))

    # mask downsamplers
    self.mask_down1 = nn.Sequential(*conv_block(16, 32, 3, 2, 1, act="leaky"))
    self.mask_down2 = nn.Sequential(*conv_block(32, 64, 3, 2, 1, act="leaky"))
    self.mask_down3 = nn.Sequential(*conv_block(64, 128, 3, 2, 1, act="leaky"))

    # image downsamplers
    self.img_down1 = nn.Sequential(*conv_block(16, 32, 3, 2, 1, act="leaky"))
    self.img_down2 = nn.Sequential(*conv_block(32, 64, 3, 2, 1, act="leaky"))
    self.img_down3 = nn.Sequential(*conv_block(64, 128, 3, 2, 1, act="leaky"))

    # layer to incorporate the noise vector
    self.add_noise = nn.Sequential(*conv_block(256, 128, 3, 1, 1, act="leaky"))

    # upsample modules, which contain MP modules
    self.up1 = Up(128, 64, 128+8+1, hparams['shortcut_16'])
    self.up2 = Up(64, 32, 64+8+1, hparams['shortcut_32'])
    self.up3 = Up(32, 16, 32+8+1, hparams['shortcut_64'])

    # final MP module
    self.final_mp = CMP(in_channels=16, out_channels=16, extra_channels=16+8+1)

    # final decoder
    self.decoder = nn.Sequential(
        *conv_block(16, 256, 3, 1, 1, act="leaky"),
        *conv_block(256, 128, 3, 1, 1, act="leaky"),
        *conv_block(128, 1, 3, 1, 1, act='none', batch_norm=False))


  def forward(self, z, given_masks, ind_masks, semantics, edges, topo_vecs, image=None):
    # z = z.view(-1, 128)
    # semantics = semantics.view(-1, 8) # num_classes
    # topo_vecs = topo_vecs.view(-1, 32)

    z = z.unsqueeze(2).unsqueeze(2)
    semantics = semantics.unsqueeze(2).unsqueeze(2)
    ind_masks = ind_masks.unsqueeze(2).unsqueeze(2)

    # mask down
    mask_64 = self.mask_proj(given_masks)
    mask_32 = self.mask_down1(mask_64)
    mask_16 = self.mask_down2(mask_32)
    mask_8 = self.mask_down3(mask_16)

    # image down
    image = image.unsqueeze(0).unsqueeze(0)
    image = image.repeat(len(given_masks), 1, 1, 1)
    img_64 = self.img_proj(image)
    img_32 = self.img_down1(img_64)
    img_16 = self.img_down2(img_32)
    img_8 = self.img_down3(img_16)

    # add noise
    x = torch.cat([mask_8, z.repeat([1, 1, 8, 8])], dim=1)
    x = self.add_noise(x)

    # up
    extra_feat = torch.cat([img_8,
                            semantics.repeat([1, 1, 8, 8]),
                            ind_masks.repeat([1, 1, 8, 8])], dim=1)
    x = self.up1(low_res_mask=x, high_res_mask=mask_16,
                 edges=edges, extra_feat=extra_feat)

    extra_feat = torch.cat([img_16,
                            semantics.repeat([1, 1, 16, 16]),
                            ind_masks.repeat([1, 1, 16, 16])], dim=1)
    x = self.up2(low_res_mask=x, high_res_mask=mask_32,
                 edges=edges, extra_feat=extra_feat)

    extra_feat = torch.cat([img_32,
                            semantics.repeat([1, 1, 32, 32]),
                            ind_masks.repeat([1, 1, 32, 32])], dim=1)
    x = self.up3(low_res_mask=x, high_res_mask=mask_64,
                 edges=edges, extra_feat=extra_feat)

    # final MP
    extra_feat = torch.cat([img_64,
                            semantics.repeat([1, 1, 64, 64]),
                            ind_masks.repeat([1, 1, 64, 64])], dim=1)
    x = self.final_mp(x, edges, extra_feat)

    # final decoding
    x = self.decoder(x)

    return x.squeeze(), None
