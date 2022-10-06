from torchvision import models
import torch
import torch.nn as nn



class FeatureExtractor(nn.Module):
    '''
    The main modules architecture. Based on a resnet34 backbone with Nearest neighbour upsampling
    layers. Input is a (b, c, h, w) FloatTensor. output is (b, embedding_dim, h, w) FloatTensor
    with the embedded pixels. To get equal input and output size, the input dimensions (h,w) should
    be a multiple of 2^5=32.
    :param embedding_dim - the number of output channels. i.e. the length of the embedded
    pixel vector.
    :param context - boolean, If True, a context layer is added to the model. See ContextModule for more.
    '''
    def __init__(self, embedding_dim, context=False):
        super(FeatureExtractor, self).__init__()
        self.embedding_dim = embedding_dim
        self.context = context
        self.resnet = models.resnet34(True)  # can be resnet34 or 50
        for param in self.resnet.parameters():   # Freeze resnet layers
            param.requires_grad = False
        self.upsample1 = UpsamplingBlock(512, 256, skip=True)
        self.upsample2 = UpsamplingBlock(256, 128, skip=True)
        self.upsample3 = UpsamplingBlock(128, 64, skip=True)
        self.upsample4 = UpsamplingBlock(64, 64, skip=True)
        self.upsample5 = UpsamplingBlock(64, 64, skip=False)
        self.finalConv = nn.Sequential(nn.Conv2d(64, self.embedding_dim, 1, 1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(self.embedding_dim))

        if self.context:
            self.upsample4 = UpsamplingBlock(128, 64, skip=True)
            self.contextLayer = ContextModule(64,32)

    def forward(self, x):
        outputs = {}
        for name, module in list(self.resnet.named_children())[:-2]:
            x = module(x)
            outputs[name] = x
        features = outputs['layer4']  # Resnet output before final avgpool and fc layer
        features = self.upsample1(features, outputs['layer3'])
        features = self.upsample2(features, outputs['layer2'])
        features = self.upsample3(features, outputs['layer1'])

        if self.context:
            context = self.contextLayer(features)
            features = torch.cat([features, context],1)

        features = self.upsample4(features, outputs['relu'])
        features = self.upsample5(features)
        features = self.finalConv(features)

        return features


class UpsamplingBlock(nn.Module):
    '''
    For the up-sampling I chose Nearest neighbour over deconvolution, to avoid artifacts in the output.
    If skip is set to True, the input to the forward pass must include skip input - i.e. the equivalent sized output
    of the downsampling backbone (here resnet).
    :param channels_in - number of filters/channels in the input.
    :param channels_in - number of filters/channels in the output.
    :param skip - whether or not to use skip input. recommended to set to true.
    '''
    def __init__(self, channels_in, channels_out, skip=False):
        super(UpsamplingBlock, self).__init__()
        #self.upsamplingLayer = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=2)
        self.upsamplingLayer = nn.Sequential(nn.Upsample(scale_factor=2),
                                             nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1),
                                             nn.ReLU(),
                                             nn.BatchNorm2d(channels_out))

        if skip:
            self.conv1 = nn.Conv2d(2*channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)

        self.convLayer1 = nn.Sequential(self.conv1,
                                        nn.ReLU(),
                                        nn.BatchNorm2d(channels_out))

        self.convLayer2 = nn.Sequential(self.conv2,
                                        nn.ReLU(),
                                        nn.BatchNorm2d(channels_out))

    def forward(self, x, skip_input=None):
        x = self.upsamplingLayer(x)
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)
        x = self.convLayer1(x)
        x = self.convLayer2(x)
        return x


class ContextModule(nn.Module):
    '''
    this is essentialy a bi-LSTM that process the feature vectors.
    It recieves a (b, c, h, w) tensor and outputs a tensor
    of the same size after the rnn pass.
    :param input_size - number of channels in the input.
    :param hidden_size - dimension of the LSTM hidden layers.
    '''

    def __init__(self, input_size, hidden_size):
        super(ContextModule, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True,
                            bidirectional=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        bs, h, w, f = x.size()
        x = x.view(bs, h * w, f)
        x, _ = self.lstm(x)
        x = x.contiguous().view(bs, h, w, 2 * self.hidden_size)
        x = x.permute(0, 3, 1, 2)
        return x
