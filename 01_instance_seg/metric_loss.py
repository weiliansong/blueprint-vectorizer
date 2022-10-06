# https://github.com/alicranck/instance-seg/blob/master/loss.py
import numpy as np
import torch
from torch.autograd import Variable
from utils.config import Struct, load_config, compose_config_str
from PIL import Image
from torchvision import transforms

config_dict = load_config(file_path='./utils/config.yaml')
configs = Struct(**config_dict)

# Loss parameters
dv = configs.dv
dd = configs.dd
gamma = configs.gamma


class metricLoss(torch.nn.Module):
    '''
    A class designed purely to run a givan loss on a batch of samples.
    input is a batch of samples (as autograd Variables) and a batch of labels (ndarrays),
    output is the average loss (as autograd variable).
    '''
    def __init__(self):
        super(metricLoss, self).__init__()
        self.loss = contrasive_loss

    def forward(self, features_batch, labels_batch):
        running_loss = Variable(torch.Tensor([0]).type(torch.FloatTensor))
        batch_size = features_batch.size()[0]
        for i in range(batch_size):
            running_loss += self.loss(features_batch[i], labels_batch[i])

        ret = running_loss/(batch_size+1)
        return ret


def contrasive_loss(features, label):
    '''
    This loss is taken from "Semantic Instance Segmentation with a Discriminative Loss Function"
    by Bert De Brabandere, Davy Neven, Luc Van Gool at https://arxiv.org/abs/1708.02551
    :param features: a FloatTensor of (embedding_dim, h, w) dimensions.
    :param label: an nd-array of size (h, w) with ground truth instance segmentation. background is
                    assumed to be 0.
    :return: The loss calculated as described in the paper.
    '''
    label = label.flatten()
    features = features.permute(1,2,0).contiguous()
    shape = features.size()
    features = features.view(shape[0]*shape[1], shape[2])

    instances = torch.unique(label)

    means = []
    var_loss = Variable(torch.Tensor([0]).type(torch.DoubleTensor))
    dist_loss = Variable(torch.Tensor([0]).type(torch.DoubleTensor))

    # calculate intra-cluster loss
    for instance in instances:
        # collect all feature vector of a certain instance
        locations = Variable(torch.LongTensor(torch.where(label == instance)[0]).type(torch.LongTensor))
        vectors = torch.index_select(features,dim=0,index=locations).type(torch.DoubleTensor)
        size = vectors.size()[0]

        ## get instance mean and distances to mean of all points in an instance
        # if instance == 0:  # Ignore background
        #    continue
        # else:

        mean = torch.sum(vectors, dim=0) / size
        dists = vectors - mean
        dist2mean = torch.sum(dists**2,1)

        var_loss += torch.sum(dist2mean)/size
        means.append(mean)

    # get inter-cluster loss - penalize close cluster centers
    if len(means)==0: # no instances in image
        return Variable(torch.Tensor([0]).type(torch.FloatTensor), requires_grad=True)

    means = torch.stack(means)
    num_clusters = means.data.shape[0]

    for i in range(num_clusters):
        if num_clusters==1:  # no inter cluster loss
            break
        for j in range(i+1, num_clusters):
            dist = torch.norm(means[i]-means[j])
            if dist < dd*2:
                dist_loss += torch.pow(2*dd - dist,2)/(num_clusters-1)

    # regularization term
    reg_loss = torch.sum(torch.norm(means, 2, 1))

    total_loss = (var_loss + dist_loss + gamma*reg_loss) / num_clusters

    return total_loss.type(torch.FloatTensor)
