import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from nemo.models.unet import unet_res50
from nemo.models.upsampling_layer import DoubleConv
from nemo.models.upsampling_layer import Up

try: 
    from VoGE.Sampler import sample_features
    from VoGE.Renderer import Fragments
    enable_voge = True
except:
    enable_voge = False

vgg_layers = {"pool4": 24, "pool5": 31}
net_stride = {
    "vgg_pool4": 16,
    "vgg_pool5": 32,
    "resnet50": 32,
    "resnext50": 32,
    "resnetext": 8,
    "resnetupsample": 8,
    "resnetext2": 4,
    "resnetext3": 4,
    "unet_res50": 1,
}
net_out_dimension = {
    "vgg_pool4": 512,
    "vgg_pool5": 512,
    "resnet50": 2048,
    "resnext50": 2048,
    "resnetext": 256,
    "resnetupsample": 2048,
    "resnetext2": 256,
    "resnetext3": 256,
    "unet_res50": 64,
}


class ResnetUpSample(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        net = models.resnet50(pretrained=pretrained)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.extractor = nn.Sequential()
        self.extractor.add_module("0", net.conv1)
        self.extractor.add_module("1", net.bn1)
        self.extractor.add_module("2", net.relu)
        self.extractor.add_module("3", net.maxpool)
        self.extractor.add_module("4", net.layer1)
        self.extractor.add_module("5", net.layer2)
        self.extractor.add_module("6", net.layer3)
        self.extractor.add_module("7", net.layer4)

    def forward(self, x):
        x = self.extractor(x)
        return self.upsample(x)


class ResNetExt2(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        net = models.resnet50(pretrained=pretrained)
        self.extractor = nn.Sequential()
        self.extractor.add_module("0", net.conv1)
        self.extractor.add_module("1", net.bn1)
        self.extractor.add_module("2", net.relu)
        self.extractor.add_module("3", net.maxpool)
        self.extractor.add_module("4", net.layer1)
        self.extractor0 = net.layer2
        self.extractor1 = net.layer3
        self.extractor2 = net.layer4

        self.upsample3 = DoubleConv(2048, 1024)
        self.upsample0 = Up(2048, 1024, 512)
        self.upsample1 = Up(1024, 512, 256)
        self.upsample2 = Up(512, 512, 256)

    def forward(self, x):
        x1 = self.extractor(x)  # 256
        x2 = self.extractor0(x1)  # 512
        x3 = self.extractor1(x2)  # 1024
        x4 = self.extractor2(x3)  # 2048
        ret = self.upsample3(x4)
        ret = self.upsample0(ret, x3)
        ret = self.upsample1(ret, x2)
        ret = self.upsample2(ret, x1)
        return ret


class ResNetExt3(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        net = models.resnet50(pretrained=pretrained)
        self.extractor = nn.Sequential()
        self.extractor.add_module("0", net.conv1)
        self.extractor.add_module("1", net.bn1)
        self.extractor.add_module("2", net.relu)
        self.extractor.add_module("3", net.maxpool)
        self.extractor.add_module("4", net.layer1)
        self.extractor0 = net.layer2
        self.extractor1 = net.layer3
        self.extractor2 = net.layer4

        self.upsample0 = Up(3072, 1024, 512)
        self.upsample1 = Up(1024, 512, 256)
        self.upsample2 = Up(512, 512, 256)

    def forward(self, x):
        x1 = self.extractor(x)
        x2 = self.extractor0(x1)
        x3 = self.extractor1(x2)
        x4 = self.extractor2(x3)
        ret = self.upsample0(x4, x3)
        ret = self.upsample1(ret, x2)
        ret = self.upsample2(ret, x1)
        return ret


class ResNetExt(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        net = models.resnet50(pretrained=pretrained)
        self.extractor = nn.Sequential()
        self.extractor.add_module("0", net.conv1)
        self.extractor.add_module("1", net.bn1)
        self.extractor.add_module("2", net.relu)
        self.extractor.add_module("3", net.maxpool)
        self.extractor.add_module("4", net.layer1)
        self.extractor.add_module("5", net.layer2)
        self.extractor1 = net.layer3
        self.extractor2 = net.layer4

        self.upsample0 = DoubleConv(2048, 1024)
        self.upsample1 = Up(2048, 1024, 512)
        self.upsample2 = Up(1024, 512, 256)

    def forward(self, x):
        x1 = self.extractor(x)
        x2 = self.extractor1(x1)
        x3 = self.extractor2(x2)
        return self.upsample2(self.upsample1(self.upsample0(x3), x2), x1)


def resnetupsample(pretrain):
    net = ResnetUpSample(pretrained=pretrain)
    return net


def resnetext(pretrain):
    net = ResNetExt(pretrained=pretrain)
    return net


def resnetext2(pretrain):
    net = ResNetExt2(pretrained=pretrain)
    return net


def resnetext3(pretrain):
    net = ResNetExt3(pretrained=pretrain)
    return net


def vgg16(layer="pool4"):
    net = models.vgg16(pretrained=True)
    model = nn.Sequential()
    features = nn.Sequential()
    for i in range(0, vgg_layers[layer]):
        features.add_module("{}".format(i), net.features[i])
    model.add_module("features", features)
    return model


def resnet50(pretrain):
    net = models.resnet50(pretrained=pretrain)
    extractor = nn.Sequential()
    extractor.add_module("0", net.conv1)
    extractor.add_module("1", net.bn1)
    extractor.add_module("2", net.relu)
    extractor.add_module("3", net.maxpool)
    extractor.add_module("4", net.layer1)
    extractor.add_module("5", net.layer2)
    extractor.add_module("6", net.layer3)
    extractor.add_module("7", net.layer4)
    return extractor


# original_img_size = torch.Size([224, 300])
# calculate which patch contains kp. if (1, 1) and line size = 9, return 1*9+1 = 10
def keypoints_to_pixel_index(keypoints, downsample_rate, original_img_size=(480, 640)):
    # line_size = 9
    line_size = original_img_size[1] // downsample_rate
    # round down, new coordinate (keypoints[:,:,0]//downsample_rate, keypoints[:, :, 1] // downsample_rate)
    return (
        keypoints[:, :, 0] // downsample_rate * line_size
        + keypoints[:, :, 1] // downsample_rate
    )


def get_noise_pixel_index(keypoints, max_size, n_samples, obj_mask=None):
    n = keypoints.shape[0]

    # remove the point in keypoints by set probability to 0 otherwise 1 -> mask [n, size] with 0 or 1
    mask = torch.ones((n, max_size), dtype=torch.float32).to(keypoints.device)
    mask = mask.scatter(1, keypoints.type(torch.long), 0.0)
    if obj_mask is not None:
        mask *= obj_mask

    # generate the sample by the probabilities
    try:
        return torch.multinomial(mask, n_samples)
    except:
        return get_noise_pixel_index(keypoints, max_size, n_samples, obj_mask=None)
        
        # return None
    """
    return torch.multinomial(mask, n_samples)
    """


class GlobalLocalConverter(nn.Module):
    def __init__(self, local_size):
        super().__init__()
        self.local_size = local_size
        self.padding = sum(([t - 1 - t // 2, t // 2] for t in local_size[::-1]), [])

    def forward(self, X):
        n, c, h, w = X.shape  # torch.Size([1, 2048, 8, 8])

        # N, C, H, W -> N, C, H + local_size0 - 1, W + local_size1 - 1
        X = F.pad(X, self.padding)

        # N, C, H + local_size0 - 1, W + local_size1 - 1 -> N, C * local_size0 * local_size1, H * W
        X = F.unfold(X, kernel_size=self.local_size)

        # N, C * local_size0 * local_size1, H * W -> N, C, local_size0, local_size1, H * W
        # X = X.view(n, c, *self.local_size, -1)

        # X:  N, C * local_size0 * local_size1, H * W
        return X


class MergeReduce(nn.Module):
    def __init__(self, reduce_method="mean"):
        super().__init__()
        self.reduce_method = reduce_method
        self.local_size = -1

    def register_local_size(self, local_size):
        self.local_size = local_size[0] * local_size[1]
        if self.reduce_method == "mean":
            self.foo_test = torch.nn.AvgPool2d(
                local_size,
                stride=1,
                padding=local_size[0] // 2,
            )
        elif self.reduce_method == "max":
            self.foo_test = torch.nn.MaxPool2d(
                local_size,
                stride=1,
                padding=local_size[0] // 2,
            )

    def forward(self, X):

        X = X.view(X.shape[0], -1, self.local_size, X.shape[2])
        if self.reduce_method == "mean":
            return torch.mean(X, dim=2)
        elif self.reduce_method == "max":
            return torch.max(X, dim=2)

    def forward_test(self, X):
        return self.foo_test(X)


def ind_sel(target: torch.Tensor, ind: torch.Tensor, dim: int=1):
    """
    :param target: [... (can be k or 1), n > M, ...]
    :param ind: [... (k), M]
    :param dim: dim to apply index on
    :return: sel_target [... (k), M, ...]
    """
    assert len(ind.shape) > dim, "Index must have the target dim, but get dim: %d, ind shape: %s" % (dim, str(ind.shape))

    target = target.expand(*tuple([ind.shape[k] if target.shape[k] == 1 else -1 for k in range(dim)] + [-1, ] * (len(target.shape) - dim)))

    ind_pad = ind

    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            ind_pad = ind_pad.unsqueeze(-1)
        ind_pad = ind_pad.expand(*(-1, ) * (dim + 1), *target.shape[(dim + 1)::])

    return torch.gather(target, dim=dim, index=ind_pad)


class NetE2E(nn.Module):
    def __init__(
        self,
        pretrain,
        net_type,
        local_size,
        output_dimension,
        reduce_function=None,
        n_noise_points=0,
        num_stacks=8,
        num_blocks=1,
        noise_on_mask=True,
        **kwargs
    ):
        # output_dimension = 128
        super().__init__()
        if net_type == "vgg_pool4":
            self.net = vgg16("pool4")
        elif net_type == "vgg_pool5":
            self.net = vgg16("pool5")
        elif net_type == "resnet50":
            self.net = resnet50(pretrain)
        elif net_type == "resnetext":
            self.net = resnetext(pretrain)
        elif net_type == "resnetext2":
            self.net = resnetext2(pretrain)
        elif net_type == "resnetext3":
            self.net = resnetext3(pretrain)
        elif net_type == "resnetupsample":
            self.net = resnetupsample(pretrain)
        elif net_type == "unet_res50":
            self.net = unet_res50(pretrain)

        self.size_number = local_size[0] * local_size[1]
        self.output_dimension = output_dimension
        # size_number = reduce((lambda x, y: x * y), local_size)
        if reduce_function:
            reduce_function.register_local_size(local_size)
            self.size_number = 1

        self.reduce_function = reduce_function
        self.net_type = net_type
        self.net_stride = net_stride[net_type]
        self.converter = GlobalLocalConverter(local_size)
        self.noise_on_mask = noise_on_mask

        # output_dimension == -1 for abilation study.
        if self.output_dimension == -1:
            self.out_layer = None
        else:
            self.out_layer = nn.Linear(
                net_out_dimension[net_type] * self.size_number, self.output_dimension
            )
            # output_dimension , net_out_dimension[net_type] * size_number

        self.n_noise_points = n_noise_points
        self.kwargs = kwargs
        # self.norm_layer = lambda x: F.normalize(x, p=2, dim=1)

    # forward
    def forward_test(self, X, return_features=False, do_normalize=True):
        m = self.net.forward(X)

        # not used
        if self.reduce_function:
            X = self.reduce_function.forward_test(m)
        else:
            X = m

        if self.out_layer is None:
            if do_normalize:
                return F.normalize(X, p=2, dim=1)
            else:
                return X
        if self.size_number == 1:
            X = torch.nn.functional.conv2d(X, self.out_layer.weight.unsqueeze(2).unsqueeze(3))
        elif self.size_number > 1:
            X = torch.nn.functional.conv2d(X, self.out_layer.weight.view(self.output_dimension,
                                                                         net_out_dimension[self.net_type],
                                                                         self.size_number).permute(2, 0, 1).reshape(
                self.size_number * self.output_dimension, net_out_dimension[self.net_type]).unsqueeze(2).unsqueeze(3))
        # print('X_new.shape', X.shape)
        # n, c, w, h
        # 1, 128, (w_original - 1) // 32 + 1, (h_original - 1) // 32 + 1
        # X = F.interpolate(X, scale_factor=2, mode='bilinear')
        if do_normalize:
            X = F.normalize(X, p=2, dim=1)

        if return_features:
            return X, m
        else:
            return X

    def forward_step0(self, X, do_normalize=True, return_features=False, **kwargs):
        # downsample_rate = 32
        # pre--X.shape torch.Size([1, 3, 256, 256])
        m = self.net.forward(X)

        if self.out_layer is not None:
            X = torch.nn.functional.conv2d(m, self.out_layer.weight.unsqueeze(2).unsqueeze(3))
        else:
            X = m

        if do_normalize:
            X = F.normalize(X, p=2, dim=1)

        if return_features:
            return X, m
        return X

    def forward_step1(self, X, keypoint_positions, img_shape, obj_mask=None, **kwargs):
        n = X.shape[0]

        # N, C * local_size0 * local_size1, H * W
        X = self.converter(X)

        keypoint_idx = keypoints_to_pixel_index(keypoints=keypoint_positions,
                                                downsample_rate=self.net_stride,
                                                original_img_size=img_shape).type(torch.long)

        # Never use this reduce_function part.
        if self.reduce_function:
            X = self.reduce_function(X)
        if self.n_noise_points == 0:
            keypoint_all = keypoint_idx
        else:
            if obj_mask is not None:
                obj_mask = F.max_pool2d(obj_mask.unsqueeze(dim=1),
                                        kernel_size=self.net_stride,
                                        stride=self.net_stride,
                                        padding=(self.net_stride - 1) // 2)
                obj_mask = obj_mask.view(obj_mask.shape[0], -1)
                assert obj_mask.shape[1] == X.shape[2], 'mask_: ' + str(obj_mask.shape) + ' fearture_: ' + str(X.shape)
            if self.noise_on_mask:
                keypoint_noise = get_noise_pixel_index(keypoint_idx,
                                                       max_size=X.shape[2],
                                                       n_samples=self.n_noise_points,
                                                       obj_mask=obj_mask)
            else:
                keypoint_noise = get_noise_pixel_index(keypoint_idx,
                                                       max_size=X.shape[2],
                                                       n_samples=self.n_noise_points,
                                                       obj_mask=None)
            
            keypoint_all = torch.cat((keypoint_idx, keypoint_noise), dim=1)

        # n * c * k -> n * k * c
        # N, C * local_size0 * local_size1, H * W - >  #N, H * W, C * local_size0 * local_size1
        X = torch.transpose(X, 1, 2)

        # N, H * W, C * local_size0 * local_size1 -> N, keypoint_all, C * local_size0 * local_size1
        X = ind_sel(X, dim=1, ind=keypoint_all)

        if self.out_layer is None:
            X = X.view(n, -1, net_out_dimension[self.net_type])
        else:
            X = X.view(n, -1, self.out_layer.weight.shape[0])

        return X

    def forward(self, *args, mode=-1, **kwargs):
        if mode == -1:
            if 'X' in kwargs.keys():
                img_shape = kwargs['X'].shape[2::]
            else:
                img_shape = args[0].shape[2::]
            X = self.forward_step0(*args, **kwargs)
            if len(args):
                args = args[1::]
            if 'X' in kwargs.keys():
                del kwargs['X']
            return self.forward_step1(*args, X=X, img_shape=img_shape, **kwargs)
        elif mode == 0:
            return self.forward_step0(*args, **kwargs)
        elif mode == 1:
            return self.forward_step1(*args, **kwargs)

    def cuda(self, device=None):
        self.net.cuda(device=device)
        self.out_layer.cuda(device=device)
        return self


if enable_voge:
    def dict_to_frag(kdict):
        return Fragments(vert_weight=kdict['vert_weight'], 
                         vert_index=kdict['vert_index'], 
                         valid_num=kdict['valid_num'], 
                         vert_hit_length=kdict['vert_hit_length'])


def get_noise_pixel_index_voge(keypoints, max_size, n_samples, ):
    """
    :param keypoints: [n, h, w, k]
    :param max_size: int -> h * w
    :param n_samples:  int -> k_noise
    :return:  samples -> [n, k_noise]
    """
    n = keypoints.shape[0]

    mask = torch.ones((n, max_size), dtype=torch.float32).to(keypoints.device)
    # print(keypoints.shape, keypoints.max(), keypoints.min())
    # [n, h, w, k] -> [n, h * w]
    # print((keypoints + (keypoints < 0).float()).min(), keypoints.min())
    mask = torch.nn.functional.relu(mask - (keypoints + (keypoints < 0).float()).sum(3).view(n, -1))
    # print(mask.sum(1))

    # print(torch.sum(mask, dim=1, keepdim=True) <= n_samples)
    # if the image is full occupied by object -> fill the value to avoid error, rarely happened
    mask = mask + (torch.sum(mask, dim=1, keepdim=True) <= n_samples).float() * torch.rand_like(mask)

    return torch.multinomial(mask, n_samples)


from VoGE.Utils import ind_fill
def sample_features_debug(frag, image, n_vert=None):
    weight = torch.zeros(image.shape[0:3] + (n_vert, )).to(image.device)
    weight = ind_fill(weight, frag.vert_index.long(), dim=3, src=frag.vert_weight)
    vert_sum_weight = torch.sum(weight, dim=(0, 1, 2), keepdim=True)
    vert_feature = weight.contiguous().view(-1, weight.shape[-1]).T @ image.contiguous().view(-1, image.shape[-1]) # [B * H * W, N].T @ [B * H * W, C] -> [N, C]

    return vert_feature, vert_sum_weight

def scatter_max_debug(frag, image, n_vert=None):
    weight = torch.zeros(image.shape[0:3] + (n_vert, )).to(image.device)
    weight = ind_fill(weight, frag.vert_index.long(), dim=3, src=frag.vert_weight)
    vert_max_weight = torch.max(weight.view(-1, n_vert), dim=0)

    return vert_max_weight

@torch.no_grad()
def frag_index_modifier(keypoint_positions, k):
    keypoint_positions['vert_index'] -= keypoint_positions['start_idx'].min() * k
    keypoint_positions['vert_index'][keypoint_positions['vert_index'] < 0] = -1


# For VoGE-NeMo
class WeightSampleNetE2E(NetE2E):
    def forward_step1(self, X, keypoint_positions, img_shape, obj_mask=None, **kwargs):
        assert enable_voge
        # X: [n, c, h, w]    keypoint_positions->frag, each_shape: [n, h, w, k]
        n, c, h, w = X.shape
        k = self.kwargs['n_vert']
        
        # print(keypoint_positions['start_idx'].shape, keypoint_positions['vert_index'].shape)
        # print(keypoint_positions['start_idx'].min(), ':', (keypoint_positions['vert_index'] - (keypoint_positions['start_idx'][:, None, None, None] - torch.arange(n).to(X.device)[:, None, None, None]) * k)[:, 38, 38, 0])
        frag_index_modifier(keypoint_positions, k)
        frag_ = dict_to_frag(keypoint_positions)

        # print(keypoint_positions['start_idx'].min(), ':', keypoint_positions['vert_index'][keypoint_positions['vert_index'] >= 0].min(), keypoint_positions['vert_index'].max())
        # [n, h, w, c] -> [n * k, c]
        vert_feature, vert_sum_weight = sample_features(frag_, X.permute(0, 2, 3, 1), n_vert=n * k)

        # [n * k, c]
        X_kp = vert_feature / (vert_sum_weight[:, None] + 1e-8)

        X_kp = X_kp.view(n, k, c)

        if self.n_noise_points == 0:
            X_noise = torch.zeros((n, c, 0), device=X.device)
        else:
            # [n, w * h]
            keypoint_noise = get_noise_pixel_index_voge(frag_.vert_weight, max_size=h * w, n_samples=self.n_noise_points)

            # [n, c, k_noise]
            X_noise = ind_sel(X.view(n, c, -1), ind=keypoint_noise.unsqueeze(1).expand(-1, c, -1), dim=2)

        # [n, k, c] , [n, c, k_noise] -> [n, k + k_noise, c]
        X_out = torch.cat((X_kp, X_noise.transpose(1, 2)), dim=1)

        # X_out = F.normalize(X_out, p=2, dim=-1)
        return X_out


# For VoGE-NeMo, used in voge paper
class WeightDotNetE2E(NetE2E):
    def forward_step1(self, X, keypoint_positions, img_shape, obj_mask=None, **kwargs):
        # X: [n, c, h, w]    keypoint: [n, h, w, k]
        n, c, h, w = X.shape
        k = keypoint_positions.shape[-1]

        assert keypoint_positions.shape[1] == X.shape[2], "Unaligned feature and keypoints, feature: " + str(X.shape) + ' keypoints: ' + str(keypoint_positions.shape)
        assert keypoint_positions.shape[2] == X.shape[3], "Unaligned feature and keypoints, feature: " + str(X.shape) + ' keypoints: ' + str(keypoint_positions.shape)

        # [n, w, h, k] -> [n, k]
        kp_sum = keypoint_positions.sum((1, 2))

        # [n, c, w * h] @ [n, w * h, k] -> [n, c, k]
        X_kp = torch.bmm(X.view(n, c, -1), keypoint_positions.view(n, -1, k)) / (kp_sum.unsqueeze(1) + 1e-10)

        if self.n_noise_points == 0:
            X_noise = torch.zeros((n, c, 0), device=X.device)
        else:
            # [n, w * h]
            keypoint_noise = get_noise_pixel_index_voge(keypoint_positions, max_size=h * w, n_samples=self.n_noise_points)

            # [n, c, k_noise]
            X_noise = ind_sel(X.view(n, c, -1), ind=keypoint_noise.unsqueeze(1).expand(-1, c, -1), dim=2)

        # [n, c, k + k_noise] -> [n, k + k_noise, c]
        X_out = torch.cat((X_kp, X_noise), dim=2).transpose(1, 2)

        # X_out = F.normalize(X_out, p=2, dim=-1)
        return X_out
