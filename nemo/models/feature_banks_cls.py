import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch import nn
import math
ALL_CLASSES=["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]


def one_hot(y, max_size=None):
    if not max_size:
        max_size = int(torch.max(y).item() + 1)
    # print('y.shape',y.shape)
    # print('max_size',max_size)
    y = y.view(-1, 1)
    y_onehot = torch.zeros((y.shape[0], max_size), dtype=torch.float32, device=y.device)
    y_onehot.scatter_(1, y.type(torch.long), 1)
    return y_onehot

def bank_selection_fun(label):
    ret = torch.zeros((len(ALL_CLASSES), len(ALL_CLASSES)),  dtype=torch.float32, label=y.device)
    for l in label:
        ret[l][l] = 1
    return ret

def to_mask(y, max_size):
    y_onehot = torch.zeros((len(y), max_size), dtype=torch.float32, device=y[0].device)
    for i in range(len(y)):
        y_onehot[i].scatter_(0, y[i].type(torch.long), 1)
    return y_onehot


def mask_remove_near(keypoints, thr, img_label, n_list, pad_index, zeros, dtype_template=None, num_neg=0, neg_weight=1, eps=1e5):

    if dtype_template is None:
        dtype_template = torch.ones(1, dtype=torch.float32)

    if dtype_template.dtype != zeros.dtype:
        zeros = zeros.type_as(dtype_template)



    # keypoints -> [n, k, 2]
    with torch.no_grad():
        # distance -> [n, k, k]
        distance = torch.sum((torch.unsqueeze(keypoints, dim=1) - torch.unsqueeze(keypoints, dim=2)).pow(2), dim=3).pow(0.5)
        if num_neg == 0:
            return ((distance <= thr).type_as(dtype_template) - torch.eye(keypoints.shape[1]).type_as(dtype_template).unsqueeze(dim=0)) * eps
        else:
            tem = (distance <= thr).type_as(dtype_template) - torch.eye(keypoints.shape[1]).type_as(dtype_template).unsqueeze(dim=0)
            # TODO: Change its dimension according to current bank, and also sets values of padded keypoints in 2nd dimension to large nums.

            if zeros.shape[0] != keypoints.shape[0]:
                zeros = torch.zeros(tem.shape[0], tem.shape[1], tem.shape[1]*len(ALL_CLASSES)).type_as(dtype_template)

            for i in range(tem.shape[0]):
                zeros[i, :, img_label[i] * tem.shape[1] : (img_label[i] + 1) * tem.shape[1]] = tem[i] * eps

            # for i in range(12):
                # ret[:, :, tem.shape[1] * i + n_list[i] : tem.shape[1] * (i+1)] = eps

            zeros[:, :, pad_index.view(-1)] = eps

            # what is concated is the clutter part, does not need to change
            return torch.cat([zeros, - torch.ones(keypoints.shape[0: 2] + (num_neg, )).type_as(dtype_template) * math.log(neg_weight)], dim=2)

def fun_label_onehot(img_label, count_label):
    ret = torch.zeros(img_label.shape[0], len(ALL_CLASSES)).to(img_label.device)
    ret = ret.scatter_(1, img_label.unsqueeze(1), 1.0).to(img_label.device)
    for i in range(len(ALL_CLASSES)):
        count = count_label[i]
        if count == 0:
            continue
        ret[:, i] /= count
    return ret


class NearestMemorySelective(nn.Module):
    def forward(self, x, y, visible, n_pos, n_neg, lru, memory, params, eps=1e-8):
        group_size = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()

        similarity = torch.sum(torch.unsqueeze(x[0:n_pos], 1) * torch.unsqueeze(memory, 0), dim=2)

        n_class = n_pos // group_size
        y_onehot = one_hot(y, n_class)

        if not group_size == 1:
            y_onehot = y_onehot.unsqueeze(2).expand(-1, -1, group_size).contiguous().view(y.shape[0], -1)

        y_idx = torch.argmax(similarity[:, 0:n_pos] + y_onehot * 2, dim=1).type(torch.long)
        visible = to_mask(visible, n_pos)

        with torch.no_grad():
            # update memory keypoints
            # [n, k]
            idx_onehot = one_hot(y_idx, n_pos)

            # [k, d]
            get = torch.mm(torch.t(idx_onehot), x[0:n_pos, :])
            counts = torch.t(torch.sum(idx_onehot, dim=0, keepdim=True))
            valid_mask = (counts > 0.1).type(counts.dtype) * visible.view(-1, 1)
            get /= counts + eps

            memory[0:n_pos, :].copy_(F.normalize(
                memory[0:n_pos, :] * (valid_mask * momentum + 1 - valid_mask) + get * (1 - momentum) * valid_mask,
            dim=1, p=2))

            # Update trash bin
            memory[n_pos + lru * n_neg: n_pos + (lru + 1) * n_neg, :].copy_(x[n_pos::, :])

        return similarity, y_idx


class NearestMemoryManager(nn.Module):
    def __init__(self, input_size, output_size, K, num_pos, n_list_set, T=0.07, momentum=0.5, Z=None, max_groups=-1, num_noise=-1):
        super(NearestMemoryManager, self).__init__()
        self.nLem = output_size
        self.K = K
        self.n_list_set = n_list_set

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(input_size / 3)
        self.single_feature_dim = int(num_pos / len(ALL_CLASSES))

        # TODO: add one more dimension for 12 classes, each class has a memory bank.
        '''
        self.num_clutter = outputSize - num_pos

        self.memory_set = []
        for i in range(12):
            self.memory_set.append(torch.rand(self.single_feature_dim, inputSize).mul_(2 * stdv).add_(-stdv))
        self.clutter_memory = torch.rand(self.num_clutter, inputSize).mul_(2 * stdv).add_(-stdv)
        '''

        self.memory = torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv)
        self.memory.requires_grad = False

        self.lru = 0
        if max_groups > 0:
            self.max_lru = max_groups
        else:
            self.max_lru = -1

        if num_noise < 0:
            self.num_noise = self.K
        else:
            self.num_noise = num_noise

        self.num_pos = num_pos

        # self.accumulate_num = torch.zeros(self.num_pos, dtype=torch.long, device=self.memory.device)
        # self.accumulate_num.requires_grad = False

    # x: feature: [128, 128], y: indexes [128] -- a batch of data's index directly from the dataloader.
    def forward(self, x, y, visible, img_label):
        # k = feature_space
        # n = batch_size
        n_pos = self.num_pos
        n_neg = self.num_noise
        count_label = torch.bincount(img_label, minlength=len(ALL_CLASSES))
        label_weight_onehot = fun_label_onehot(img_label, count_label)

        # set max_group is it is explicitly given. (We give max_group as 512, so not in our case)
        if self.max_lru == -1 and n_neg > 0 and x.shape[0] <= (self.nLem - n_pos) / n_neg:
            self.max_lru = (self.memory.shape[0] - n_pos) // (n_neg * x.shape[0])

        # group size is 1, no need to care.
        # print(self.params)
        group_size = int(self.params[0].item())
        momentum = self.params[3].item()
        assert group_size == 1, 'Currently only support group size = 1'

        # set all classes memory padded keypoints to 0.


        # x [n, k, d] * memory [l, d] = similarity : [n, k, l]
        if n_neg == 0:
            similarity = torch.matmul(x, torch.transpose(self.memory, 0, 1))
            noise_similarity = torch.zeros(1)
        else:
            # first dimension is batch, get first n space that represnets keypoints
            t_ = x[:, 0:self.single_feature_dim, :]
            # calculate similarity that includes clutter in memory bank
            similarity = torch.matmul(t_, torch.transpose(self.memory, 0, 1))
            # calculate the space after first n space that serves as noise for noise similarity calculation, calculated with feature part of memory
            noise_similarity = torch.matmul(x[:, self.single_feature_dim:, :], torch.transpose(self.memory[0:n_pos, :], 0, 1))



        # group size is 1, n_class equals to n_pos (feature space)
        n_class = n_pos // group_size

        with torch.set_grad_enabled(False):
            # [n, k, k]
            '''
            # change label to one-hot format. 1 at diagonal.
            y_onehot = one_hot(y, n_class).view(x.shape[0], -1, n_class)

            # useless branch since group size is always 1
            if not group_size == 1:
                # [n, k, k * g]
                y_onehot = y_onehot.unsqueeze(2).expand(-1, -1, group_size).contiguous().view(x.shape[0], -1, n_pos)
                visible = visible.unsqueeze(2).expand(-1, -1, group_size).contiguous().view(x.shape[0], -1, n_pos)
                y_idx = torch.argmax(similarity[:, :, 0:n_pos] + y_onehot * 2, dim=2).type(torch.long)
            else:
            '''
            y_idx = y.type(torch.long)

            # Calculate memory keypoints update
            # [n, k, k] * [n, k, d] -> [n, k, d]
            # Timing the visible will elimiate the effect of those keypoints that are not visible
            # get = torch.bmm(torch.transpose(y_onehot, 1, 2), x[:, 0:n_pos, :] * visible.type(x.dtype).view(*visible.shape, 1))

            get = torch.matmul(label_weight_onehot.transpose(0, 1), (x[:, 0:self.single_feature_dim, :] * visible.type(x.dtype).view(*visible.shape, 1)).view(x.shape[0], -1))
            get = get.view(get.shape[0] ,-1, x.shape[-1])
            # handle 0 in get, case that no img of one class is in the batch
            tmp = (count_label == 0).nonzero(as_tuple=True)[0]
            for i in tmp:
                # copy memory to get
                get[i] = self.memory[i*self.single_feature_dim : (i+1)*self.single_feature_dim]
            get = get.view(-1, x.shape[-1])


            if n_neg > 0:

                # Update for the memory.
                # TODO: The view here for clutter part should not be applied like this for classification, batch and clutter cannot be viewed to same dimension since img now may come from different class.
                # if batch > max_group
                # n_pos:: has the same meaning as n_pos:, indicating taking the rest, continguous is combined with view() (no independent meaning)

                if x.shape[0] > (self.nLem - n_pos) / n_neg:
                    self.memory = F.normalize(torch.cat([self.memory[0:n_pos, :] * momentum + get * (1 - momentum), x[:, self.single_feature_dim::, :].contiguous().view(-1, x.shape[2])[0:self.memory.shape[0] - n_pos]], dim=0), dim=1, p=2)
                else:
                    # handle case if batchsize is not larger than max_group
                    # neg_parts updated based on tagging
                    neg_parts = torch.cat([self.memory[n_pos:n_pos + self.lru * n_neg * x.shape[0], :],
                                           x[:, self.single_feature_dim::, :].contiguous().view(-1, x.shape[2]),
                                           self.memory[n_pos + (self.lru + 1) * n_neg * x.shape[0]::, :]], dim=0)

                    self.memory = F.normalize(torch.cat([self.memory[0:n_pos, :] * momentum + get * (1 - momentum), neg_parts], dim=0), dim=1, p=2)
            else:

                self.memory = F.normalize(self.memory[0:n_pos, :] * momentum + get * (1 - momentum), dim=1, p=2)


            # self.accumulate_num += torch.sum((visible > 0).type(self.accumulate_num.dtype).to(self.accumulate_num.dtype), dim=0)
            self.lru += 1
            self.lru = self.lru % self.max_lru
        # out.shape: [d, n_neg + n_pos]
        return similarity, y_idx, noise_similarity, label_weight_onehot

    def forward_local(self, x, y, visible):
        # print('x',x.shape)
        # print('noise',self.num_noise)
        n_pos = self.num_pos
        n_neg = self.num_noise

        if self.max_lru == -1 and n_neg > 0:
            self.max_lru = (self.memory.shape[0] - n_pos) // (n_neg * x.shape[0])

        group_size = int(self.params[0].item())
        momentum = self.params[3].item()

        assert group_size == 1, 'Currently only support group size = 1'

        # x [n, k, d] * memory [l, d] = similarity : [n, k, l]
        if n_neg == 0:
            t_ = x[:, 0:n_pos, :]

            t_neg = x[:, n_pos::, :]
            similarity = torch.matmul(t_, torch.transpose(self.memory, 0, 1))

            # [n, k, 1, d] * [n, 1, c, d] -> [n, k, c]
            similarity_neg = torch.sum(t_.unsqueeze(2) * t_neg.unsqueeze(1), dim=3)

            similarity = torch.cat([similarity, similarity_neg], dim=2)

            noise_similarity = torch.zeros(1)
        else:
            t_ = x[:, 0:n_pos, :]
            similarity = torch.matmul(t_, torch.transpose(self.memory, 0, 1))
            noise_similarity = torch.matmul(x[:, n_pos:, :], torch.transpose(self.memory[0:n_pos, :], 0, 1))

        n_class = n_pos // group_size

        with torch.set_grad_enabled(False):
            # [n, k, k]
            y_onehot = one_hot(y, n_class).view(x.shape[0], -1, n_class)

            if not group_size == 1:
                # [n, k, k * g]
                y_onehot = y_onehot.unsqueeze(2).expand(-1, -1, group_size).contiguous().view(x.shape[0], -1, n_pos)
                visible = visible.unsqueeze(2).expand(-1, -1, group_size).contiguous().view(x.shape[0], -1, n_pos)

                y_idx = torch.argmax(similarity[:, :, 0:n_pos] + y_onehot * 2, dim=2).type(torch.long)
            else:
                y_idx = y.type(torch.long)

            # update memory keypoints
            # [n, k, d]
            get = torch.bmm(torch.transpose(y_onehot, 1, 2), x[:, 0:n_pos, :] * visible.type(x.dtype).view(*visible.shape, 1))

            # [k, d]
            get = torch.mean(get, dim=0) # / torch.sum(visible, dim=0).view(-1, 1)

            self.accumulate_num += torch.sum(visible.type(self.accumulate_num.dtype), dim=0)
            self.lru += 1
            self.lru = self.lru % self.max_lru

        # out.shape -- torch.Size([128, 4097])
        return similarity, y_idx, noise_similarity

    def set_zero(self, n_pos):
        self.accumulate_num = torch.zeros(n_pos, dtype=torch.long, device=self.memory.device)
        self.memory.fill_(0)

    def accumulate_memory(self, x, y, visible, eps=1e-8):
        # print(self.params)
        group_size = int(self.params[0].item())

        # Currently only support group size = 1
        assert group_size == 1, 'Currently only support group size = 1'

        n_pos = self.num_pos
        n_neg = self.num_noise

        with torch.no_grad():
            # print(visible.shape)

            # update memory keypoints
            # [n, k, k]
            idx_onehot = one_hot(y, n_pos).view(x.shape[0], -1, n_pos)

            # [k, d]
            get = torch.bmm(torch.transpose(idx_onehot, 1, 2), x[:, 0:n_pos, :] * visible.type(x.dtype).view(*visible.shape, 1))
            get = torch.sum(get, dim=0)

            self.memory[0:n_pos, :].copy_(self.memory[0:n_pos, :] + get)
            self.accumulate_num += torch.sum(visible.type(self.accumulate_num.dtype), dim=0)

    def normalize_memory(self):
        self.memory.copy_(F.normalize(self.memory, p=2, dim=1))

    def cuda(self, device=None):
        super().cuda(device)
        # self.accumulate_num = self.accumulate_num.cuda(device)
        self.memory = self.memory.cuda(device)
        # print(self.params)
        return self
