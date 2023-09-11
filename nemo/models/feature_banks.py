import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo.utils.pascal3d_utils import CATEGORIES

try:
    from CuNeMo import get_mask
    from CuNeMo import gather_features
    enable_cunemo = True
except:
    enable_cunemo = False


def one_hot(y, max_size=None):
    if not max_size:
        max_size = int(torch.max(y).item() + 1)
    # print('y.shape',y.shape)
    # print('max_size',max_size)
    y = y.view(-1, 1)
    y_onehot = torch.zeros((y.shape[0], max_size), dtype=torch.float32, device=y.device)
    y_onehot.scatter_(1, y.type(torch.long), 1)
    return y_onehot


def fun_label_onehot(img_label, count_label):
    ret = torch.zeros(img_label.shape[0], len(CATEGORIES)).to(img_label.device)
    ret = ret.scatter_(1, img_label.unsqueeze(1), 1.0).to(img_label.device)
    for i in range(len(CATEGORIES)):
        count = count_label[i]
        if count == 0:
            continue
        ret[:, i] /= count
    return ret



class MaskCreater():
    def __init__(self, dist_thr, kappas, n_noise, verts_ori=None, device='cpu'):
        """
        dist_thr: float, distance threshold for hard negetive mining in CoKe
        kappas: dict, {'pos', 'near', 'clutter', 'class'} weight for each term in NeMo loss, applied via exp(f * theta - kappa_x), kappa_class = 1e5 when on classification loss
        n_noise: int
        verts_ori: list of tensor, [(K, 3), ] verts locations
        """
        if verts_ori is not None:
            with torch.no_grad():
                vert_max_num = max([v.shape[0] for v in verts_ori])
                self.vert_sum_num = sum([v.shape[0] for v in verts_ori])
                
                out_vert_dist = []
                assert isinstance(verts_ori, list)
                for vv in verts_ori:
                    get_vert_ = torch.cat([vv, torch.zeros((vert_max_num - vv.shape[0], 3), dtype=vv.dtype, device=vv.device)], dim=0)
                    vert_dist = (get_vert_[:, None] - get_vert_[None]).pow(2).sum(-1).pow(.5)
                    out_vert_dist.append(vert_dist)
                vert_dists = torch.stack(out_vert_dist)
            self.verts_dist_weight = (vert_dists <= dist_thr).to(device).type(torch.float32) * kappas['near'] + torch.eye(vert_max_num).to(device).type(torch.float32).unsqueeze(dim=0) * (kappas['pos'] - kappas['near'])
            self.mesh_n_list = [v.shape[0] for v in verts_ori]
        else:
            self.verts_dist_weight = None

        self.kappas = {'pos':0, 'near':1e5, 'clutter': 5, 'class':1e5}
        self.kappas.update(kappas)
        self.dist_thr = dist_thr
        self.n_noise = n_noise
        self.device = device

    def __call__(self, sample_indexs=None, K_padded=None, vis_mask=None, kps=None, dtype_template=None):
        """
        Usable combinations: 
        One object per image:
            [kps, (optional) vis_mask] => Ori NeMo
            [dtype_template, (optional) vis_mask] => VoGE NeMo, OmniNeMo
        Multiple objects per image:
            [sample_indexs, dtype_template, (optional) vis_mask]
            [sample_indexs, K_padded, (optional) vis_mask]

        sample_indexs: (B, ) or (B, M), class label of each instance in each image
        K_padded: int, number of max verts in each branch -> dtype_template.shape[1]
        vis_mask: (B, K_padded)
        kps: (B, K_padded, 2) project keypoints locations
        dtype_template: (B, K_padded) or (B, K_padded, N_verts_total + N_noise) reference for tensor shape

        return:
        weight_mask: (S, N_verts_total + N_noise), the weight matrix
        vert_index: (S, ), type=long, index label for computing coke loss
        """
        if kps is not None:
            # assert self.verts_dist_weight is None, 'Ethier keypoint location of verts is specificed!'
            # (b, k, k)
            vert_dists = (kps[:, None] - kps[:, :, None]).pow(2).sum(-1).pow(.5)
            verts_dist_weight = (vert_dists <= self.dist_thr).to(self.device).type(torch.float32) * self.kappas['near'] + torch.eye(kps.shape[1]).to(self.device).type(torch.float32).unsqueeze(dim=0) * (self.kappas['pos'] - self.kappas['near'])
        else:
            verts_dist_weight = self.verts_dist_weight
            
        if sample_indexs is None:
            # CoKe loss only -> for pose
            if verts_dist_weight.dim() == 2:
                verts_dist_weight = verts_dist_weight[None]
            get = torch.cat([verts_dist_weight, torch.ones(verts_dist_weight.shape[0: 2] + (self.n_noise, ), device=verts_dist_weight.device) * self.kappas['clutter']], dim=2)
            if get.shape[0] == 1:
                assert dtype_template is not None
                b_ = dtype_template.shape[0]
                get = get.expand(b_, -1, -1).contiguous()
            else:
                b_ = get.shape[0]
            
            vert_index = torch.arange(verts_dist_weight.shape[1], device=self.device)[None].long().expand(b_, -1).contiguous()

            if vis_mask is not None:
                return get[vis_mask], vert_index[vis_mask]
            else:
                return get.view(-1), vert_index.view(-1)
        else:
            if not enable_cunemo:
                raise Exception("Multi class in same batch requires CuNeMo (located at ./cu_layers)")
            assert kps is None, 'Current only support verts based distance constrin'
            # Classification & CoKe -> for multiple class of instance in same batch
            
            if sample_indexs.dim() == 1:
                sample_indexs = sample_indexs[:, None]
            # assert K_padded is not None
            if K_padded is None:
                assert dtype_template is not None
                K_padded = dtype_template.shape[1]
            total_size = self.vert_sum_num + self.n_noise

            return get_mask(verts_dist_weight, sample_indexs, self.mesh_n_list, total_size, K_padded, self.kappas['class'], self.kappas['clutter'], mask_sel=vis_mask, n_noise=self.n_noise)


class FeatureBankNeMo(nn.Module):
    # New and clear implementation of NeMo feature banks
    def __init__(
        self,
        input_size,  # n channels
        num_pos,  # n vertex total
        num_noise=-1,  # n clutter per image
        max_groups=-1,  # n image contains clutter saved
        momentum=0.5,
        **kwargs
    ):
        super().__init__()

        stdv = 1.0 / math.sqrt(input_size / 3)

        self.memory_pos = torch.rand(num_pos, input_size).mul_(2 * stdv).add_(-stdv)
        self.memory_neg = torch.rand(num_noise * max_groups if max_groups > 0 else 0, input_size).mul_(2 * stdv).add_(-stdv)

        self.memory_pos.requires_grad = False
        self.memory_neg.requires_grad = False

        self.num_noise = num_noise

        self.lru = 0
        if max_groups > 0:
            self.max_lru = max_groups
        else:
            self.max_lru = -1

        self.kwargs = kwargs
        self.momentum = momentum
    
    def forward(self, x, visible, x_to_bank=None, object_labels=None, vis_mask=None):
        """
        x (B, K, C): extracted vertex features
        visible (B, K): vertex visibility, should handle the padded vertex -> padded vertex vis = False
        x_to_bank (B, K, C) or (M, C): can be directly pass to banks, otherwise x_to_bank is reduce_function(x)
        object_labels (B, N_obj_per_img): indicates object class in each image, -1 for padding
        vis_mask: (B, K_padded)

        return:
        similarity (S, N_verts_total + N_noise)
        noise_similarity (B, N_noise, N_verts_total)
        """
        if self.num_noise == 0:
            t_ = x
            noise_similarity = torch.zeros(1)
        else:   
            t_ = x[:, 0:(x.shape[1] - self.num_noise), :]
            noise_similarity = torch.matmul(
                x[:, -self.num_noise:, :], torch.transpose(self.memory_pos, 0, 1)
            )
        
        similarity = torch.matmul(t_.view(-1) if vis_mask is None else t_[vis_mask], torch.transpose(self.memory, 0, 1))

        with torch.no_grad():
            if x_to_bank is None:
                x_to_bank = x[:, 0:(x.shape[1] - self.num_noise), :]
            b_ = x_to_bank.shape[0]

            # Expect x_to_bank to be (n_pos, c), if not, do reduce
            if x_to_bank.dim() == 3:
                # Average reducation
                if object_labels is None:
                    x_to_bank = torch.mean(x_to_bank * visible.type(x.dtype)[..., None], dim=0)
                else:
                    x_to_bank, x_vis_count = gather_features(x_to_bank, weights=visible.type(torch.float32), sample_indexs=object_labels.type(torch.int32), mesh_n_list=self.kwargs.get('mesh_n_list'))
                    x_to_bank = x_to_bank / b_

            # Assume x is aligned to banked features
            self.memory_pos = F.normalize(self.memory_pos * self.momentum + x_to_bank * (1 - self.momentum), dim=1, p=2, )

            if self.num_noise > 0:
                if x.shape[0] * self.num_noise > self.memory_neg.shape[0]:
                    self.memory_neg = x[:, -self.num_noise:, :].contiguous().view(-1, x.shape[2])[0:self.memory_neg.shape[0]]
                else:
                    self.memory_neg = torch.cat(
                        [
                            self.memory_neg[0:self.lru * self.num_noise, :],
                            x[:, -self.num_noise:, :].contiguous().view(-1, x.shape[2]),
                            self.memory_neg[(self.lru + x.shape[0]) * self.num_noise ::,:],
                        ],
                        dim=0,
                    )[:self.max_lru * self.num_noise]

            self.lru += x.shape[0]
            self.lru = self.lru % self.max_lru

        # out.shape: [d, n_neg + n_pos]
        return similarity, noise_similarity

    def cuda(self, device=None):
        super().cuda(device)
        self.memory_pos = self.memory_pos.cuda(device)
        self.memory_neg = self.memory_neg.cuda(device)
        return self

    @property
    def memory(self):
        return torch.cat((self.memory_pos, self.memory_neg), dim=0)


class NearestMemoryManager(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        K,
        num_pos,
        T=0.07,
        momentum=0.5,
        Z=None,
        max_groups=-1,
        num_noise=-1,
        classification=False,
        **kwargs
    ):
        super().__init__()
        self.nLem = output_size
        self.K = K

        self.register_buffer("params", torch.tensor([K, T, -1, momentum]))
        stdv = 1.0 / math.sqrt(input_size / 3)

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

        # For classification
        self.classification = classification
        self.single_cate_pos = int(self.num_pos / len(CATEGORIES))

        if classification:
            self.accumulate_num = torch.zeros(
                self.single_cate_pos, dtype=torch.long, device=self.memory.device
            ) 
        else:
            self.accumulate_num = torch.zeros(
                self.num_pos, dtype=torch.long, device=self.memory.device
            ) 
        self.accumulate_num.requires_grad = False

    # x: feature: [128, 128], y: indexes [128] -- a batch of data's index directly from the dataloader.
    def forward(self, x, y, visible, img_label=None):
        n_pos = self.num_pos  # 1024 
        n_neg = self.num_noise  # 5
        
        if (
            self.max_lru == -1
            and n_neg > 0
            and x.shape[0] <= (self.nLem - n_pos) / n_neg
        ):
            self.max_lru = (self.memory.shape[0] - n_pos) // (n_neg * x.shape[0])

        group_size = int(self.params[0].item())
        momentum = self.params[3].item()

        assert group_size == 1, "Currently only support group size = 1"

        # x [n, k, d] * memory [l, d] = similarity : [n, k, l]
        if n_neg == 0:
            similarity = torch.matmul(x, torch.transpose(self.memory, 0, 1))
            noise_similarity = torch.zeros(1)
        else:
            if self.classification:
                t_ = x[:, 0:self.single_cate_pos, :]
                similarity = torch.matmul(t_, torch.transpose(self.memory, 0, 1))
                noise_similarity = torch.matmul(
                    x[:, self.single_cate_pos:, :], torch.transpose(self.memory[0:n_pos, :], 0, 1)
                )
            else:    
                t_ = x[:, 0:n_pos, :]
                similarity = torch.matmul(t_, torch.transpose(self.memory, 0, 1))
                noise_similarity = torch.matmul(
                    x[:, n_pos:, :], torch.transpose(self.memory[0:n_pos, :], 0, 1)
                )

        n_class = n_pos // group_size

        with torch.set_grad_enabled(False):
            # [n, k, k]
            y_onehot = one_hot(y, n_class).view(x.shape[0], -1, n_class)

            y_idx = y.type(torch.long)

            # update memory keypoints
            # [n, k, d]
            if self.classification:
                count_label = torch.bincount(img_label, minlength=len(CATEGORIES))
                label_weight_onehot = fun_label_onehot(img_label, count_label)
                get = torch.matmul(label_weight_onehot.transpose(0, 1), (x[:, 0:self.single_cate_pos, :] * visible.type(x.dtype).view(*visible.shape, 1)).view(x.shape[0], -1))
                get = get.view(get.shape[0] ,-1, x.shape[-1])
                exist_cate = (count_label == 0).nonzero(as_tuple=True)[0]
                # handle case that one category has no instance in current batch
                for i in exist_cate:
                    # copy memory to get
                    get[i] = self.memory[i * self.single_cate_pos : (i + 1) * self.single_cate_pos]
                get = get.view(-1, x.shape[-1])
            else:
                get = x[:, 0:n_pos, :] * visible.type(x.dtype).view(*visible.shape, 1)
                get = torch.mean(get, dim=0)  # / torch.sum(visible, dim=0).view(-1, 1)

            if self.classification:
                clutter_start = self.single_cate_pos
            else:
                clutter_start = n_pos


            if n_neg > 0:
                if x.shape[0] > (self.nLem - n_pos) / n_neg:
                    self.memory = F.normalize(
                        torch.cat(
                            [
                                self.memory[0:n_pos, :] * momentum
                                + get * (1 - momentum),
                                x[:, clutter_start::, :]
                                .contiguous()
                                .view(-1, x.shape[2])[0 : self.memory.shape[0] - n_pos],
                            ],
                            dim=0,
                        ),
                        dim=1,
                        p=2,
                    )
                else:
                    neg_parts = torch.cat(
                        [
                            self.memory[
                                n_pos : n_pos + self.lru * n_neg * x.shape[0], :
                            ],
                            x[:, clutter_start::, :].contiguous().view(-1, x.shape[2]),
                            self.memory[
                                n_pos + (self.lru + 1) * n_neg * x.shape[0] : :, :
                            ],
                        ],
                        dim=0,
                    )

                    self.memory = F.normalize(
                        torch.cat(
                            [
                                self.memory[0:n_pos, :] * momentum
                                + get * (1 - momentum),
                                neg_parts,
                            ],
                            dim=0,
                        ),
                        dim=1,
                        p=2,
                    )
            else:
                self.memory = F.normalize(
                    self.memory[0:n_pos, :] * momentum + get * (1 - momentum),
                    dim=1,
                    p=2,
                )

            self.accumulate_num += torch.sum(
                (visible > 0)
                .type(self.accumulate_num.dtype)
                .to(self.accumulate_num.dtype),
                dim=0,
            )
            self.lru += 1
            self.lru = self.lru % self.max_lru
        # out.shape: [d, n_neg + n_pos]
        return similarity, y_idx, noise_similarity

    def forward_local(self, x, y, visible):
        # print('x',x.shape)
        # print('noise',self.num_noise)
        n_pos = self.num_pos
        n_neg = self.num_noise

        if self.max_lru == -1 and n_neg > 0:
            self.max_lru = (self.memory.shape[0] - n_pos) // (n_neg * x.shape[0])

        group_size = int(self.params[0].item())
        momentum = self.params[3].item()

        assert group_size == 1, "Currently only support group size = 1"

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
            noise_similarity = torch.matmul(
                x[:, n_pos:, :], torch.transpose(self.memory[0:n_pos, :], 0, 1)
            )

        n_class = n_pos // group_size

        with torch.set_grad_enabled(False):
            # [n, k, k]
            y_onehot = one_hot(y, n_class).view(x.shape[0], -1, n_class)

            if not group_size == 1:
                # [n, k, k * g]
                y_onehot = (
                    y_onehot.unsqueeze(2)
                    .expand(-1, -1, group_size)
                    .contiguous()
                    .view(x.shape[0], -1, n_pos)
                )
                visible = (
                    visible.unsqueeze(2)
                    .expand(-1, -1, group_size)
                    .contiguous()
                    .view(x.shape[0], -1, n_pos)
                )

                y_idx = torch.argmax(
                    similarity[:, :, 0:n_pos] + y_onehot * 2, dim=2
                ).type(torch.long)
            else:
                y_idx = y.type(torch.long)

            # update memory keypoints
            # [n, k, d]
            get = torch.bmm(
                torch.transpose(y_onehot, 1, 2),
                x[:, 0:n_pos, :] * visible.type(x.dtype).view(*visible.shape, 1),
            )

            # [k, d]
            get = torch.mean(get, dim=0)  # / torch.sum(visible, dim=0).view(-1, 1)

            self.accumulate_num += torch.sum(
                visible.type(self.accumulate_num.dtype), dim=0
            )
            self.lru += 1
            self.lru = self.lru % self.max_lru

        # out.shape -- torch.Size([128, 4097])
        return similarity, y_idx, noise_similarity

    def set_zero(self, n_pos):
        self.accumulate_num = torch.zeros(
            n_pos, dtype=torch.long, device=self.memory.device
        )
        self.memory.fill_(0)

    def accumulate_memory(self, x, y, visible, eps=1e-8):
        # print(self.params)
        group_size = int(self.params[0].item())

        # Currently only support group size = 1
        assert group_size == 1, "Currently only support group size = 1"

        n_pos = self.num_pos
        n_neg = self.num_noise

        with torch.no_grad():
            # print(visible.shape)

            # update memory keypoints
            # [n, k, k]
            idx_onehot = one_hot(y, n_pos).view(x.shape[0], -1, n_pos)

            # [k, d]
            get = torch.bmm(
                torch.transpose(idx_onehot, 1, 2),
                x[:, 0:n_pos, :] * visible.type(x.dtype).view(*visible.shape, 1),
            )
            get = torch.sum(get, dim=0)

            self.memory[0:n_pos, :].copy_(self.memory[0:n_pos, :] + get)
            self.accumulate_num += torch.sum(
                visible.type(self.accumulate_num.dtype), dim=0
            )

    def compute_feature_dist(self, x, vis, loss_foo=torch.nn.functional.mse_loss):
        return (loss_foo(F.normalize(x, p=2, dim=-1), self.memory[0:x.shape[0], None], reduce=False).sum(-1)[..., :vis.shape[-1]] * vis).mean()
        
    def normalize_memory(self):
        self.memory.copy_(F.normalize(self.memory, p=2, dim=1))

    def cuda(self, device=None):
        super().cuda(device)
        self.accumulate_num = self.accumulate_num.cuda(device)
        self.memory = self.memory.cuda(device)
        return self


class StaticLatentMananger():
    def __init__(self, n_latents, to_device='cuda', store_device='cpu'):
        self.latent_set = [dict() for _ in range(n_latents)]
        self.to_device = to_device
        self.store_device = store_device
        self.n_latent = n_latents

    def save_latent(self, names, *args):
        out_all = []
        for k in range(self.n_latent):
            out = []
            for i, name_ in enumerate(names):
                if torch.isnan(args[k][i].max()).item():
                    out.append(self.latent_set[k][name_].to(args[k].device))
                    continue
                self.latent_set[k][name_] = args[k][i].detach().to(self.store_device)
                out.append(args[k][i])
            out_all.append(torch.stack(out))
        return tuple(out_all)

    def get_latent(self, names, *default_value):
        out = [[] for _ in range(self.n_latent)]
        for i, name_ in enumerate(names):
            for k in range(self.n_latent):
                if name_ in self.latent_set[k].keys():
                    out[k].append(self.latent_set[k][name_].to(self.to_device))
                else:
                    out[k].append(default_value[k][i])
        return tuple([torch.stack(t) for t in out])

    def get_latent_without_default(self, names, ):
        out = [[] for _ in range(self.n_latent)]
        for i, name_ in enumerate(names):
            for k in range(self.n_latent):
                if name_ in self.latent_set[k].keys():
                    out[k].append(self.latent_set[k][name_].to(self.to_device))
                else:
                    return None
                
        return tuple([torch.stack(t) for t in out])
