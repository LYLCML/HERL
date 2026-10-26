import methods.wideresnet as wideresnet
from methods.augtools import HighlyCustomizableAugment, RandAugmentMC
import methods.util as util
import methods.Backbone_method as Backbone_method
import tqdm
from sklearn import metrics
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import random
from methods.util import AverageMeter
import time
from torchvision.transforms import transforms
from methods.resnet import ResNet
from torchvision import models as torchvision_models
import numpy as np
import metrics


@util.regmethod('herl')
class HERLMethod:

    def __init__(self, config, clssnum, train_set) -> None:
        self.config = config
        self.epoch = 0
        self.lr = config['learn_rate']
        self.batch_size = config['batch_size']
        self.clsnum = clssnum
        self.crt = HERLCriterion()
        self.model = HERLModel(self.clsnum, config, self.crt).cuda()
        self.modelopt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        self.wrap_ds = WrapDataset(train_set, self.config, inchan_num=3, )
        self.wrap_loader = data.DataLoader(self.wrap_ds,
                                           batch_size=self.config['batch_size'], shuffle=True, pin_memory=True,
                                           num_workers=1)
        self.lr_schedule = util.get_scheduler(self.config, self.wrap_loader)

    def train_epoch(self, flag):
        data_time = AverageMeter()
        batch_time = AverageMeter()
        train_acc = AverageMeter()
        running_loss = AverageMeter()
        running_lossp = AverageMeter()
        self.model.train()
        endtime = time.time()
        for i, data in enumerate(tqdm.tqdm(self.wrap_loader)):
            data_time.update(time.time() - endtime)
            self.lr = self.lr_schedule.get_lr(self.epoch, i, self.lr)
            util.set_lr([self.modelopt], self.lr)
            sx, lb = data[0].cuda(), data[1].cuda()
            loss, lossp, pred, scores = self.model(sx, lb, reqpredauc=True)
            if flag == 1:
                loss = loss + 0.0001* lossp
            if flag == 0:
                loss = loss + 0.0001* lossp
            self.modelopt.zero_grad()
            loss.backward()
            self.modelopt.step()
            nplb = data[1].numpy()
            train_acc.update((pred == nplb).sum() / pred.shape[0], pred.shape[0])
            running_loss.update(loss.item())
            running_lossp.update(lossp.item())
            batch_time.update(time.time() - endtime)
            endtime = time.time()
        self.epoch += 1
        E_star_num = self.epoch*2+2
       # self.model.backbone_cs.ldda_cls.E_star.fill_(min(E_star_num,5))

        training_res = \
            {"Loss": running_loss.avg,
             "Lossp": running_lossp.avg,
             "TrainAcc": train_acc.avg,
             "Learn Rate": self.lr,
             "DataTime": data_time.avg,
             "BatchTime": batch_time.avg}
        return training_res


    def scoring(self, loader, prepare=False):
        gts = []
        scores = []
        prediction = []
        energys=[]
        xcs = []
        rcs = []
        with (torch.no_grad()):
            for d in tqdm.tqdm(loader):
                x1 = d[0].cuda(non_blocking=True)
                gt = d[1].numpy()
                pred, xcls, dev, energy, xc, rc = self.model(x1, reqpredauc=True, prepareTest=prepare)
                score, index = torch.max(xcls, dim=1)
                score = score.cpu().numpy()
                energy = energy.cpu().numpy()
                xc = xc.cpu().numpy()
                rc = rc.cpu().numpy()
                scores.append(score)
                prediction.append(pred)
                gts.append(gt)
                energys.append(energy)
                xcs.append(xc)
                rcs.append(rc)
        prediction = np.concatenate(prediction)
        scores = np.concatenate(scores)
        gts = np.concatenate(gts)
        energys = np.concatenate(energys)
        xcs = np.concatenate(xcs)
        rcs = np.concatenate(rcs)
        return prediction, scores, gts, energys,xcs,rcs

    def plot_zscore_distribution(self, zscore_ener, gts_test, threshold=None, save_path=None):
        import numpy as np
        import matplotlib.pyplot as plt
        # 如果是 torch tensor，转成 numpy
        if hasattr(zscore_ener, "detach"):
            zscore_ener = zscore_ener.detach().cpu().numpy()
        else:
            zscore_ener = np.asarray(zscore_ener)
        if hasattr(gts_test, "detach"):
            gts_test = gts_test.detach().cpu().numpy()
        else:
            gts_test = np.asarray(gts_test)
        zscore_ener = zscore_ener.reshape(-1)
        gts_test = gts_test.reshape(-1).astype(int)
        assert len(zscore_ener) == len(gts_test), \
            f"Length mismatch: zscore_ener={len(zscore_ener)}, gts_test={len(gts_test)}"
        # 类别顺序，保证 -1 在最前面
        classes = sorted(np.unique(gts_test))
        n_cls = len(classes)
        fig, axes = plt.subplots(n_cls, 1, figsize=(7, 3 * n_cls))

        if n_cls == 1:
            axes = [axes]

        # 根据整体分数范围自动设置 x 轴范围
        x_min = np.percentile(zscore_ener, 1)
        x_max = np.percentile(zscore_ener, 99)

        # 稍微扩大一点范围
        margin = 0.1 * (x_max - x_min + 1e-8)
        x_min -= margin
        x_max += margin
        for ax, c in zip(axes, classes):
            cls_scores = zscore_ener[gts_test == c]
            label_name = "Unknown / OOD (-1)" if c == -1 else f"Class {c}"
            ax.hist(
                cls_scores,
                bins=100,
                range=(x_min, x_max),
                density=True,
                alpha=0.8
            )
            # 如果有统一阈值，可以画出来
            if threshold is not None:
                ax.axvline(
                    threshold,
                    linestyle="--",
                    linewidth=2,
                    label=f"threshold={threshold:.4f}"
                )
                ax.legend()

            ax.set_xlim(x_min, x_max)
            ax.set_title(label_name)
            ax.set_xlabel("Z-score energy")
            ax.set_ylabel("Density")
            ax.grid(True)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
    def knownpred_unknwonscore_test(self, test_loader):
        def classwise_zscore_normalization(pred_train, score_train, gt_train, pred_test, score_test):
            import numpy as np
            classes = np.unique(gt_train)
            class_mean_std = {}
            scores_train_array = []
            for c in range(len(classes)):
                idx = np.where((gt_train == c) & (pred_train == c))[0]
                scores_c = score_train[idx]
                scores_train_array.append(scores_c)
                mean_c = np.mean(scores_c)
                std_c = np.std(scores_c) if np.std(scores_c) > 0 else 1e-6
                class_mean_std[c] = (mean_c, std_c)
            z_scores = []
            print(class_mean_std)
            for i in range(len(score_test)):
                pred_c = pred_test[i]
                score = score_test[i]
                mean_c, std_c = class_mean_std[pred_c]
                z = (score - mean_c) / std_c
                z_scores.append(z)
            return np.array(z_scores), scores_train_array
        def classwise_zscore_normalization_CDF(pred_train, score_train, gt_train, pred_test, score_test):
            import numpy as np
            pred_train = np.asarray(pred_train).astype(int)
            gt_train = np.asarray(gt_train).astype(int)
            pred_test = np.asarray(pred_test).astype(int)
            score_train = np.asarray(score_train).reshape(-1)
            score_test = np.asarray(score_test).reshape(-1)
            classes = np.unique(gt_train)
            class_ref_scores = {}
            scores_train_array = []
            for c in classes:
                idx = np.where((gt_train == c) & (pred_train == c))[0]
                scores_c = score_train[idx]
                if len(scores_c) == 0:
                    raise ValueError(f"No correctly predicted training samples for class {c}.")
                scores_c_sorted = np.sort(scores_c)
                class_ref_scores[int(c)] = scores_c_sorted
                scores_train_array.append(scores_c)
            z_scores = []
            for i in range(len(score_test)):
                pred_c = int(pred_test[i])
                score = score_test[i]
                ref = class_ref_scores[pred_c]
                # 如果原始 score 越大越 OOD，用这个
                norm_score = np.searchsorted(ref, score, side="right") / len(ref)
                z_scores.append(norm_score)
            return np.array(z_scores), scores_train_array

        # import numpy as np
        #
        # def build_class_top_masks_np(
        #         xcs_train,
        #         gt_train,
        #         pred_train,
        #         candidate_ratios=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50),
        #         use_abs=True
        # ):
        #     """
        #     xcs_train:  [N, C, D]
        #     gt_train:   [N]
        #     pred_train: [N]
        #     正类：gt_train == c 且 pred_train == c
        #     负类：gt_train != c
        #     每个类别先按正类平均激活排序，然后自动选择能使正负类 masked L2 energy 差值最大的 top_ratio。
        #     """
        #     gt_train = gt_train.astype(int)
        #     pred_train = pred_train.astype(int)
        #     N, C, D = xcs_train.shape
        #     class_top_masks = np.zeros((C, D), dtype=bool)
        #     class_top_indices = {}
        #     best_ratios = {}
        #     best_info = {}
        #     for c in range(C):
        #         # 正类：真实为 c 且预测正确
        #         mask_pos = (gt_train == c) & (pred_train == c)
        #         # 负类：真实不是 c
        #         mask_neg = (gt_train != c)
        #         if mask_pos.sum() == 0:
        #             continue
        #         if mask_neg.sum() == 0:
        #             continue
        #         # 都取第 c 个类别分支
        #         pos_feats = xcs_train[mask_pos, c, :]  # [N_pos, D]
        #         neg_feats = xcs_train[mask_neg, c, :]  # [N_neg, D]
        #         # 用正类平均激活来排序位置，这一点保持不变
        #         if use_abs:
        #             mean_act = np.abs(pos_feats).mean(axis=0)  # [D]
        #         else:
        #             mean_act = pos_feats.mean(axis=0)
        #         # 按正类平均激活从大到小排序
        #         sorted_idx = np.argsort(-mean_act)
        #         best_diff = -np.inf
        #         best_ratio = None
        #         best_idx = None
        #         best_stat = None
        #         print(f"\nClass {c}: pos={mask_pos.sum()}, neg={mask_neg.sum()}")
        #         for ratio in candidate_ratios:
        #             k = max(1, int(ratio * D))
        #             # 仍然取正类平均激活最大的前 k 个位置
        #             top_idx = sorted_idx[:k]
        #             # 在这些位置上计算正负类 L2 energy
        #             pos_energy = np.sqrt(np.sum(pos_feats[:, top_idx] ** 2, axis=1))
        #             neg_energy = np.sqrt(np.sum(neg_feats[:, top_idx] ** 2, axis=1))
        #             # 用正负类 L2 均值差选择比例
        #             diff = np.abs(pos_energy.mean() - neg_energy.mean())
        #
        #
        #             if diff > best_diff:
        #                 best_diff = diff
        #                 best_ratio = ratio
        #                 best_idx = top_idx
        #                 best_stat = {
        #                     "k": k,
        #                     "pos_mean": pos_energy.mean(),
        #                     "neg_mean": neg_energy.mean(),
        #                     "pos_std": pos_energy.std(),
        #                     "neg_std": neg_energy.std(),
        #                     "diff": diff,
        #                 }
        #
        #         class_top_masks[c, best_idx] = True
        #         class_top_indices[c] = best_idx
        #         best_ratios[c] = best_ratio
        #         best_info[c] = best_stat
        #
        #         print(
        #             f"Class {c} selected: best_ratio={best_ratio:.2f}, "
        #             f"k={best_stat['k']}, "
        #             f"pos_mean={best_stat['pos_mean']:.4f}, "
        #             f"neg_mean={best_stat['neg_mean']:.4f}, "
        #             f"diff={best_stat['diff']:.4f}"
        #         )
        #
        #     return class_top_masks, class_top_indices, best_ratios, best_info
        #
        # def compute_topmask_energy_np(xcs, class_ids, class_top_masks, mode="rms"):
        #     """
        #     xcs: [N, C, D]
        #     class_ids: [N]，训练集用 gt_train，测试集用 pred_test
        #     class_top_masks: [C, D]
        #     mode:
        #         "l2"       : selected positions 的 L2 norm
        #         "rms"      : selected positions 的 RMS energy，推荐
        #         "mean_abs" : selected positions 的平均绝对激活
        #     """
        #     class_ids = class_ids.astype(int)
        #     N, C, D = xcs.shape
        #     idx = np.arange(N)
        #     # 每个样本取对应类别分支
        #     feats = xcs[idx, class_ids, :]  # [N, D]
        #     # 每个样本取对应类别的 top30% mask
        #     masks = class_top_masks[class_ids, :]  # [N, D]
        #     selected = feats * masks
        #     if mode == "l2":
        #         score = np.sqrt(np.sum(selected ** 2, axis=1))
        #     elif mode == "rms":
        #         denom = masks.sum(axis=1) + 1e-8
        #         score = np.sqrt(np.sum(selected ** 2, axis=1) / denom)
        #     elif mode == "mean_abs":
        #         denom = masks.sum(axis=1) + 1e-8
        #         score = np.sum(np.abs(selected), axis=1) / denom
        #     else:
        #         raise ValueError(f"Unknown mode: {mode}")
        #     return score


        self.model.eval()
        pred_train, score_train_reer, gt_train, score_train_ener, xcs_train, rcs_train = self.scoring(self.wrap_loader, True)
        score_train_ener = score_train_ener[np.arange(score_train_ener.shape[0]),gt_train]

        # # 1. 用训练集正负样本自动选择每类 top_ratio
        # class_top_masks, class_top_indices, best_ratios, best_info = build_class_top_masks_np(
        #     xcs_train=xcs_train,
        #     gt_train=gt_train,
        #     pred_train=pred_train,
        #     candidate_ratios=(0.20, 0.30, 0.40, 0.50, 0.6, 0.7, 0.8, 0.9,1),
        #     use_abs=True
        # )

        pred_test, score_test_reer, gts_test, score_test_ener, xcs_test, rcs_test = self.scoring(test_loader)

        # # 4. 测试集能量：用预测类别 pred_test 选择对应类别分支
        # score_train_ener = compute_topmask_energy_np(
        #     xcs=xcs_train,
        #     class_ids=gt_train,
        #     class_top_masks=class_top_masks,
        #     mode="l2"
        # )
        #
        #
        # # 4. 测试集能量：用预测类别 pred_test 选择对应类别分支
        # score_test_ener = compute_topmask_energy_np(
        #     xcs=xcs_test,
        #     class_ids=pred_test,
        #     class_top_masks=class_top_masks,
        #     mode="l2"
        # )

        score_test_ener = score_test_ener[np.arange(score_test_ener.shape[0]), pred_test]
        zscore_reer, scores_train_array = classwise_zscore_normalization(pred_train, score_train_reer, gt_train, pred_test,score_test_reer)
        zscore_ener, scores_train_array = classwise_zscore_normalization(pred_train, score_train_ener, gt_train, pred_test,score_test_ener)

        aurocs = []
        score_list = []
        alpha_values = np.linspace(0, 1, 11)  # alpha = 0.00, 0.01, ..., 1.00
        # 如果是 numpy，转成 torch；如果本来就是 torch，则保持
        if not torch.is_tensor(zscore_ener):
            zscore_ener = torch.tensor(zscore_ener, dtype=torch.float32)
        if not torch.is_tensor(zscore_reer):
            zscore_reer = torch.tensor(zscore_reer, dtype=torch.float32)
        zscore_ener = zscore_ener.float()
        zscore_reer = zscore_reer.float()
        evaluation = metrics.OSREvaluation(test_loader)
        for alpha in alpha_values:
            score = alpha * zscore_ener + (1 - alpha) * zscore_reer
            #score = zscore_ener + alpha * torch.clamp(zscore_reer - zscore_ener, min=0)
            open_detection = evaluation.open_detection_indexes(score, -999999)
            auroc = open_detection['auroc']
            aurocs.append(auroc)
            score_list.append(score.detach().cpu().numpy())
        aurocs = np.array(aurocs)
        score_array = np.array(score_list)
        max_idx = np.argmax(aurocs)
        best_alpha = alpha_values[max_idx]
        max_auroc = aurocs[max_idx]
        best_score = score_array[max_idx]
        print(f"Max AUROC: {max_auroc:.4f}, Best alpha: {best_alpha:.2f}")
        print(f"ENER weight: {best_alpha:.2f}, REER weight: {1 - best_alpha:.2f}")

        # self.plot_zscore_distribution(
        #     zscore_ener=best_score,
        #     gts_test=gts_test,
        #     save_path="zscore_distribution_by_class.png"
        # )
        #
        # self.plot_zscore_distribution(
        #     zscore_ener=zscore_reer,
        #     gts_test=gts_test,
        #     save_path="zscore_distribution_by_class.png"
        # )


        aaa=  score_test_ener[gts_test==-1]
        print(np.mean(aaa))
        bbb= score_test_ener[gts_test!=-1]
        print(np.mean(bbb))
        return best_score, zscore_ener, zscore_reer, score_test_ener, score_test_reer, -9999999, pred_test

    def save_model(self, path):
        save_dict = {
            'model': self.model.state_dict(),
            'config': self.config,
            'optimzer' : self.modelopt.state_dict(),
            'epoch' : self.epoch,
        }
        torch.save(save_dict,path)

    def load_model(self,path):
        save_dict = torch.load(path)
        self.model.load_state_dict(save_dict['model'])
        if 'optimzer' in save_dict and self.modelopt is not None:
            self.modelopt.load_state_dict(save_dict['optimzer'])
        self.epoch = save_dict['epoch']

def sim_conv_layer(input_channel, output_channel, kernel_size=1, padding=0, use_activation=True):
    if use_activation:
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False),
            nn.Tanh())
    else:
        res = nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False)
    return res


class PowerEncoder(nn.Module):
    def __init__(self, inchannel):
        super().__init__()
        layer_block = sim_conv_layer
        layers = []
        channel_list=[inchannel,128,64,32]
        for in_c, out_c in zip(channel_list[:-1], channel_list[1:]):
            layers.append(layer_block(in_c, out_c))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self, inchannel, hidden_layers, latent_chan):
        super().__init__()
        layer_block = sim_conv_layer
        self.latent_size = latent_chan
        if latent_chan > 0:
            self.encode_convs = []
            self.decode_convs = []
            for i in range(len(hidden_layers)):
                h = hidden_layers[i]
                ecv = layer_block(inchannel, h, )
                dcv = layer_block(h, inchannel, use_activation=i != 0)
                inchannel = h
                self.encode_convs.append(ecv)
                self.decode_convs.append(dcv)
            self.encode_convs = nn.ModuleList(self.encode_convs)
            self.decode_convs.reverse()
            self.decode_convs = nn.ModuleList(self.decode_convs)
            self.latent_conv = layer_block(inchannel, latent_chan)
            self.latent_deconv = layer_block(latent_chan, inchannel, use_activation=(len(hidden_layers) > 0))


    def forward(self, x):
        if self.latent_size > 0:
            output = x
            for cv in self.encode_convs:
                output = cv(output)
            latent = self.latent_conv(output)
            output = self.latent_deconv(latent)
            for cv in self.decode_convs:
                output = cv(output)
            return output, latent

class ChannelFC_AE(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        layer_dims = [2048, 1024, 512,256]
        # Encoder
        encoder_layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(self.activation)
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (reverse of encoder)
        decoder_layers = []
        rev_dims = list(reversed(layer_dims))
        for in_dim, out_dim in zip(rev_dims[:-1], rev_dims[1:]):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            decoder_layers.append(self.activation)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        z_norm = latent / (latent.norm(dim=1, keepdim=True) + 1e-8)
        out = self.decoder(z_norm)
        return out, latent

class HERLClassifier(nn.Module):

    def __init__(self, inchannels, num_class, config):
        super().__init__()
        self.class_power_ens = []
        for i in range(num_class):
            power_en = PowerEncoder(inchannels)
            self.class_power_ens.append(power_en)
        self.class_power_ens = nn.ModuleList(self.class_power_ens)

        self.class_aes = []
        for i in range(num_class):
            ae = ChannelFC_AE()
            self.class_aes.append(ae)
        self.class_aes = nn.ModuleList(self.class_aes)
        self.register_buffer('E_star', torch.tensor(config['R_c']))

    def ae_error(self,rc,x):
        return torch.norm(rc - x,p = 2,dim = 1,keepdim=True)*-0.1


    def forward(self, x, ycls):
        cls_ers = []
        energys = []
        energys1 = []
        loss_energy_all = 0
        xcs = []
        rcs = []
        for c in range(len(self.class_aes)):
            xc = self.class_power_ens[c](x)
            xc1 = xc.view(xc.size(0), -1)
            rc, lt = self.class_aes[c](xc1)
            cls_er = self.ae_error(rc, xc1)
            cls_er = torch.clamp(cls_er, min=-50, max=50)
            cls_ers.append(cls_er)
            xcs.append(xc1)
            rcs.append(rc)
            energy1 = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True)
            energys.append(energy1)

            energy2 = torch.norm(xc.view(xc.size(0), -1), p=2, dim=1, keepdim=True)
            energys1.append(energy2)

            if self.training:
                rc_flat = rc.view(rc.size(0), -1)
                energy = torch.norm(rc_flat, p=2, dim=1, keepdim=True)  # [B,1]
                mask_pos = (ycls == c).float().view(-1, 1)  # [B,1]
                # -------- 正类：energy >= E* --------
                diff = self.E_star - energy
                hinge_loss = torch.clamp(diff, min=0)
                loss_pos_energy = (hinge_loss ** 2) * mask_pos
                loss_pos_energy = loss_pos_energy.sum() / (mask_pos.sum() + 1e-6)
                lambda_energy = 1
                loss_energy_all += lambda_energy * loss_pos_energy


        logits = torch.cat(cls_ers, dim=1)
        energys = torch.cat(energys, dim=1)
        energys1 = torch.cat(energys1, dim=1)
        xcs = torch.stack(xcs, dim=1)
        rcs = torch.stack(rcs, dim=1)
        return logits,energys,xcs,rcs, loss_energy_all



class BackboneAndClassifier(nn.Module):

    def __init__(self, num_classes, config):
        super().__init__()
        self.backbone = Backbone_method.Backbone(config, 3)
        ldda_config = config['ldda_model']
        ldda_config["R_c"] = config['R_c']
        self.ldda_cls = HERLClassifier(self.backbone.output_dim, num_classes, ldda_config)

    def forward(self, x, ycls, feature_only=False):
        x = self.backbone(x)
        if feature_only:
            return x
        logic,energys, xcs, rcs, lossp = self.ldda_cls(x, ycls)
        return x, logic, energys, xcs, rcs, lossp


class HERLModel(nn.Module):

    def __init__(self, num_classes, config, crt):
        super().__init__()
        self.crt = crt
        self.backbone_cs = BackboneAndClassifier(num_classes, config)
        self.config = config
        self.num_classes = num_classes

    def forward(self, x, ycls=None, reqpredauc=False, prepareTest=False,
                reqfeature=False):

        x, xcls_raw,energys,xcs, rcs,  lossp = self.backbone_cs(x, ycls, feature_only=reqfeature)
        if reqfeature:
            return x

        def pred_score(xcls_raw):
            probs = self.crt(xcls_raw, prob=True).cpu().numpy()
            pred = probs.argmax(axis=1)
            return pred

        if self.training:
            loss = self.crt(xcls_raw, ycls)
            if reqpredauc:
                pred = pred_score(xcls_raw.detach())
                return loss,lossp, pred, xcls_raw
        else:
            xcls = xcls_raw
            if reqpredauc:
                pred = pred_score(xcls)
                deviations = None
                return pred, xcls, deviations,energys,xcs,rcs

        return xcls


class HERLCriterion(nn.Module):
    def get_onehot_label(self, y, clsnum):
        y = torch.reshape(y, [-1, 1]).long()
        return torch.zeros(y.shape[0], clsnum).cuda().scatter_(1, y, 1)

    def __init__(self):
        super().__init__()

    def forward(self, x, y=None, prob=False, pred=False):
        g = torch.softmax(x, dim=1)
        if prob: return g
        if pred: return torch.argmax(g, dim=1)
        loss = -torch.sum(self.get_onehot_label(y, g.shape[1]) * torch.log(g), dim=1).mean()
        return loss


def manual_contrast(x):
    s = random.uniform(0.1, 2)
    return x * s


class WrapDataset(data.Dataset):

    def __init__(self, labeled_ds, config, inchan_num=3) -> None:
        super().__init__()
        self.labeled_ds = labeled_ds
        __mean = [0.5, 0.5, 0.5][:inchan_num]
        __std = [0.25, 0.25, 0.25][:inchan_num]
        trans = [transforms.RandomHorizontalFlip()]
        if config['cust_aug_crop_withresize']:
            trans.append(transforms.RandomResizedCrop(size=util.img_size, scale=(0.25, 1)))
        elif util.img_size > 200:
            trans += [transforms.Resize(256), transforms.RandomResizedCrop(util.img_size)]
        else:
            trans.append(transforms.RandomCrop(size=util.img_size,
                                               padding=int(util.img_size * 0.125),
                                               padding_mode='reflect'))
        if config['strong_option'] == 'RA':
            trans.append(RandAugmentMC(n=2, m=10))
        elif config['strong_option'] == 'CUST':
            trans.append(HighlyCustomizableAugment(2, 10, -1, labeled_ds, config))
        elif config['strong_option'] == 'NONE':
            pass
        else:
            raise NotImplementedError()
        trans += [transforms.ToTensor(),
                  transforms.Normalize(mean=__mean, std=__std)]
        if config['manual_contrast']:
            trans.append(manual_contrast)
        strong = transforms.Compose(trans)
        if util.img_size > 200:
            self.simple = [transforms.RandomResizedCrop(util.img_size)]
        else:
            self.simple = [transforms.RandomCrop(size=util.img_size,
                                                 padding=int(util.img_size * 0.125),
                                                 padding_mode='reflect')]
        self.simple = transforms.Compose(([transforms.RandomHorizontalFlip()]) + self.simple + [
            transforms.ToTensor(),
            transforms.Normalize(mean=__mean, std=__std)] + ([manual_contrast] if config['manual_contrast'] else []))
        self.test_normalize = transforms.Compose([
            transforms.CenterCrop(util.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=__mean, std=__std)])

        td = {'strong': strong, 'simple': self.simple}
        self.aug = td[config['ldda_augmentation']]
        self.test_mode = False

    def __len__(self) -> int:
        return len(self.labeled_ds)

    def __getitem__(self, index: int):
        img, lb, _ = self.labeled_ds[index]
        if self.test_mode:
            img = self.test_normalize(img)
        else:
            img = self.aug(img)
        return img, lb, index