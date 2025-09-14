import torch
import numpy as np
from torch.nn import functional as F


def extract_feature(model, dataloader, flip=False):
    model.eval()
    feat_s = torch.FloatTensor()
    pid_s, cid_s, mid_s, camid_s = [], [], [], []
    with torch.no_grad():
        for imgs, pids, cids, mids, camids, _ in dataloader:
            pid_s.extend(pids)
            cid_s.extend(cids)
            mid_s.extend(mids)
            camid_s.extend(camids)
            imgs = imgs.cuda()
            mids = torch.tensor([int(i) for i in mids]).cuda()
            feat = model(imgs)
            if flip:
                flip_imgs = torch.flip(imgs, [3])
                flip_imgs = flip_imgs.cuda()
                featf = model(flip_imgs)
                feat += featf
            feat_s = torch.cat((feat_s, feat.cpu()), 0)
    pid_s = np.asarray(pid_s, dtype=np.int64)
    cid_s = np.asarray(cid_s, dtype=np.int64)
    mid_s = np.asarray(mid_s, dtype=np.int64)
    camid_s = np.asarray(camid_s, dtype=np.int64)
    return feat_s, pid_s, cid_s, mid_s, camid_s


def extract_feature6(model, dataloader, flip=False):
    model.eval()
    f1s = torch.FloatTensor()
    f2s = torch.FloatTensor()
    f3s = torch.FloatTensor()
    f4s = torch.FloatTensor()
    f5s = torch.FloatTensor()
    f6s = torch.FloatTensor()
    pid_s, cid_s, mid_s, camid_s = [], [], [], []
    with torch.no_grad():
        for imgs, pids, cids, mids, camids, _ in dataloader:
            pid_s.extend(pids)
            cid_s.extend(cids)
            mid_s.extend(mids)
            camid_s.extend(camids)
            imgs = imgs.cuda()
            mids = torch.tensor([int(i) for i in mids]).cuda()
            f1, f2, f3, f4, f5, f6 = model(imgs)
            if flip:
                flip_imgs = torch.flip(imgs, [3])
                flip_imgs = flip_imgs.cuda()
                f1f, f2f, f3f, f4f, f5f, f6f = model(flip_imgs)
                f1 += f1f
                f2 += f2f
                f3 += f3f
                f4 += f4f
                f5 += f5f
                f6 += f6f
            f1s = torch.cat((f1s, f1.cpu()), 0)
            f2s = torch.cat((f2s, f2.cpu()), 0)
            f3s = torch.cat((f3s, f3.cpu()), 0)
            f4s = torch.cat((f4s, f4.cpu()), 0)
            f5s = torch.cat((f5s, f5.cpu()), 0)
            f6s = torch.cat((f6s, f6.cpu()), 0)
    pid_s = np.asarray(pid_s, dtype=np.int64)
    cid_s = np.asarray(cid_s, dtype=np.int64)
    mid_s = np.asarray(mid_s, dtype=np.int64)
    camid_s = np.asarray(camid_s, dtype=np.int64)
    return f1s, f2s, f3s, f4s, f5s, f6s, pid_s, cid_s, mid_s, camid_s


def eval(q, q_pids, q_cids, q_mids, q_camids,
         g, g_pids, g_cids, g_mids, g_camids,
         mode1, mode2, distmat=None, dataset=None):
    if distmat is None:
        distmat = -torch.mm(F.normalize(q, p=2, dim=1), F.normalize(g, p=2, dim=1).t()).numpy()
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)  # 每个query检索结果从小到大排序后在原矩阵的位置
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int64)  # g_pids[indices]排序后对应的g_pids,matches排序后的正确匹配
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        order = indices[q_idx]  # 每个query检索结果从小到大排序后在原矩阵的下标
        q_pid = q_pids[q_idx]
        q_cid = q_cids[q_idx]
        q_mid = q_mids[q_idx]
        q_camid = q_camids[q_idx]
        g_pid = g_pids[order]
        g_cid = g_cids[order]
        g_mid = g_mids[order]
        g_camid = g_camids[order]

        remove_cam = (q_pid == g_pid) & (q_camid == g_camid)#保留同相机负样本
        remove_sc = (q_pid == g_pid) & (q_cid != g_cid)
        remove_cc = (q_pid == g_pid) & (q_cid == g_cid)
        remove_vm = (q_mid != 1) | (g_mid != 1)
        remove_im = (q_mid != 2) | (g_mid != 2)
        remove_cm = ((q_mid == 1) & (g_mid == 1)) | ((q_mid == 2) & (g_mid == 2))#去掉同模态正负样本
        remove_sysu = ((q_camid == 3) & (g_camids[order] == 2))
        if dataset != 'deepchange':
            remove = remove_cam
        else:
            remove = remove_vm
        if dataset == 'sysu':
            remove = remove | remove_sysu

        if mode1 == 'vm':
            remove = remove | remove_vm
        if mode1 == 'im':
            remove = remove | remove_im
        if mode1 == 'cm':
            remove = remove | remove_cm
        if mode1 == 'all':
            remove = remove

        if mode2 == 'sc':
            remove = remove | remove_sc
        if mode2 == 'cc':
            remove = remove | remove_cc
        if mode2 == 'all':
            remove = remove

        keep = np.invert(remove)
        # compute cmc curve
        orig_cmc = matches[q_idx][keep]  # 排序后的正确匹配,根据条件去点一些gallery sample 【0，1，0，0，1，0】

        if len(orig_cmc) > 0:
            cmc, AP, num_valid = compute_cmc_ap(orig_cmc)
            if num_valid == 1:
                all_cmc.append(cmc)
                all_AP.append(AP)
                num_valid_q += num_valid
    print(num_valid_q)
    if num_valid_q == 0:
        return np.zeros(20), 0
    else:
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        return all_cmc, mAP


def compute_cmc_ap(orig_cmc, max_rank=20):
    if not np.any(orig_cmc):
        num_valid = 0
    else:
        num_valid = 1
    cmc = orig_cmc.cumsum()  # 按行累加 【0，1，1，1，2，2】
    cmc[cmc > 1] = 1  # 真正的cmc，但还没平均其实【0，1，1，1，1，1】
    # compute average precision
    num_rel = orig_cmc.sum()  # 有几个正确匹配 2
    tmp_cmc = orig_cmc.cumsum()  # 按行累加 【0，1，1，1，2，2】
    tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]  # P@K 【0/1，1/2，1/3，1/4，2/5，2/6】
    tmp_cmc = np.asarray(tmp_cmc) * orig_cmc  # recall变化的几个位置，【0，1/2，0，0，2/5，0】
    AP = tmp_cmc.sum() / num_rel.clip(1)  # 除以num_rel后就是delt recall，相当于矩形法计算面积
    return cmc[:max_rank], AP, num_valid


if __name__ == '__main__':
    pass
