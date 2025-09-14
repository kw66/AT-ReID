import time
from others.eval import eval, extract_feature, extract_feature6
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset.datamanager import ImageDataset
from others.transforms import get_transform
import torch


def test_1(args, d='market', dataset=None, model=None):
    start = time.time()
    t = 1
    if d in ['sysu', 'sysu_indoor', 'sysu_multi', 'sysu_indoor_multi',
             'regdb_v2i', 'regdb_i2v', 'llcm_v2i', 'llcm_i2v']:
        mode1 = 'cm'
    else:
        mode1 = 'vm'
    if d in ['prcc', 'ltcc', 'deepchange', 'vc']:
        mode2 = 'cc'
    elif d in ['prcc_sc', 'vc_sc']:
        mode2 = 'sc'
    else:
        mode2 = 'all'
    if d in ['sysu', 'sysu_indoor', 'llcm_v2i', 'llcm_i2v', 'regdb_v2i', 'regdb_i2v']:
        t = 10
    for trial in range(t):
        if d in ['market', 'cuhk', 'msmt', 'deepchange', 'ltcc_all', 'vc', 'vc_sc']:
            query = np.array(dataset.query)
            gallery = np.array(dataset.gallery)
        if d == 'sysu':
            query = np.array(dataset.query)
            gallery = np.array(dataset.gallery_all_single[trial])
        if d == 'sysu_indoor':
            query = np.array(dataset.query)
            gallery = np.array(dataset.gallery_indoor_single[trial])
        if d == 'sysu_multi':
            query = np.array(dataset.query)
            gallery = np.array(dataset.gallery_all_multi)
        if d == 'sysu_indoor_multi':
            query = np.array(dataset.query)
            gallery = np.array(dataset.gallery_indoor_multi)
        if d == 'regdb_v2i':
            query = np.array(dataset.test_ir[trial+1])
            gallery = np.array(dataset.test_rgb[trial+1])
        if d == 'llcm_v2i':
            query = np.array(dataset.q_ir)
            gallery = np.array(dataset.g_rgb[trial])
        if d == 'regdb_i2v':
            query = np.array(dataset.test_rgb[trial+1])
            gallery = np.array(dataset.test_ir[trial+1])
        if d == 'llcm_i2v':
            query = np.array(dataset.q_rgb)
            gallery = np.array(dataset.g_ir[trial])
        if d == 'prcc':
            query = np.array(dataset.query_cc)
            gallery = np.array(dataset.gallery)
        if d == 'prcc_sc':
            query = np.array(dataset.query_sc)
            gallery = np.array(dataset.gallery)
        if d == 'ltcc':
            query = np.array(dataset.query_cc)
            gallery = np.array(dataset.gallery_cc)
        transform_train, transform_test = get_transform(args)
        queryloader = DataLoader(
            ImageDataset(query, transform=transform_test),
            batch_size=args.test_batch, num_workers=args.workers,
            pin_memory=True, drop_last=False, )
        galleryloader = DataLoader(
            ImageDataset(gallery, transform=transform_test),
            batch_size=args.test_batch, num_workers=args.workers,
            pin_memory=True, drop_last=False, )
        if args.nfeature == 1:
            q, q_pids, q_cids, q_mids, q_camids = extract_feature(model, queryloader, args.flip)
            g, g_pids, g_cids, g_mids, g_camids = extract_feature(model, galleryloader, args.flip)
            distmat = 1 - torch.mm(F.normalize(q, p=2, dim=1), F.normalize(g, p=2, dim=1).t()).numpy()
        if args.nfeature == 6:
            if d in ['market', 'cuhk', 'msmt', 'prcc_sc', 'vc_sc']:
                q, _, _, _, _, _, q_pids, q_cids, q_mids, q_camids = extract_feature6(model, queryloader, args.flip)
                g, _, _, _, _, _, g_pids, g_cids, g_mids, g_camids = extract_feature6(model, galleryloader, args.flip)
            if d in ['prcc', 'ltcc', 'deepchange', 'vc', 'ltcc_all']:
                _, q, _, _, _, _, q_pids, q_cids, q_mids, q_camids = extract_feature6(model, queryloader, args.flip)
                _, g, _, _, _, _, g_pids, g_cids, g_mids, g_camids = extract_feature6(model, galleryloader, args.flip)
            if d in ['sysu', 'sysu_indoor', 'sysu_multi', 'sysu_indoor_multi', 'regdb_v2i', 'regdb_i2v', 'llcm_v2i', 'llcm_i2v']:
                _, _, _, _, q, _, q_pids, q_cids, q_mids, q_camids = extract_feature6(model, queryloader, args.flip)
                _, _, _, _, g, _, g_pids, g_cids, g_mids, g_camids = extract_feature6(model, galleryloader, args.flip)
        cmc_t, mAP_t = eval(q, q_pids, q_cids, q_mids, q_camids, g, g_pids, g_cids, g_mids, g_camids, mode1, mode2, distmat, d)
        if trial == 0:
            cmc, mAP = cmc_t / t, mAP_t / t
        else:
            cmc, mAP = cmc + cmc_t / t, mAP + mAP_t / t
    print(f'{mAP * 100:-2.2f}\t{cmc[0] * 100:-2.2f}\t{cmc[4] * 100:-2.2f}\t{cmc[9] * 100:-2.2f}\t{cmc[19] * 100:-2.2f}')
    print(f'Evaluation Time:\t {time.time() - start:.3f}')
    return cmc, mAP


def test_all(args, d='market', dataset=None, model=None):
    if d == 'sysu':
        cmc, mAP = test_1(args, d='sysu', dataset=dataset, model=model)
        _, _ = test_1(args, d='sysu_multi', dataset=dataset, model=model)
        _, _ = test_1(args, d='sysu_indoor', dataset=dataset, model=model)
        _, _ = test_1(args, d='sysu_indoor_multi', dataset=dataset, model=model)
    elif d == 'llcm':
        cmc1, mAP1 = test_1(args, d='llcm_v2i', dataset=dataset, model=model)
        cmc2, mAP2 = test_1(args, d='llcm_i2v', dataset=dataset, model=model)
        cmc = cmc1/2+cmc2/2
        mAP = mAP1/2+mAP2/2
    elif d == 'regdb':
        cmc1, mAP1 = test_1(args, d='regdb_v2i', dataset=dataset, model=model)
        cmc2, mAP2 = test_1(args, d='regdb_i2v', dataset=dataset, model=model)
        cmc = cmc1 / 2 + cmc2 / 2
        mAP = mAP1 / 2 + mAP2 / 2
    elif d == 'prcc':
        _, _ = test_1(args, d='prcc_sc', dataset=dataset, model=model)
        cmc, mAP = test_1(args, d='prcc', dataset=dataset, model=model)
    elif d == 'ltcc':
        _, _ = test_1(args, d='ltcc_all', dataset=dataset, model=model)
        cmc, mAP = test_1(args, d='ltcc', dataset=dataset, model=model)
    elif d == 'vc':
        _, _ = test_1(args, d='vc_sc', dataset=dataset, model=model)
        cmc, mAP = test_1(args, d='vc', dataset=dataset, model=model)
    else:
        cmc, mAP = test_1(args, d=d, dataset=dataset, model=model)
    return cmc, mAP








