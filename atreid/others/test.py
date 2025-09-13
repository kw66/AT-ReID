import time
from torch.utils.data import DataLoader
from dataset.datamanager import ImageDataset
from others.transforms import get_transform
import torch
from others.eval import eval, extract_feature, extract_feature6
from torch.nn import functional as F


def attest(args, dataset, model):
    start = time.time()
    transform_train, transform_test = get_transform(args)
    queryloader1 = DataLoader(
        ImageDataset(dataset.q_ad_st, transform=transform_test),
        batch_size=args.test_batch, num_workers=args.workers,
        pin_memory=True, drop_last=False, )
    galleryloader1 = DataLoader(
        ImageDataset(dataset.g_ad_st, transform=transform_test),
        batch_size=args.test_batch, num_workers=args.workers,
        pin_memory=True, drop_last=False, )
    queryloader2 = DataLoader(
        ImageDataset(dataset.q_ad_lt, transform=transform_test),
        batch_size=args.test_batch, num_workers=args.workers,
        pin_memory=True, drop_last=False, )
    galleryloader2 = DataLoader(
        ImageDataset(dataset.g_ad_lt, transform=transform_test),
        batch_size=args.test_batch, num_workers=args.workers,
        pin_memory=True, drop_last=False, )
    if args.nfeature == 1:
        q, q_pids, q_cids, q_mids, q_camids = extract_feature(model, queryloader1, args.flip)
        g, g_pids, g_cids, g_mids, g_camids = extract_feature(model, galleryloader1, args.flip)
        distmat = -torch.mm(F.normalize(q, p=2, dim=1), F.normalize(g, p=2, dim=1).t()).numpy()
        cmc1, mAP1 = eval(q, q_pids, q_cids, q_mids, q_camids, g, g_pids, g_cids, g_mids, g_camids, 'vm', 'sc', distmat)
        print(f'R-1: {cmc1[0]:.2%} | R-5: {cmc1[4]:.2%} | R-10: {cmc1[9]:.2%} | R-20: {cmc1[19]:.2%} | mAP: {mAP1:.2%}')
        cmc3, mAP3 = eval(q, q_pids, q_cids, q_mids, q_camids, g, g_pids, g_cids, g_mids, g_camids, 'im', 'sc', distmat)
        print(f'R-1: {cmc3[0]:.2%} | R-5: {cmc3[4]:.2%} | R-10: {cmc3[9]:.2%} | R-20: {cmc3[19]:.2%} | mAP: {mAP3:.2%}')
        cmc5, mAP5 = eval(q, q_pids, q_cids, q_mids, q_camids, g, g_pids, g_cids, g_mids, g_camids, 'cm', 'sc', distmat)
        print(f'R-1: {cmc5[0]:.2%} | R-5: {cmc5[4]:.2%} | R-10: {cmc5[9]:.2%} | R-20: {cmc5[19]:.2%} | mAP: {mAP5:.2%}')
        q, q_pids, q_cids, q_mids, q_camids = extract_feature(model, queryloader2)
        g, g_pids, g_cids, g_mids, g_camids = extract_feature(model, galleryloader2)
        distmat = -torch.mm(F.normalize(q, p=2, dim=1), F.normalize(g, p=2, dim=1).t()).numpy()
        cmc2, mAP2 = eval(q, q_pids, q_cids, q_mids, q_camids, g, g_pids, g_cids, g_mids, g_camids, 'vm', 'cc', distmat)
        print(f'R-1: {cmc2[0]:.2%} | R-5: {cmc2[4]:.2%} | R-10: {cmc2[9]:.2%} | R-20: {cmc2[19]:.2%} | mAP: {mAP2:.2%}')
        cmc4, mAP4 = eval(q, q_pids, q_cids, q_mids, q_camids, g, g_pids, g_cids, g_mids, g_camids, 'im', 'cc', distmat)
        print(f'R-1: {cmc4[0]:.2%} | R-5: {cmc4[4]:.2%} | R-10: {cmc4[9]:.2%} | R-20: {cmc4[19]:.2%} | mAP: {mAP4:.2%}')
        cmc6, mAP6 = eval(q, q_pids, q_cids, q_mids, q_camids, g, g_pids, g_cids, g_mids, g_camids, 'cm', 'cc', distmat)
        print(f'R-1: {cmc6[0]:.2%} | R-5: {cmc6[4]:.2%} | R-10: {cmc6[9]:.2%} | R-20: {cmc6[19]:.2%} | mAP: {mAP6:.2%}')
    if args.nfeature == 6:
        q1, q2, q3, q4, q5, q6, q_pids, q_cids, q_mids, q_camids = extract_feature6(model, queryloader1, args.flip)
        g1, g2, g3, g4, g5, g6, g_pids, g_cids, g_mids, g_camids = extract_feature6(model, galleryloader1, args.flip)
        distmat1 = -torch.mm(F.normalize(q1, p=2, dim=1), F.normalize(g1, p=2, dim=1).t()).numpy()

        cmc1, mAP1 = eval(q1, q_pids, q_cids, q_mids, q_camids, g1, g_pids, g_cids, g_mids, g_camids, 'vm', 'sc',
                          distmat1)
        print(f'R-1: {cmc1[0]:.2%} | R-5: {cmc1[4]:.2%} | R-10: {cmc1[9]:.2%} | R-20: {cmc1[19]:.2%} | mAP: {mAP1:.2%}')
        distmat3 = -torch.mm(F.normalize(q3, p=2, dim=1), F.normalize(g3, p=2, dim=1).t()).numpy()
        cmc3, mAP3 = eval(q3, q_pids, q_cids, q_mids, q_camids, g3, g_pids, g_cids, g_mids, g_camids, 'im', 'sc',
                          distmat3)
        print(f'R-1: {cmc3[0]:.2%} | R-5: {cmc3[4]:.2%} | R-10: {cmc3[9]:.2%} | R-20: {cmc3[19]:.2%} | mAP: {mAP3:.2%}')
        distmat5 = -torch.mm(F.normalize(q5, p=2, dim=1), F.normalize(g5, p=2, dim=1).t()).numpy()
        cmc5, mAP5 = eval(q5, q_pids, q_cids, q_mids, q_camids, g5, g_pids, g_cids, g_mids, g_camids, 'cm', 'sc',
                          distmat5)
        print(f'R-1: {cmc5[0]:.2%} | R-5: {cmc5[4]:.2%} | R-10: {cmc5[9]:.2%} | R-20: {cmc5[19]:.2%} | mAP: {mAP5:.2%}')
        q1, q2, q3, q4, q5, q6, q_pids, q_cids, q_mids, q_camids = extract_feature6(model, queryloader2)
        g1, g2, g3, g4, g5, g6, g_pids, g_cids, g_mids, g_camids = extract_feature6(model, galleryloader2)
        distmat2 = -torch.mm(F.normalize(q2, p=2, dim=1), F.normalize(g2, p=2, dim=1).t()).numpy()

        cmc2, mAP2 = eval(q2, q_pids, q_cids, q_mids, q_camids, g2, g_pids, g_cids, g_mids, g_camids, 'vm', 'cc',
                          distmat2)
        print(f'R-1: {cmc2[0]:.2%} | R-5: {cmc2[4]:.2%} | R-10: {cmc2[9]:.2%} | R-20: {cmc2[19]:.2%} | mAP: {mAP2:.2%}')
        distmat4 = -torch.mm(F.normalize(q4, p=2, dim=1), F.normalize(g4, p=2, dim=1).t()).numpy()
        cmc4, mAP4 = eval(q4, q_pids, q_cids, q_mids, q_camids, g4, g_pids, g_cids, g_mids, g_camids, 'im', 'cc',
                          distmat4)
        print(f'R-1: {cmc4[0]:.2%} | R-5: {cmc4[4]:.2%} | R-10: {cmc4[9]:.2%} | R-20: {cmc4[19]:.2%} | mAP: {mAP4:.2%}')
        distmat6 = -torch.mm(F.normalize(q6, p=2, dim=1), F.normalize(g6, p=2, dim=1).t()).numpy()
        cmc6, mAP6 = eval(q6, q_pids, q_cids, q_mids, q_camids, g6, g_pids, g_cids, g_mids, g_camids, 'cm', 'cc',
                          distmat6)
        print(f'R-1: {cmc6[0]:.2%} | R-5: {cmc6[4]:.2%} | R-10: {cmc6[9]:.2%} | R-20: {cmc6[19]:.2%} | mAP: {mAP6:.2%}')

    print(f'Evaluation Time:\t {time.time() - start:.3f}')
    cmc0 = (cmc1[0] + cmc2[0] + cmc3[0] + cmc4[0] + cmc5[0] + cmc6[0]) / 6
    mAP0 = (mAP1 + mAP2 + mAP3 + mAP4 + mAP5 + mAP6) / 6
    print(
        f'{cmc0 * 100:-2.2f}\t{cmc1[0] * 100:-2.2f}\t{cmc2[0] * 100:-2.2f}\t{cmc3[0] * 100:-2.2f}\t'
        f'{cmc4[0] * 100:-2.2f}\t{cmc5[0] * 100:-2.2f}\t{cmc6[0] * 100:-2.2f}\n'
        f'{mAP0 * 100:-2.2f}\t{mAP1 * 100:-2.2f}\t{mAP2 * 100:-2.2f}\t{mAP3 * 100:-2.2f}\t'
        f'{mAP4 * 100:-2.2f}\t{mAP5 * 100:-2.2f}\t{mAP6 * 100:-2.2f}')
    return cmc6, mAP6
