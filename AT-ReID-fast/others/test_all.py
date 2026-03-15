import time

import numpy as np

from others.eval import evaluate_bundle_head, extract_sample_feature_bundles
from others.runtime import get_device, select_amp_dtype
from others.transforms import get_transform


def _limit_samples(samples, limit):
    if limit and limit > 0:
        return samples[:limit]
    return samples


def _dataset_head_index(dataset_name, args):
    if int(getattr(args, "nfeature", 6)) == 1:
        return 0
    if dataset_name in ['market', 'cuhk', 'msmt', 'prcc_sc', 'vc_sc']:
        return 0
    if dataset_name in ['prcc', 'ltcc', 'deepchange', 'vc', 'ltcc_all']:
        return 1
    return 4


def _select_query_gallery(dataset_name, dataset, trial):
    if dataset_name in ['market', 'cuhk', 'msmt', 'deepchange', 'ltcc_all', 'vc', 'vc_sc']:
        return np.array(dataset.query), np.array(dataset.gallery)
    if dataset_name == 'sysu':
        return np.array(dataset.query), np.array(dataset.gallery_all_single[trial])
    if dataset_name == 'sysu_indoor':
        return np.array(dataset.query), np.array(dataset.gallery_indoor_single[trial])
    if dataset_name == 'sysu_multi':
        return np.array(dataset.query), np.array(dataset.gallery_all_multi)
    if dataset_name == 'sysu_indoor_multi':
        return np.array(dataset.query), np.array(dataset.gallery_indoor_multi)
    if dataset_name == 'regdb_v2i':
        return np.array(dataset.test_ir[trial + 1]), np.array(dataset.test_rgb[trial + 1])
    if dataset_name == 'llcm_v2i':
        return np.array(dataset.q_ir), np.array(dataset.g_rgb[trial])
    if dataset_name == 'regdb_i2v':
        return np.array(dataset.test_rgb[trial + 1]), np.array(dataset.test_ir[trial + 1])
    if dataset_name == 'llcm_i2v':
        return np.array(dataset.q_rgb), np.array(dataset.g_ir[trial])
    if dataset_name == 'prcc':
        return np.array(dataset.query_cc), np.array(dataset.gallery)
    if dataset_name == 'prcc_sc':
        return np.array(dataset.query_sc), np.array(dataset.gallery)
    if dataset_name == 'ltcc':
        return np.array(dataset.query_cc), np.array(dataset.gallery_cc)
    raise ValueError(f'Unsupported test dataset: {dataset_name}')


def _dataset_modes(dataset_name):
    if dataset_name in ['sysu', 'sysu_indoor', 'sysu_multi', 'sysu_indoor_multi',
                        'regdb_v2i', 'regdb_i2v', 'llcm_v2i', 'llcm_i2v']:
        mode1 = 'cm'
    else:
        mode1 = 'vm'

    if dataset_name in ['prcc', 'ltcc', 'deepchange', 'vc']:
        mode2 = 'cc'
    elif dataset_name in ['prcc_sc', 'vc_sc']:
        mode2 = 'sc'
    else:
        mode2 = 'all'
    return mode1, mode2


def _dataset_trials(dataset_name):
    if dataset_name in ['sysu', 'sysu_indoor', 'llcm_v2i', 'llcm_i2v', 'regdb_v2i', 'regdb_i2v']:
        return 10
    return 1


def test_1(args, d='market', dataset=None, model=None):
    start = time.time()
    trials = _dataset_trials(d)
    mode1, mode2 = _dataset_modes(d)
    _, transform_test = get_transform(args, test_pre_resized=str(getattr(args, "decode_cache", "off")).strip().lower() in {"ram", "disk"})
    amp_dtype = select_amp_dtype(args.amp_dtype)
    device = get_device()
    feature_cache = {}
    head_index = _dataset_head_index(d, args)

    cmc = None
    mAP = None
    for trial in range(trials):
        query, gallery = _select_query_gallery(d, dataset, trial)
        bundles = extract_sample_feature_bundles(
            model,
            {
                "query": _limit_samples(query, args.limit_query),
                "gallery": _limit_samples(gallery, args.limit_gallery),
            },
            transform_test,
            args,
            amp_dtype=amp_dtype,
            device=device,
            feature_cache=feature_cache,
        )
        query_bundle = bundles["query"]
        gallery_bundle = bundles["gallery"]
        cmc_t, mAP_t = evaluate_bundle_head(
            query_bundle,
            gallery_bundle,
            head_index=head_index,
            mode1=mode1,
            mode2=mode2,
            args=args,
            default_device=device,
            dataset=d,
        )
        if trial == 0:
            cmc = cmc_t / trials
            mAP = mAP_t / trials
        else:
            cmc = cmc + cmc_t / trials
            mAP = mAP + mAP_t / trials
    print(f'mAP:{mAP * 100:-2.2f}\t rank1{cmc[0] * 100:-2.2f}\t rank5{cmc[4] * 100:-2.2f}\t rank10{cmc[9] * 100:-2.2f}\t rank20{cmc[19] * 100:-2.2f}')
    print(f'Evaluation Time:\t {time.time() - start:.3f}')
    return cmc, mAP


def test_all(args, d='market', dataset=None, model=None):
    if d == 'sysu':
        print('sysu-single-all')
        cmc, mAP = test_1(args, d='sysu', dataset=dataset, model=model)
        print('sysu-multi-all')
        test_1(args, d='sysu_multi', dataset=dataset, model=model)
        print('sysu-single-indoor')
        test_1(args, d='sysu_indoor', dataset=dataset, model=model)
        print('sysu-multi-indoor')
        test_1(args, d='sysu_indoor_multi', dataset=dataset, model=model)
    elif d == 'llcm':
        print('llcm-v-to-i')
        cmc1, mAP1 = test_1(args, d='llcm_v2i', dataset=dataset, model=model)
        print('llcm-i-to-v')
        cmc2, mAP2 = test_1(args, d='llcm_i2v', dataset=dataset, model=model)
        cmc = cmc1 / 2 + cmc2 / 2
        mAP = mAP1 / 2 + mAP2 / 2
    elif d == 'regdb':
        print('regdb-v-to-i')
        cmc1, mAP1 = test_1(args, d='regdb_v2i', dataset=dataset, model=model)
        print('regdb-i-to-v')
        cmc2, mAP2 = test_1(args, d='regdb_i2v', dataset=dataset, model=model)
        cmc = cmc1 / 2 + cmc2 / 2
        mAP = mAP1 / 2 + mAP2 / 2
    elif d == 'prcc':
        print('prcc-sc')
        test_1(args, d='prcc_sc', dataset=dataset, model=model)
        print('prcc-cc')
        cmc, mAP = test_1(args, d='prcc', dataset=dataset, model=model)
    elif d == 'ltcc':
        print('ltcc-all')
        test_1(args, d='ltcc_all', dataset=dataset, model=model)
        print('ltcc-cc')
        cmc, mAP = test_1(args, d='ltcc', dataset=dataset, model=model)
    elif d == 'vc':
        print('vc-sc')
        test_1(args, d='vc_sc', dataset=dataset, model=model)
        print('vc-cc')
        cmc, mAP = test_1(args, d='vc', dataset=dataset, model=model)
    else:
        cmc, mAP = test_1(args, d=d, dataset=dataset, model=model)
    return cmc, mAP
