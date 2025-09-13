import argparse


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = bool
        if isinstance(v, list):
            parser.add_argument(f"-{k}", default=v, type=int, nargs='*')
        elif isinstance(v, bool) and v:
            parser.add_argument(f"-{k}", action='store_false')
        elif isinstance(v, bool) and not v:
            parser.add_argument(f"-{k}", action='store_true')
        else:
            parser.add_argument(f"-{k}", default=v, type=v_type)


def create_argparser():
    defaults = dict(
        log_path='./log', model_path='./save_model',
        d='atustc',
        v=1, gpu=0, test=False,
        wait=True, t=0.0, mb=100,
        drop=0.2, stride=16,
        ncls=6, said=False, hdw=False, moae=False,
        optim='sgd', lr=0.008,
        wd=0.0005, dwd=True,
        dlr=10, dlrl=[10, 11, 12],
        clip=20, warmup_loss=999,
        warmup_rate=0.1, warmup_epoch=10,
        scheduler_rate=0.1, scheduler_epoch=[],
        cosmin_rate=0.002, max_epoch=120,
        era=0.4, gray=0.0,
        ih=256, iw=128,
        p=8, k=8, n=200, sample='m',
        md1=[1, 3, 2, 0], md2=[1, 3, 0, 0],
        d1=0.6, d2=6,
        seed=0, workers=4,
        test_epoch=999, last_test=0, 
        test_batch=256, flip=False, nfeature=6
    )
    parser = argparse.ArgumentParser(description='ReID')
    add_dict_to_argparser(parser, defaults)
    return parser
