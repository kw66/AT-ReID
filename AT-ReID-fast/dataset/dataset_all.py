from dataset.atustc import atustc
from dataset.market import market
from dataset.cuhk import cuhk
from dataset.msmt import msmt
from dataset.sysu import sysu
from dataset.llcm import llcm
from dataset.regdb import regdb
from dataset.prcc import prcc
from dataset.ltcc import ltcc
from dataset.vc import vc
from dataset.deepchange import deepchange


def dataset_all(name='market', data_root=None, data_root_config=None):
    if name == 'atustc':
        return atustc(data_dir=data_root, data_root_config=data_root_config)
    if name == 'market':
        return market(data_dir=data_root, data_root_config=data_root_config)
    if name == 'cuhk':
        return cuhk(data_dir=data_root, data_root_config=data_root_config)
    if name == 'msmt':
        return msmt(data_dir=data_root, data_root_config=data_root_config)
    if name == 'sysu':
        return sysu(data_dir=data_root, data_root_config=data_root_config)
    if name == 'llcm':
        return llcm(data_dir=data_root, data_root_config=data_root_config)
    if name == 'regdb':
        return regdb(data_dir=data_root, data_root_config=data_root_config)
    if name == 'prcc':
        return prcc(data_dir=data_root, data_root_config=data_root_config)
    if name == 'ltcc':
        return ltcc(data_dir=data_root, data_root_config=data_root_config)
    if name == 'vc':
        return vc(data_dir=data_root, data_root_config=data_root_config)
    if name == 'deepchange':
        return deepchange(data_dir=data_root, data_root_config=data_root_config)


if __name__ == '__main__':

    dataset_all()
