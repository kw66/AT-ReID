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
from dataset.ag import ag


def dataset_all(name='market'):
    if name == 'atustc':
        return atustc()
    if name == 'market':
        return market()
    if name == 'cuhk':
        return cuhk()
    if name == 'msmt':
        return msmt()
    if name == 'sysu':
        return sysu()
    if name == 'llcm':
        return llcm()
    if name == 'regdb':
        return regdb()
    if name == 'prcc':
        return prcc()
    if name == 'ltcc':
        return ltcc()
    if name == 'vc':
        return vc()
    if name == 'deepchange':
        return deepchange()
    if name == 'ag':
        return ag()


if __name__ == '__main__':
    dataset_all()