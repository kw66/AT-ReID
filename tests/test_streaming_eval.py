"""Streaming ranking regression without importing training dependencies."""
import ast
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if (ROOT/'AT-ReID-fast').exists():
    ROOT = ROOT/'AT-ReID-fast'


def functions(path, names, scope):
    tree = ast.parse(path.read_text(encoding='utf8'))
    nodes = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope


E = functions(ROOT/'others/eval.py',
              {'_eval_torch','_as_torch_float_tensor','_as_torch_int_tensor',
               '_CosineDistanceBlocks','cosine_distance_matrix','eval','compute_cmc_ap'},
              {'torch':torch,'np':np,'F':F,'time':time})
R = functions(ROOT/'others/runtime.py',{'resolve_rank_device'},
              {'torch':torch,'_AUTO_GPU_METRIC_MIN_PAIRS':5000000})


class StreamingTests(unittest.TestCase):
    def test_provider_matches_dense_protocols(self):
        rng=np.random.default_rng(21)
        nq,ng=19,83
        q=[rng.integers(0,5,nq),rng.integers(0,3,nq),rng.integers(1,3,nq),rng.integers(1,7,nq)]
        g=[rng.integers(0,5,ng),rng.integers(0,3,ng),rng.integers(1,3,ng),rng.integers(1,7,ng)]
        D=rng.integers(-4,5,(nq,ng)).astype(np.float32) # deliberate ties
        for dataset in ('market','sysu','prcc','deepchange'):
            for mode1 in ('all','vm','im','cm'):
                for mode2 in ('all','sc','cc'):
                    args=(*q,*g,mode1,mode2)
                    dense=E['_eval_torch'](*args,D,dataset,device=torch.device('cpu'),sort_batch_size=4)
                    calls=[]
                    def blocks(a,b):
                        calls.append((a,b))
                        return torch.from_numpy(D[a:b])
                    streamed=E['_eval_torch'](*args,blocks,dataset,device=torch.device('cpu'),sort_batch_size=4)
                    np.testing.assert_array_equal(dense[0],streamed[0])
                    self.assertEqual(dense[1],streamed[1])
                    self.assertTrue(all(b-a<=4 for a,b in calls))
                    self.assertEqual(calls[-1][1],nq)

    def test_real_feature_distance_blocks(self):
        torch.manual_seed(8)
        q,g=torch.randn(13,32),torch.randn(61,32)
        stream=E['_CosineDistanceBlocks'](q,g,torch.device('cpu'))
        actual=torch.cat([stream(a,min(a+4,len(q))) for a in range(0,len(q),4)])
        expected=E['cosine_distance_matrix'](q,g,device=torch.device('cpu'),return_tensor=True)
        torch.testing.assert_close(actual,expected)
        self.assertGreater(stream.elapsed(),0)

    def test_large_dataset_uses_streaming_cuda(self):
        kwargs=dict(default_device=torch.device('cuda'),num_query=11659,num_gallery=82161,
                    max_cuda_elements=64000000)
        with patch.object(torch.cuda,'is_available',return_value=True):
            self.assertEqual(R['resolve_rank_device']('auto',**kwargs).type,'cpu')
            self.assertEqual(R['resolve_rank_device']('auto',**kwargs,streaming=True).type,'cuda')

    def test_empty(self):
        empty=np.array([],dtype=int)
        cmc,ap=E['_eval_torch'](*([empty]*8),'all','all',lambda a,b:None,device=torch.device('cpu'))
        self.assertEqual(ap,0.)
        self.assertTrue((cmc==0).all())


if __name__=='__main__':
    unittest.main()
