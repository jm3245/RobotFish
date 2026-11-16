"""Independent input, chronology, physical-signal and frozen-model checks."""
from __future__ import annotations
import argparse
import json
import numpy as np
import h5py
from .config import RUN,TRAIN_N,SEEDS,PROTOCOL,write,sha


def dataset():
    path=RUN/'dataset.h5'
    contract=json.loads((RUN/'dataset_audit.json').read_text())
    assert contract['passed'] and contract['sha256']==sha(path)
    with h5py.File(path,'r') as h:
        keys={k:tuple(v.shape) for k,v in h.items()}
        assert h['history'].shape[1:]==(50,36)
        assert h['observation'].shape[1:]==(17,)
        assert h['action'].shape[1:]==(4,)
        assert h['labels'].shape[1:]==(3,44)
        assert set(np.unique(h['n'][:]))==set(TRAIN_N)
        s=h['split'][:];e=h['episode'][:]
        groups=[set(e[s==i]) for i in range(3)]
        assert all(not(groups[i]&groups[j]) for i in range(3) for j in range(i+1,3))
        assert all(len(x)>0 for x in groups)
        assert np.isfinite(h['history'][:]).all() and np.isfinite(h['labels'][:]).all()
        assert 'flow' not in h.attrs['policy_input_whitelist']
        assert h.attrs['policy_input_whitelist']=='observation,history'
        # Every history ends at the pre-action state t, and its final executed
        # wave equals the current 17D control observation.
        for start in range(0,len(s),1024):
            end=min(start+1024,len(s))
            hist=h['history'][start:end]
            obs=h['observation'][start:end]
            assert np.allclose(hist[:,-1,-4:],obs[:,11:15],rtol=1e-5,atol=1e-5)
    rows=dict(passed=True,dataset_sha256=sha(path),keys=keys,episodes_by_split=[len(g) for g in groups],
        windows_by_split=[int((s==i).sum()) for i in range(3)],n8_in_fit=False,
        teacher_privileged_in_student=False,true_flow_in_student_or_world_supervision=False,
        future_policy_input=False,history_ends_at_pre_state_t=True,
        oracle_force=True,pressure_sensor=False,body_to_eight_regions='axial overlap; exact total, approximate within rigid body')
    write(RUN/'independent_dataset_audit.json',rows)


def final():
    dataset()
    assert json.loads((RUN/'learning_complete.json').read_text())['passed']
    assert json.loads((RUN/'evaluation_complete.json').read_text())['passed']
    paths=list((RUN/'bc').glob('*/*/best.pt'))+list((RUN/'world').glob('*/best.pt'))
    assert len(paths)==12
    frozen=json.loads((RUN/'evaluation_frozen_manifest.json').read_text())
    assert frozen=={str(p):sha(p) for p in paths}
    summaries=list((RUN/'evaluation/N8').glob('*/*/*/summary.json'))
    assert len(summaries)==len(SEEDS)*len(PROTOCOL['suites'])*3
    for seed in SEEDS:
        for suite in PROTOCOL['suites']:
            rows=[json.loads((RUN/'evaluation/N8'/str(seed)/suite/k/'summary.json').read_text()) for k in 'ABC']
            assert len({r['initial_hash'] for r in rows})==1
            assert len({r['flow_hash'] for r in rows})==1
            assert all(r['n8_training_updates']==0 and r['parameters_unchanged'] for r in rows)
    write(RUN/'final_complete.json',dict(passed=True,scope='oracle_force_success_subset',
        data_sha256=sha(RUN/'dataset.h5'),model_hashes=frozen,
        paired_n8_suites=len(summaries),n8_train_updates=0,
        report_sha256=sha(RUN/'RESULTS_zh.md')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('dataset','final'))
    getattr(__import__(__name__,fromlist=['dataset','final']),p.parse_args().stage)()
