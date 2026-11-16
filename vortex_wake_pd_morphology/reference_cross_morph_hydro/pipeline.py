"""Resumable isolated Oracle-Force collection, learning and paired evaluation."""
from __future__ import annotations
import json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from .config import RUN,ROOT,PARENT_RUN,TRAIN_N,SEEDS,PROTOCOL,write,status,source_manifest,sha,check_stop

PREFIX='fish_rl.cross_morph_20261021_hydro_v1.'
GPUS=('1','3','4')
BASE=ROOT.parent
ENV=os.environ|{'OMP_NUM_THREADS':'2','MUJOCO_GL':'egl','CUDA_DEVICE_ORDER':'PCI_BUS_ID',
    '__EGL_VENDOR_LIBRARY_FILENAMES':str(BASE/'nvidia_egl.json'),'PYTHONUNBUFFERED':'1'}

def alive(pid):
    p=Path(f'/proc/{pid}/cmdline')
    return p.exists() and bool(p.read_bytes())

def stage_sources():
    return source_manifest()|{str(p.relative_to(ROOT)):sha(p) for p in (RUN/'frozen_teachers').glob('N*.pt')}

def frozen_teachers():
    dest=RUN/'frozen_teachers';dest.mkdir(parents=True,exist_ok=True)
    parent_manifest=json.loads((PARENT_RUN/'frozen_teachers/manifest.json').read_text())
    for n in TRAIN_N:
        src=PARENT_RUN/'frozen_teachers'/f'N{n}.pt';dst=dest/src.name
        assert src.is_file()
        if not dst.exists():shutil.copy2(src,dst)
        assert sha(src)==sha(dst),f'Teacher checkpoint mismatch N{n}'
    write(dest/'provenance.json',dict(parent=str(PARENT_RUN),
        checkpoint_hashes={f'N{n}':sha(dest/f'N{n}.pt') for n in TRAIN_N},
        parent_manifest=parent_manifest,no_teacher_updates=True))

def spawn(module,args,label,gpu):
    with (RUN/f'{label}.log').open('a') as log:
        return subprocess.Popen([sys.executable,'-u','-m',PREFIX+module,*map(str,args)],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=ENV|{'CUDA_VISIBLE_DEVICES':gpu,'MUJOCO_EGL_DEVICE_ID':gpu})

def serial(module,*args):
    check_stop();p=spawn(module,args,module,'1')
    write(RUN/'active_stage.json',dict(module=module,pid=p.pid,args=args))
    while p.poll() is None:check_stop();time.sleep(3)
    if p.returncode:raise RuntimeError(f'{module} failed {p.returncode}; inspect {module}.log')

def batch(stage,module,specs):
    path=RUN/f'{stage}_jobs.json';old=json.loads(path.read_text()) if path.exists() else {}
    active={};handles={};pending=[]
    for label,args,done in specs:
        prior=old.get('active',{}).get(label)
        if prior and alive(prior['pid']):
            cmd=Path(f"/proc/{prior['pid']}/cmdline").read_bytes().split(b'\0')
            assert (PREFIX+module).encode() in cmd and Path(f"/proc/{prior['pid']}/cwd").resolve()==ROOT
            assert prior['args']==list(args) and prior['gpu'] in GPUS
            active[label]=prior
        elif done.exists():assert json.loads(done.read_text())['passed']
        else:pending.append((label,args,done))
    assert len({v['gpu'] for v in active.values()})==len(active)
    done_paths={label:done for label,_,done in specs}
    while active or pending:
        check_stop()
        if shutil.disk_usage(RUN).free<15*1024**3:raise RuntimeError('Under 15 GiB free; preserving data')
        while pending and len(active)<len(GPUS):
            label,args,done=pending.pop(0)
            gpu=next(g for g in GPUS if g not in {v['gpu'] for v in active.values()})
            p=spawn(module,args,label,gpu);handles[label]=p
            active[label]=dict(pid=p.pid,gpu=gpu,args=list(args),module=module)
        write(path,dict(active=active,pending=[x[0] for x in pending],module=module))
        for label,record in list(active.items()):
            running=handles[label].poll() is None if label in handles else alive(record['pid'])
            if not running:
                done=done_paths[label]
                assert done.exists() and json.loads(done.read_text())['passed'],f'{label} failed; see log'
                active.pop(label)
        time.sleep(3)
    write(path,dict(active={},pending=[],passed=True,module=module))

def main():
    RUN.mkdir(parents=True,exist_ok=True);lock=RUN/'supervisor.lock'
    if lock.exists() and alive(json.loads(lock.read_text())['pid']):raise RuntimeError('Supervisor already live')
    write(lock,dict(pid=os.getpid(),started=time.time()))
    frozen_teachers()
    manifest=stage_sources()
    if (RUN/'source_manifest.json').exists():assert manifest==json.loads((RUN/'source_manifest.json').read_text())
    else:write(RUN/'source_manifest.json',manifest)
    write(RUN/'protocol.json',PROTOCOL)
    try:
        status('physical_signal_preflight');serial('preflight')
        status('fresh_success_trajectory_collection',gpu_pool=GPUS,attempts_per_cell=20)
        batch('collection','data',[(f'collect_N{n}',('--n',n),RUN/f'data/N{n}/complete.json') for n in TRAIN_N])
        status('whole_episode_causal_audit');serial('data','--build');serial('audit','dataset')
        assert json.loads((RUN/'independent_dataset_audit.json').read_text())['passed']
        status('fresh_hydrodynamic_WM_and_equal_budget_ABC',gpu_pool=GPUS)
        batch('learning','learn',[(f'learn_{s}',('--seed',s),RUN/f'learning_workers/{s}.json') for s in SEEDS])
        write(RUN/'learning_complete.json',dict(passed=True,seeds=SEEDS,bc_kinds=['A','B','C'],
            n8_updates=0,dataset_sha256=sha(RUN/'dataset.h5')))
        paths=list((RUN/'bc').glob('*/*/best.pt'))+list((RUN/'world').glob('*/best.pt'))
        assert len(paths)==12
        frozen={str(p):sha(p) for p in paths}
        write(RUN/'evaluation_frozen_manifest.json',frozen)
        status('frozen_paired_N8_six_suite_evaluation',gpu_pool=GPUS)
        batch('evaluation','evaluate',[(f'eval_{s}',('--seed',s),RUN/f'evaluation_workers/{s}.json') for s in SEEDS])
        assert frozen=={str(p):sha(p) for p in paths}
        write(RUN/'evaluation_complete.json',dict(passed=True,parameters_unchanged=True,n8_updates=0))
        status('report_and_final_audit');serial('report');serial('audit','final')
        assert manifest==stage_sources()
        assert json.loads((RUN/'final_complete.json').read_text())['passed']
        status('complete',oracle_force=True,full_teacher_coverage_qualified=False)
    except BaseException:
        write(RUN/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()))
        status('needs_diagnosis');raise

if __name__=='__main__':main()
