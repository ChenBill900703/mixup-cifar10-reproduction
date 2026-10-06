"""Runs real, capped CUDA jobs in separate processes; never starts a formal run."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from train import configure
from utils.state import environment, save_json, source_hashes, sha256


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    folder = ROOT/'validation'/stamp
    folder.mkdir(parents=True)
    base = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    configure(base)
    env = environment()
    save_json(folder/'environment.json',env)
    result = {'status':'RUNNING','run_id':stamp,'checks':{},'source_hashes':source_hashes(ROOT),'environment':env}
    save_json(ROOT/'validation/readiness.json',result)
    def run(name,args,expected=0):
        print(f'VALIDATION: {name}',flush=True)
        with open(folder/(name+'.log'),'w',encoding='utf-8') as f:
            proc = subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        text = (folder/(name+'.log')).read_text(encoding='utf-8')
        print(text[-2000:],flush=True)
        if proc.returncode != expected:
            raise RuntimeError(f'{name} returned {proc.returncode}; see {folder/(name+".log")}')
        result['checks'][name]='PASS'
        return text
    try:
        assert env['cuda_available'], 'torch.cuda.is_available() must be True'
        a = torch.randn(32,32,device='cuda',requires_grad=True)
        (a@a).sum().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(a.grad).all()
        result['checks']['cuda_forward_backward']='PASS'
        run('syntax_import',['-m','compileall','-q','train.py','models','utils','tests'])
        run('invariants',['tests/test_invariants.py'])
        run('formal_without_approval',['train.py','--formal'],expected=1)
        for method in ('ERM','Mixup'):
            args=['train.py','--quick','--method',method,'--quick-id',stamp]
            first=run(method+'_epoch1',args+['--stop-after-epoch','1'])
            out=ROOT/'outputs/quick'/stamp/method
            cp=torch.load(out/'resume_checkpoint.pt',map_location='cpu',weights_only=False)
            assert cp['epoch']==1 and len(cp['history'])==1
            epoch1=dict(cp['history'][0])
            assert 'FRESH RUN - STARTING EPOCH 1' in first
            changed=dict(base)
            changed['seed']+=1
            changed_path=folder/(method+'_incompatible_config.json')
            save_json(changed_path,changed)
            checkpoint_before=sha256(out/'resume_checkpoint.pt')
            rejected=run(method+'_reject_incompatible_resume',args+['--config',str(changed_path)],expected=1)
            assert 'REFUSING RESUME: config mismatch' in rejected
            assert checkpoint_before==sha256(out/'resume_checkpoint.pt')
            resumed=run(method+'_resume_epoch2',args)
            assert 'LAST COMPLETED EPOCH = 1' in resumed and 'NEXT EPOCH = 2' in resumed
            cp=torch.load(out/'resume_checkpoint.pt',map_location='cpu',weights_only=False)
            assert cp['epoch']==2 and cp['history'][0]==epoch1
            assert [r['epoch'] for r in cp['history']]==[1,2]
            assert len(cp['optimizer_state_dict']['state'])>0
            assert all('momentum_buffer' in s for s in cp['optimizer_state_dict']['state'].values())
            proof=json.loads((out/'resume_verification_epoch_1.json').read_text())
            assert proof['rng_digest_saved']==proof['rng_digest_restored']
            summary=json.loads((out/'summary.json').read_text())
            assert summary['result_claim_allowed'] is False and summary['difference_final_minus_paper'] is None
            before=sha256(out/'resume_checkpoint.pt')
            run(method+'_completed_no_retrain',args)
            assert before==sha256(out/'resume_checkpoint.pt')
            result['checks'][method+'_resume_state']='PASS'
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc))
        raise
    finally:
        save_json(folder/'readiness.json',result)
        save_json(ROOT/'validation/readiness.json',result)
    print('READY FOR FORMAL RUN - user approval still required',flush=True)


if __name__=='__main__':
    main()
