"""User-authorized ERM run followed by verified final comparison, without retuning."""
import subprocess
import sys
import json
import os
import tempfile
import shutil
import ctypes
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def save_json(path, value):
    # Keep the waiting launcher free of PyTorch/CUDA DLLs and their memory cost.
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def memory_snapshot():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [
            (name,ctypes.c_ulonglong) for name in ('total_physical','available_physical',
             'total_commit','available_commit','total_virtual','available_virtual','extended')]
    m=MemoryStatus()
    m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return {'memory_read_failed':True}
    return {k:getattr(m,k) for k in ('load','total_physical','available_physical','total_commit','available_commit')}


def main():
    state={'status':'RUNNING_ERM','started_utc':datetime.now(timezone.utc).isoformat(),
           'authorization':'User asked to finish the remaining work after Mixup completes',
           'training_command':[sys.executable,'-u','train.py','--formal','--approve-formal','--method','ERM']}
    path=ROOT/'outputs/remaining_work.json'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if path.exists():
        shutil.copyfile(path,ROOT/'outputs'/f'remaining_work_before_{stamp}.json')
    save_json(path,state)
    try:
        with open(ROOT/'outputs'/f'memory_{stamp}.jsonl','a',encoding='utf-8') as log:
            with subprocess.Popen(state['training_command'],cwd=ROOT) as child:
                while True:
                    record={'utc':datetime.now(timezone.utc).isoformat(),**memory_snapshot()}
                    log.write(json.dumps(record)+'\n')
                    log.flush()
                    try:
                        code=child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if code:
                    raise subprocess.CalledProcessError(code,state['training_command'])
        state['status']='VERIFYING_AND_REPORTING'
        save_json(path,state)
        subprocess.run([sys.executable,'-u','finalize_results.py'],cwd=ROOT,check=True)
        state['status']='COMPLETE'
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc))
        raise
    finally:
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        save_json(path,state)


if __name__=='__main__':
    main()
