"""Re-evaluate saved epoch-200 weights without training or modifying results."""
import json
from datetime import datetime, timezone
from pathlib import Path
import torch
from models import ResNet18
from train import configure
from utils.data import build_loaders
from utils.state import save_json, sha256

ROOT=Path(__file__).resolve().parent


def main():
    result={'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),'methods':{}}
    for method in ('ERM','Mixup'):
        directory=ROOT/'outputs'/method
        config=json.loads((directory/'config.json').read_text())
        summary=json.loads((directory/'summary.json').read_text())
        assert summary['final_epoch']==200 and summary['status']=='COMPLETE'
        configure(config)
        loaders,_,provenance=build_loaders(config,ROOT/'data')
        model=ResNet18().cuda()
        weights=directory/'final_model_state_dict.pt'
        model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
        model.eval()
        correct,total=0,0
        with torch.no_grad():
            for x,y in loaders['test']:
                prediction=model(x.cuda()).argmax(1)
                correct+=int(prediction.eq(y.cuda()).sum().item())
                total+=len(y)
        assert total==10000
        accuracy=100*correct/total
        error=100*(total-correct)/total
        assert abs(accuracy-summary['final_test_accuracy'])<1e-9
        assert abs(error-summary['final_test_error'])<1e-9
        result['methods'][method]={'status':'PASS','test_count':total,'correct':correct,
                                  'incorrect':total-correct,'test_accuracy':accuracy,'test_error':error,
                                  'matches_epoch200_summary':True,'final_weights_sha256':sha256(weights),
                                  'dataset_integrity':provenance['integrity']}
        print(method,correct,'/',total,'correct; error',error,'PASS',flush=True)
        del model,loaders
        torch.cuda.empty_cache()
    result['status']='PASS'
    result['completed_utc']=datetime.now(timezone.utc).isoformat()
    save_json(ROOT/'validation/independent_evaluation.json',result)


if __name__=='__main__':
    main()
