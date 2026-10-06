"""Audit actual completed checkpoints and publish an honest final comparison.

Does not train, select checkpoints, or alter any scientific configuration.
"""
import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
from train import lr_for_epoch
from utils.report import comparison
from utils.state import atomic_write, save_json, sha256, source_hashes

ROOT = Path(__file__).resolve().parent


def format_pp(value):
    # Display rounding only; retain original numeric evidence in summary.json.
    return '0.00' if abs(value) < 0.005 else f'{value:+.2f}'


def write_text(path, text):
    atomic_write(path, lambda f: f.write(text), False)


def audit(method):
    directory = ROOT/'outputs'/method
    path = directory/'summary.json'
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding='utf-8'))
    if summary['status'] != 'COMPLETE':
        return None
    assert summary['final_epoch'] == 200 and summary['result_claim_allowed'] is True
    cp = torch.load(directory/'resume_checkpoint.pt', map_location='cpu', weights_only=False)
    c, history = cp['config'], cp['history']
    assert cp['epoch'] == 200 and c['QUICK_TEST'] is False and c['USE_AMP'] is False
    assert c['method'] == method and c['alpha'] == (0.0 if method == 'ERM' else 1.0)
    assert c['batch_size'] == 128 and c['source_hashes'] == source_hashes(ROOT)
    assert [h['epoch'] for h in history] == list(range(1,201))
    for h in history:
        assert h['train_examples'] == 50000 and h['test_examples'] == 10000
        assert h['lr'] == lr_for_epoch(h['epoch'])
        assert all(math.isfinite(h[k]) for k in ('train_loss','test_loss','test_accuracy','test_error'))
    assert summary['final_test_error'] == history[-1]['test_error']
    assert summary['final_test_accuracy'] == history[-1]['test_accuracy']
    best = max(history, key=lambda h:h['test_accuracy'])
    assert summary['best_epoch'] == best['epoch']
    assert summary['best_test_accuracy'] == best['test_accuracy']
    with open(directory/'training_history.csv',newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==200
    for row,h in zip(rows,history):
        assert all(float(row[k])==v for k,v in h.items())
    final = torch.load(directory/'final_model_state_dict.pt',map_location='cpu',weights_only=True)
    assert final.keys()==cp['model_state_dict'].keys()
    assert all(torch.equal(v,final[k]) for k,v in cp['model_state_dict'].items())
    required = ['config.json','environment.json','source_boundary.md','assumptions.md',
                'training_history.csv','summary.json','test_accuracy_curve.png','test_error_curve.png',
                'training_loss_curve.png','paper_vs_reproduction.png','final_model_state_dict.pt',
                'resume_checkpoint.pt','README.md']
    assert all((directory/p).is_file() and (directory/p).stat().st_size>0 for p in required)
    evidence={'status':'PASS','run_id':cp['run_id'],'method':method,'final_epoch':200,
              'full_dataset_all_epochs':True,'final_weights_equal_epoch200_checkpoint':True,
              'csv_matches_checkpoint':True,'all_lr_values_match_official':True,
              'checkpoint_sha256':sha256(directory/'resume_checkpoint.pt'),
              'final_weights_sha256':sha256(directory/'final_model_state_dict.pt')}
    return summary,c,evidence


def main(allow_incomplete=False):
    results={m:audit(m) for m in ('ERM','Mixup')}
    complete=all(results.values())
    if not complete and not allow_incomplete:
        raise RuntimeError('Both methods must complete 200 epochs before final comparison')
    if complete:
        a,b=results['ERM'][1],results['Mixup'][1]
        assert {k:v for k,v in a.items() if k not in ('method','alpha')} == {k:v for k,v in b.items() if k not in ('method','alpha')}
    comparison(ROOT/'outputs')
    status='COMPLETE' if complete else 'PARTIAL'
    timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat()
    table=['| Method | Paper Test Error (%) | Reproduction Final Error (%) | Difference (pp) |',
           '|---|---:|---:|---:|']
    for m,paper in [('ERM',5.6),('Mixup',4.2)]:
        r=results[m]
        if r:
            s=r[0]
            table.append(f"| {m} | {paper:.2f} | {s['final_test_error']:.2f} | {format_pp(s['difference_final_minus_paper'])} |")
            shutil.copyfile(ROOT/'outputs/paper_comparison.csv',ROOT/'outputs'/m/'paper_comparison.csv')
        else:
            table.append(f'| {m} | {paper:.2f} | 尚未完成 | — |')
    lines=['# CIFAR-10 / PreAct ResNet-18 複現結果', '',f'狀態：{status}。更新：{timestamp}。',
           '', 'Primary metric 固定為第 200 epoch 的 full-test error；所有數值來自實際 checkpoint 與完整紀錄。', '',*table,'']
    for m,r in results.items():
        if not r:
            continue
        s,c,_=r
        lines += [f"## {m}", '',f"Final accuracy：{s['final_test_accuracy']:.2f}%；final error：{s['final_test_error']:.2f}%。",
                  f"相對論文 {s['paper_test_error']:.2f}%，差距 {format_pp(s['difference_final_minus_paper'])} 個百分點。",
                  f"Best error：{s['best_test_error']:.2f}%（epoch {s['best_epoch']}），僅診斷，未取代 final。",
                  f"訓練與評估累計 {s['training_seconds']/60:.1f} 分鐘，不含所有啟動／存檔／繪圖時間。", '',
                  f"來源：[summary](../outputs/{m}/summary.json)、[200-epoch history](../outputs/{m}/training_history.csv)。",'']
    if complete:
        improvement=results['ERM'][0]['final_test_error']-results['Mixup'][0]['final_test_error']
        lines += [f'同一預定 seed 下，ERM final error 減去 Mixup final error = {improvement:+.2f} 個百分點。',
                  '此為單次訓練比較，不代表多 seed 的平均效果，也未建立統計顯著性。','']
    lines += ['## 忠實度、差異與解讀','',
              '兩組使用同一官方模型、seed 20170922、batch 128、200 epochs、SGD momentum .9、weight decay 1e-4、FP32。ERM alpha=0；Mixup alpha=1。原始訓練程式和設定未為了接近論文而調整。',
              '', '以下是可解釋 reproduction gap 的候選因素，不是已證實的原因：', '',
              '- Paper seed: UNKNOWN；此處固定 seed 取自官方 README 範例，未依 test error 挑選。',
              '- 論文使用 Tesla P100；此處是 RTX 3070 Ti。',
              '- PyTorch/CUDA/cuDNN 與歷史版本可能不同；原始版本未公布，尤其 BN gamma 初始化預設跨版本有差異。',
              '- cuDNN benchmark=True，不能保證 kernel 或跨程序 bitwise deterministic。',
              '- 官方實際 LR schedule 在 human epoch 102/152 切換；論文文字自然解讀是 101/151，依使用者指定採官方行為。',
              '- Windows 原本 8 workers 載入 DLL 時發生 WinError1455，改 workers=0；transforms 不變，但亂數流消耗順序不同。',
              '- Figure 3(a) 未明示 final-vs-best；本複現遵守 epoch200 endpoint。Table5 的 last-10 median 不混用。',
              '', '完整來源核對：[source_boundary.md](source_boundary.md)。工程驗證：[validation_report.md](validation_report.md)。',
              '', 'Quick Tests、失敗的初始化測試與 8-worker 資源失敗紀錄均保留；未當成正式結果，也未用於調整 seed 或科學超參數。','']
    write_text(ROOT/'docs/reproduction_results.md','\n'.join(lines))
    save_json(ROOT/'validation/final_integrity.json',{'status':status,'checked_at':timestamp,
              'methods':{m:r[2] if r else {'status':'NOT_COMPLETE'} for m,r in results.items()},
              'paired_configs_equal_except_method_alpha':complete})
    readme=ROOT/'README.md'
    text=readme.read_text(encoding='utf-8')
    marker='\n<!-- VERIFIED_RESULTS -->\n'
    text=text.split(marker)[0]
    text += marker+'\n## Verified results\n\nStatus: '+status+'\n\n'+'\n'.join(table)+'\n\nSee [完整結果與差異分析](docs/reproduction_results.md) and [integrity checks](validation/final_integrity.json).\n'
    write_text(readme,text)
    print(f'RESULT AUDIT {status}: docs/reproduction_results.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--allow-incomplete',action='store_true')
    main(p.parse_args().allow_incomplete)
