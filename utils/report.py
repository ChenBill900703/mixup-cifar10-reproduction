import csv
from pathlib import Path
from .state import atomic_write, save_json


def history_csv(path, history):
    if not history:
        return
    def write(f):
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    atomic_write(path, write, False)


def export_run(directory, checkpoint):
    c, h = checkpoint['config'], checkpoint['history']
    history_csv(directory / 'training_history.csv', h)
    if not h:
        return
    final, best = h[-1], checkpoint['best']
    quick = c['QUICK_TEST']
    completed = final['epoch'] == c['effective_epochs']
    paper = 5.6 if c['method'] == 'ERM' else 4.2
    allowed = completed and not quick and final['epoch'] == 200
    summary = dict(method=c['method'], seed=c['seed'], final_epoch=final['epoch'],
                   status='QUICK_TEST_ONLY' if quick else ('COMPLETE' if completed else 'IN_PROGRESS'),
                   result_claim_allowed=allowed, final_test_accuracy=final['test_accuracy'],
                   final_test_error=final['test_error'], best_test_accuracy=best['accuracy'],
                   best_test_error=100-best['accuracy'], best_epoch=best['epoch'],
                   paper_test_error=paper, difference_final_minus_paper=final['test_error']-paper if allowed else None,
                   training_seconds=final['elapsed_seconds'], gpu=checkpoint['environment']['gpu'],
                   official_source_revision=c['official_revision'], source_hashes=c['source_hashes'],
                   primary_metric='epoch_200_final_test_error',
                   paper_metric_caveat='Figure 3(a) does not explicitly identify final vs best; Table 5 uses last-10 median.')
    save_json(directory / 'summary.json', summary)
    if completed:
        plots(directory, h, summary)
    (directory / 'README.md').write_text(
        f"# {c['method']} run\n\nStatus: {summary['status']}. Seed: {c['seed']}. "
        f"Completed epoch: {final['epoch']}.\n\n"
        "Quick tests and in-progress metrics are not reproduction results. "
        "Best test accuracy is diagnostic only. Resume checkpoint is authoritative; CSV can be rebuilt.\n",
        encoding='utf-8')
    return summary


def plots(directory, history, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key, label, name in [('test_accuracy', 'Test accuracy (%)', 'test_accuracy_curve.png'),
                             ('test_error', 'Test error (%)', 'test_error_curve.png'),
                             ('train_loss', 'Training cross-entropy', 'training_loss_curve.png')]:
        fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
        ax.plot([h['epoch'] for h in history], [h[key] for h in history], marker='o' if len(history)<5 else None)
        ax.set(xlabel='Epoch (1-based)', ylabel=label,
               title=f"{summary['method']} | " + ('QUICK TEST - NOT A RESULT' if not summary['result_claim_allowed'] else 'Final-epoch protocol'))
        ax.grid(alpha=.2)
        fig.savefig(directory / name, dpi=160)
        plt.close(fig)
    if summary['result_claim_allowed']:
        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        values = [summary['paper_test_error'], summary['final_test_error']]
        bars = ax.bar(['Paper Figure 3(a)', 'Reproduction epoch 200'], values, color=['#777777', '#276b9d'])
        ax.bar_label(bars, fmt='%.2f%%')
        ax.set(ylabel='Test error (%)', title=summary['method'], ylim=(0, max(values)*1.2))
        fig.savefig(directory / 'paper_vs_reproduction.png', dpi=160)
        plt.close(fig)


def comparison(root):
    import json
    rows = []
    for method, paper in [('ERM', 5.6), ('Mixup', 4.2)]:
        p = root / method / 'summary.json'
        s = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        error = s.get('final_test_error') if s.get('result_claim_allowed') else None
        rows.append({'Method': method, 'Paper Test Error': paper,
                     'Reproduction Final Error': error,
                     'Difference (pp)': error-paper if error is not None else None,
                     'Status': s.get('status', 'NOT_EXECUTED')})
    def write(f):
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    atomic_write(root / 'paper_comparison.csv', write, False)
