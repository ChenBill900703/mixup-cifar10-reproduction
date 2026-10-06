"""Copy scientific source into a new empty experiment directory; never copy results."""
import argparse
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    dest=args.destination.resolve()
    if dest.exists() or dest==ROOT or ROOT in dest.parents:
        parser.error('Choose a new directory outside this publication checkout; existing paths are refused')
    dest.mkdir(parents=True)
    for name in ('train.py','config.json','requirements.txt','requirements-lock.txt',
                 'finalize_results.py','verify_final_evaluation.py','run_remaining.py'):
        shutil.copy2(ROOT/name,dest/name)
    for folder in ('models','utils','tests','docs','references','licenses'):
        shutil.copytree(ROOT/folder,dest/folder,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    (dest/'validation').mkdir()
    (dest/'outputs/ERM').mkdir(parents=True)
    (dest/'outputs/Mixup').mkdir(parents=True)
    (dest/'README.md').write_text(
        '# New experiment workspace\n\nNo previous outputs or readiness approval were copied. '
        'docs contains historical reference reports; new results belong to outputs/. '
        'Install requirements, download CIFAR10, run tests/validate.py, then explicitly start training.\n',
        encoding='utf-8')
    print(f'Created new workspace without published outputs: {dest}')


if __name__=='__main__':
    main()
