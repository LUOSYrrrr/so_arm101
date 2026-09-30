"""Print a reproducible Diffusion Policy command; execution requires --execute."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent

def command(args, entry):
    if not entry['revision']:raise ValueError('Dataset is not uploaded/verified yet')
    steps=args.steps if args.steps is not None else (1000 if args.smoke else 100000)
    if steps < 1:raise ValueError('steps must be positive')
    run_name=f'diffusion_{args.task}_{"smoke" if args.smoke else "baseline"}_seed1000'
    options=[f'--dataset.repo_id={entry["repo_id"]}',f'--dataset.revision={entry["revision"]}',
        '--dataset.episodes='+json.dumps(entry['success_episodes'],separators=(',',':')),
        '--policy.type=diffusion','--policy.device=cuda','--policy.push_to_hub=false',
        '--batch_size=8','--seed=1000',f'--steps={steps}',
        f'--save_freq={min(steps, 1000 if args.smoke else 5000)}',f'--log_freq={min(steps, 50 if args.smoke else 200)}',
        '--eval_freq=0',f'--output_dir={Path(args.output).resolve()}',f'--job_name={run_name}',
        '--wandb.enable=true','--wandb.project=so101-diffusion-baselines','--wandb.disable_artifact=true',
        f'--wandb.mode={args.wandb_mode}']
    if args.dataset_root:options.append(f'--dataset.root={Path(args.dataset_root).resolve()}')
    if args.entity:options.append(f'--wandb.entity={args.entity}')
    return [sys.executable,'-m','lerobot.scripts.lerobot_train',*options]

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['pick_lift','pick_place','battery_insertion'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--dataset-root');p.add_argument('--entity')
    p.add_argument('--steps',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--wandb-mode',choices=['online','offline'],default='online');args=p.parse_args()
    entry=json.loads((HERE.parents[1]/'data/datasets.json').read_text())[args.task];cmd=command(args,entry)
    print(shlex.join(cmd),flush=True)
    if args.execute:
        if Path(args.output).exists():raise FileExistsError('Use a new output directory; checkpoint resume is a separate operation')
        from importlib.metadata import version
        if version('lerobot')!='0.5.2':raise RuntimeError('Expected LeRobot 0.5.2 at pinned commit; check environment')
        from lerobot.utils.import_utils import require_package
        require_package('diffusers', extra='diffusion')
        import time
        started=time.monotonic()
        subprocess.run(cmd,check=True)
        elapsed=time.monotonic()-started
        print(f'Wall time including startup/save: {elapsed:.1f}s',flush=True)

if __name__=='__main__':main()
