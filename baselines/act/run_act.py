"""Print a reproducible ACT command; execution requires --execute."""
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
    run_name=f'act_{args.task}_{"smoke" if args.smoke else "baseline"}_seed1000'
    options=[f'--dataset.repo_id={entry["repo_id"]}',f'--dataset.revision={entry["revision"]}',
        '--dataset.episodes='+json.dumps(entry['success_episodes'],separators=(',',':')),
        '--policy.type=act','--policy.device=cuda','--policy.push_to_hub=false',
        '--batch_size=8','--seed=1000',f'--steps={100 if args.smoke else 100000}',
        f'--save_freq={100 if args.smoke else 20000}',f'--log_freq={10 if args.smoke else 200}',
        '--eval_freq=0',f'--output_dir={Path(args.output).resolve()}',f'--job_name={run_name}',
        '--wandb.enable=true','--wandb.project=so101-act-baselines','--wandb.disable_artifact=true',
        f'--wandb.mode={args.wandb_mode}']
    if args.dataset_root:options.append(f'--dataset.root={Path(args.dataset_root).resolve()}')
    if args.entity:options.append(f'--wandb.entity={args.entity}')
    return [sys.executable,'-m','lerobot.scripts.lerobot_train',*options]

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['pick_lift','pick_place'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--dataset-root');p.add_argument('--entity')
    p.add_argument('--smoke',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--wandb-mode',choices=['online','offline'],default='offline');args=p.parse_args()
    entry=json.loads((HERE.parents[1]/'data/datasets.json').read_text())[args.task];cmd=command(args,entry)
    print(shlex.join(cmd),flush=True)
    if args.execute:
        if Path(args.output).exists():raise FileExistsError('Use a new output directory; checkpoint resume is a separate operation')
        from importlib.metadata import version
        if version('lerobot')!='0.5.2':raise RuntimeError('Expected LeRobot 0.5.2 at pinned commit; check environment')
        subprocess.run(cmd,check=True)

if __name__=='__main__':main()
