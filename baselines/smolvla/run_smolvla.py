"""Fine-tune SmolVLA on one fixed successful-episode SO-101 dataset."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['pick_lift', 'pick_place', 'battery_insertion'], required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError('steps must be positive')
    output = args.output.resolve()
    entry = json.loads((ROOT/'data/datasets.json').read_text())[args.task]
    cmd = [sys.executable, '-m', 'lerobot.scripts.lerobot_train']
    if args.resume_from:
        checkpoint = args.resume_from.resolve()
        config = checkpoint/'pretrained_model/train_config.json'
        saved = json.loads(config.read_text())
        if saved['policy']['type'] != 'smolvla' or saved['dataset']['repo_id'] != entry['repo_id']:
            raise ValueError('Checkpoint does not match SmolVLA task')
        if not (checkpoint/'training_state/training_step.json').is_file():
            raise FileNotFoundError('Missing checkpoint training state')
        cmd += [f'--config_path={config}', '--resume=true']
    else:
        ready = json.loads((ROOT/'outputs/setup/smolvla_ready.json').read_text())
        cmd += [f'--policy.path={ready["base_path"]}', '--policy.device=cuda',
                '--policy.push_to_hub=false', '--policy.load_vlm_weights=false',
                '--rename_map='+json.dumps({'observation.images.third_person':'observation.images.camera1',
                                           'observation.images.wrist':'observation.images.camera2'}, separators=(',', ':')),
                '--policy.scheduler_decay_steps=20000',
                f'--dataset.repo_id={entry["repo_id"]}', f'--dataset.revision={entry["revision"]}',
                '--dataset.episodes='+json.dumps(entry['success_episodes'], separators=(',', ':')),
                f'--dataset.root={ROOT/"datasets"/entry["repo_id"].split("/")[-1]}',
                '--dataset.video_backend=pyav', '--batch_size=8', '--seed=1000',
                '--log_freq=200', '--eval_freq=0', f'--job_name=smolvla_{args.task}_seed1000',
                '--wandb.enable=true', '--wandb.mode=online',
                '--wandb.project=so101-smolvla-baselines', '--wandb.disable_artifact=true']
    cmd += [f'--steps={args.steps}', '--save_freq=5000', f'--output_dir={output}']
    print(shlex.join(cmd), flush=True)
    if args.execute:
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Run training through Slurm')
        if output.exists() and not args.resume_from:
            raise FileExistsError('Use a fresh output or --resume-from')
        started = time.monotonic()
        subprocess.run(cmd, check=True)
        (output/'training_run.json').write_text(json.dumps(
            {'task':args.task, 'steps':args.steps, 'wandb_mode':'online',
             'slurm_job_id':os.environ['SLURM_JOB_ID'], 'elapsed_seconds':time.monotonic()-started,
             'resume_from':str(args.resume_from) if args.resume_from else None, 'command':cmd}, indent=2)+'\n')


if __name__ == '__main__':
    main()
