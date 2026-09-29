"""Publish reviewed snapshots; never alter the original recordings."""
import argparse
import os
import hashlib
import json
import shutil
import sys
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi

REPO=Path(__file__).resolve().parents[1]
DEFAULT_STORAGE=REPO.parent if (REPO.parent/'datasets').is_dir() else REPO
PROJECT=Path(os.environ.get('SO101_STORAGE_ROOT',str(DEFAULT_STORAGE))).resolve()
sys.path.insert(0,str(REPO/'hardware/dashboard'))
from validate_collection import validate
SOURCES={
 'pick_lift': ('so101_pick_lift_v1_20260929_222914_6b71','so101_pick_lift_dual_rgb_20260929',None),
 'pick_place': ('so101_pick_black_blocl_and_place_on_blue_place_20260929_225756_1268','so101_pick_place_blue_dual_rgb_20260929','Pick up the black 25 mm cube, place it in the blue region, and release the gripper.'),
}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--upload',action='store_true');parser.add_argument('--public',action='store_true');args=parser.parse_args()
 api=HfApi();owner=api.whoami()['name'];records={}
 for task,(source,name,corrected) in SOURCES.items():
  root=PROJECT/'datasets'/source
  report=validate(root)
  if not report['passed'] or len(report['valid_success_episodes'])!=30:raise RuntimeError(f'{task}: validation/30 successes failed')
  dest=PROJECT/'outputs/hf_exports'/name
  dest.mkdir(parents=True,exist_ok=True)
  for folder in ['data','meta','videos']:shutil.copytree(root/folder,dest/folder,dirs_exist_ok=True)
  benchmark=json.loads((dest/'meta/benchmark.json').read_text());old_task=benchmark['task']
  if corrected:
   tasks=pd.read_parquet(dest/'meta/tasks.parquet');tasks.index=[corrected if t==old_task else t for t in tasks.index];tasks.to_parquet(dest/'meta/tasks.parquet')
   for path in (dest/'meta/episodes').rglob('*.parquet'):
    table=pd.read_parquet(path);table['tasks']=table['tasks'].map(lambda ts:[corrected if t==old_task else t for t in ts]);table.to_parquet(path,index=False)
   benchmark.update(task=corrected,task_type='pick_and_place',annotation_correction={'original_task':old_task,'reason':'Collector used fixed Pick-and-Lift text; owner confirmed placement in blue region and gripper release.'})
   benchmark.pop('lift_height_mm',None);benchmark.pop('hold_seconds',None)
  for k in ['session_dir','dataset_root','monitor_url']:benchmark.pop(k,None)
  benchmark['calibration_sha256']={Path(k).name:v for k,v in benchmark['calibration_sha256'].items()}
  repo=f'{owner}/{name}';benchmark['repo_id']=repo
  (dest/'meta/benchmark.json').write_text(json.dumps(benchmark,ensure_ascii=False,indent=2))
  report=validate(dest)
  if corrected:
   report['warnings']=[w for w in report['warnings'] if '5 cm' not in w]
   report['warnings'].append('Success is human annotated: verify placement in the blue region and gripper release by replay. Physical measurements still require review.')
   (dest/'meta/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
  info=json.loads((dest/'meta/info.json').read_text())
  (dest/'README.md').write_text(f'''---
tags:
- lerobot
- robotics
- so101
- act
---
# SO-101 {task.replace('_',' ')} — dual RGB

Task: {benchmark['task']}

LeRobot v3.0, 20 FPS, {info['total_episodes']} recorded episodes, **30 human-labeled successful demonstrations**. Cameras: `observation.images.wrist` and `observation.images.third_person`, 640×480 RGB. State/action: six joints; body joints in degrees, gripper 0–100. Actions are sent joint targets, not measured end-effector commands.

All recorded episodes, including failures and excluded attempts, are retained. Labels are in `meta/benchmark_episodes.jsonl`. Train ACT only with episode indices from `meta/act_success_episodes.json`; do not train all episodes indiscriminately. Technical validation is in `meta/validation.json`; it does not automatically verify task success. Reset-pose deviation warnings remain for review.

Video files are finalized per episode and view; v3 episode metadata maps each video interval. Both views share frame/action indices, with separate host receipt timestamps (not hardware exposure synchronization). Mass and workspace/spawn measurements remain incomplete.

{'Task text was corrected in this export after confirmation by the owner; original recording files are unchanged. See meta/benchmark.json.' if corrected else ''}

This is a research snapshot. No license has been assigned. Baseline training is behavior cloning (ACT), not reinforcement learning.
''')
  files={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file() and p.name!='export_sha256.json'}
  (dest/'meta/export_sha256.json').write_text(json.dumps(files,indent=2))
  revision=None
  if args.upload:
   api.create_repo(repo,repo_type='dataset',private=not args.public,exist_ok=True)
   remote=api.dataset_info(repo)
   if remote.private==args.public:
    if not args.public:raise RuntimeError('Existing repository is public; refusing unexpected visibility')
    api.update_repo_settings(repo,repo_type='dataset',private=False)
   commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=dest,commit_message='Validated dual-view SO-101 recordings with success selection and task metadata')
   api.create_tag(repo_id=repo,repo_type='dataset',tag='v3.0',revision=commit.oid,exist_ok=True)
   remote=api.dataset_info(repo,revision=commit.oid,files_metadata=True)
   paths={f.rfilename for f in remote.siblings}
   assert set(files)<=paths and remote.private== (not args.public)
   revision=commit.oid
  records[task]={'repo_id':repo,'revision':revision,'root':str(dest),'success_episodes':report['valid_success_episodes'],'total_episodes':info['total_episodes'],'total_frames':info['total_frames'],'private':not args.public}
  (REPO/'data/datasets.json').write_text(json.dumps(records,indent=2))
  print(json.dumps({'uploaded':repo,'revision':revision,'uploaded':args.upload,'private':not args.public}),flush=True)

if __name__=='__main__':main()
