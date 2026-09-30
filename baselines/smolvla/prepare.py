"""Cache pinned SmolVLA weights and VLM metadata for offline compute-node loading."""
import json
import os
from pathlib import Path

BASE_REVISION = 'd9f33c94a60fb382c90dea2164c96845bd955e28'
ROOT = Path(__file__).resolve().parents[2]


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run dependency/model setup on an allocated CPU node')
    from huggingface_hub import HfApi, snapshot_download
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from importlib.metadata import version
    api = HfApi()
    base = snapshot_download('lerobot/smolvla_base', revision=BASE_REVISION)
    vlm = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
    vlm_revision = api.model_info(vlm).sha
    snapshot_download(vlm, revision=vlm_revision,
                      allow_patterns=['*.json', '*.txt', '*.model', '*.jinja', 'tokenizer.*'])
    # Also cache the main reference used by the base policy's VLM/tokenizer name.
    snapshot_download(vlm, revision='main',
                      allow_patterns=['*.json', '*.txt', '*.model', '*.jinja', 'tokenizer.*'])
    ready = {'base_repo': 'lerobot/smolvla_base', 'base_revision': BASE_REVISION,
             'base_path': base, 'vlm_repo': vlm, 'vlm_revision': vlm_revision,
             'versions': {name: version(name) for name in ('lerobot', 'torch', 'transformers', 'num2words')}}
    (ROOT/'outputs/setup/smolvla_ready.json').write_text(json.dumps(ready, indent=2)+'\n')
    print(json.dumps(ready, indent=2), flush=True)


if __name__ == '__main__':
    main()
