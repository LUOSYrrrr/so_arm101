"""Local subprocess coordinator. No robot APIs in the HTTP server."""
import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
# Preserve this workstation's existing recordings; standalone clones use repo storage.
DEFAULT_STORAGE=REPO.parent if (REPO.parent/'datasets').is_dir() else REPO
PROJECT=Path(os.environ.get('SO101_STORAGE_ROOT',str(DEFAULT_STORAGE))).resolve()
PYTHON=Path('/home/siyuanluo/anaconda3/envs/lerobot/bin/python')
SESSIONS=PROJECT/'outputs/collection_sessions'


def target_count(value):
    if type(value) is not int or not 1 <= value <= 10000:
        raise ValueError('目标成功总条数必须为 1–10000 的整数')
    return value


def read_existing(root):
    info=json.loads((root/'meta/info.json').read_text())
    cfg=json.loads((root/'meta/benchmark.json').read_text())
    labels=[json.loads(line) for line in (root/'meta/benchmark_episodes.jsonl').read_text().splitlines() if line.strip()]
    if info['codebase_version']!='v3.0' or cfg.get('task_type')!='pick_and_lift':raise ValueError('仅支持本面板的 v3.0 Pick-and-Lift 数据集')
    if len(labels)!=info['total_episodes'] or [e['episode_index'] for e in labels]!=list(range(len(labels))):raise ValueError('episode 元数据与标注不一致，先修复再续录')
    cfg['dataset_root']=str(root.resolve())
    return cfg,labels,info


class CollectionManager:
    def __init__(self,monitor,port):
        self.monitor=monitor;self.port=port;self.lock=threading.RLock();self.process=None;self.validating=False
        self.state={'mode':'idle','saved':0,'successes':0,'frames':0,'error':None}
        self.cfg=None;self.session=None;self.log=None
        saved=sorted(SESSIONS.glob('*/config.json'),key=lambda p:p.stat().st_mtime) if SESSIONS.exists() else []
        if saved:
            self.session=saved[-1].parent;self.cfg=json.loads(saved[-1].read_text())
            status=self.session/'status.json'
            if status.exists():self.state={**json.loads(status.read_text()),'mode':'closed'}

    def running(self):return self.process is not None and self.process.poll() is None

    def status(self):
        with self.lock:
            if self.running():
                self.command({'action':'heartbeat'})
            elif self.state['mode'] not in ['idle','closed','error'] and not self.validating:
                self.state.update(mode='error',error='采集进程已退出，请查看错误日志；未自动重启机械臂')
            result={**self.state,'active':self.running(),'validating':self.validating,'config':self.cfg}
            if self.cfg:
                root=Path(self.cfg['dataset_root'])
                ann=root/'meta/benchmark_episodes.jsonl'
                result['episodes']=[json.loads(s) for s in ann.read_text().splitlines() if s.strip()] if ann.exists() else []
                report=root/'meta/validation.json'
                result['validation']=json.loads(report.read_text()) if report.exists() else None
            result['datasets']=self.datasets()
            return result

    def datasets(self):
        items=[]
        for root in sorted((PROJECT/'datasets').glob('*')):
            if not (root/'meta/benchmark.json').exists():continue
            try:
                cfg,labels,info=read_existing(root)
                items.append({'id':root.name,'saved':len(labels),'successes':sum(e['human_success'] for e in labels)})
            except (OSError,ValueError,KeyError):continue
        return items

    def existing(self,ident):
        if not isinstance(ident,str) or ident not in {d['id'] for d in self.datasets()}:raise ValueError('请选择已有数据集')
        return read_existing(PROJECT/'datasets'/ident)

    def select(self,data):
        with self.lock:
            if self.running() or self.validating:raise ValueError('先结束当前会话或等待验证完成')
            cfg,labels,_=self.existing(data.get('dataset_id'))
            self.cfg=cfg
            self.state={'mode':'closed','saved':len(labels),'successes':sum(e['human_success'] for e in labels),'frames':0,'target':cfg.get('target_successes',10),'error':None}

    def start(self,data):
        with self.lock:
            if self.running() or self.validating:raise ValueError('已有会话或验证在运行')
            if data.get('ready') is not True:raise ValueError('请先确认现场与串口已准备好')
            target=target_count(data.get('target_successes',10))
            resume=data.get('dataset_id')
            previous,labels,info=self.existing(resume) if resume else ({},[],{})
            successes=sum(e['human_success'] for e in labels)
            if resume and target<=successes:raise ValueError(f'已有 {successes} 条成功示范，目标总数需大于已有成功数')
            snapshot=self.monitor.status()
            ids={r:c['id'] for r,c in previous['cameras'].items()} if resume else data.get('camera_ids',{})
            if not isinstance(ids,dict) or set(ids)!={'wrist','third_person'} or len(set(ids.values()))!=2:
                raise ValueError('请分别选择腕部与第三视角，不能使用同一台相机')
            cameras={}
            for role, ident in ids.items():
                matches=[c for c in snapshot['cameras'] if c['id']==ident and c['live']]
                if len(matches)!=1:raise ValueError(role+' 相机必须正在出图')
                c=matches[0]
                cameras[role]={'id':c['id'],'path':c['path'],'width':c['width'],'height':c['height'],'camera_fps':c['configured_fps']}
            if resume and cameras!=previous['cameras']:raise ValueError('相机路径、角色、尺寸或帧率与原数据集不一致；请恢复原配置')
            suffixes={r['id'] for r in snapshot['robots']}
            if not all(any(s in n for n in suffixes) for s in ['5B61035080','5B61036020']):raise ValueError('两台机械臂必须都在线')
            if not PYTHON.exists():raise ValueError('LeRobot Python 环境不存在')
            label='resume' if resume else str(data.get('name','so101_pick_lift')).strip()
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,48}',label):raise ValueError('名称仅支持字母、数字、下划线、短横线，最多 48 字符')
            key=label+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]
            self.session=SESSIONS/key;self.session.mkdir(parents=True,exist_ok=False)
            self.cfg={'session_dir':str(self.session),'dataset_root':str(PROJECT/'datasets'/key),'repo_id':'local/'+key,'cameras':cameras,'fps':20,'monitor_url':f'http://127.0.0.1:{self.port}','benchmark_notes':str(data.get('notes',''))[:2000],'max_relative_target':None,'task_type':'pick_and_lift'}
            if resume:self.cfg={**previous,**{'session_dir':str(self.session),'monitor_url':f'http://127.0.0.1:{self.port}'}}
            self.cfg.update(resume=bool(resume),target_successes=target)
            (self.session/'config.json').write_text(json.dumps(self.cfg,ensure_ascii=False,indent=2))
            self.log=(self.session/'worker.log').open('a')
            self.process=subprocess.Popen([str(PYTHON),'-u',str(ROOT/'collection_worker.py'),str(self.session/'config.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,cwd=PROJECT)
            self.state={'mode':'connecting','saved':len(labels),'successes':successes,'target':target,'frames':0,'error':None}
            threading.Thread(target=self.read_output,args=(self.process,self.session,self.log),daemon=True).start()

    def read_output(self,process,session,log):
        for line in process.stdout:
            log.write(line);log.flush()
            if line.startswith('@@'):
                try:
                    with self.lock:
                        if self.process is process:self.state=json.loads(line[2:])
                except ValueError:pass
        code=process.wait()
        with self.lock:
            if self.process is process and self.state['mode'] not in ['closed','error']:
                self.state.update(mode='error',error=f'采集进程退出（{code}），日志：{session / "worker.log"}')
        log.close()

    def command(self,data):
        with self.lock:
            if not self.running():
                if data.get('action')=='heartbeat':return
                raise ValueError('采集会话未启动')
            if data.get('action') not in ['heartbeat','follow','hold','record','finish','stop']:raise ValueError('未知操作')
            try:self.process.stdin.write(json.dumps(data,ensure_ascii=False)+'\n');self.process.stdin.flush()
            except BrokenPipeError:raise ValueError('采集进程已退出')

    def validate(self):
        with self.lock:
            if self.running() or self.validating:raise ValueError('先结束采集会话，再验证数据集')
            if not self.cfg:raise ValueError('尚无数据集')
            self.validating=True
            threading.Thread(target=self.run_validation,daemon=True).start()

    def run_validation(self):
        try:
            r=subprocess.run([str(PYTHON),str(ROOT/'validate_collection.py'),self.cfg['dataset_root']],capture_output=True,text=True,timeout=300)
            if r.returncode:
                with self.lock:self.state['error']='验证失败：'+r.stdout[-1500:]+r.stderr[-500:]
        except Exception as e:
            with self.lock:self.state['error']=str(e)
        finally:self.validating=False

    def media_path(self,idx,role="scene"):
        if not self.cfg:raise ValueError('无数据集')
        root=Path(self.cfg['dataset_root']).resolve()
        ann=root/'meta/benchmark_episodes.jsonl'
        labels=[json.loads(s) for s in ann.read_text().splitlines() if s.strip()]
        label=next(a for a in labels if a['episode_index']==idx)
        media=label.get('videos',{'scene':label}).get(role)
        if media is None:raise ValueError('无此相机视角')
        p=(root/media['video_path']).resolve()
        if not p.is_relative_to(root) or p.suffix!='.mp4':raise ValueError('视频路径无效')
        return p
