"""Single owner of robot ports. Started only by an explicit collection UI action.
Uses official SO classes and LeRobot v3 writer. Never changes calibration.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

TASK = 'Pick up the black 25 mm cube from the blue spawn region, lift it approximately 5 cm above the table, and hold it for 1 second.'
NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
KEYS = [n + '.pos' for n in NAMES]


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def emit(**data):
    print('@@' + json.dumps(data, ensure_ascii=False), flush=True)


def camera_configs(cfg):
    return cfg.get("cameras") or {"scene": {"id": cfg.get("camera_id"), "height": cfg.get("height"), "width": cfg.get("width")}}


def feature_spec(cameras):
    result = {
        'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': KEYS},
        'action': {'dtype': 'float32', 'shape': (6,), 'names': KEYS},
        'telemetry.requested_action': {'dtype': 'float32', 'shape': (6,), 'names': KEYS},
    }
    for role, camera in cameras.items():
        result['observation.images.'+role] = {'dtype':'video','shape':(camera['height'],camera['width'],3),'names':['height','width','channels']}
    for k in ['observation_monotonic_s', 'action_monotonic_s', 'wall_time_s'] + ['camera_'+role+'_received_monotonic_s' for role in cameras]:
        result['telemetry.'+k] = {'dtype': 'float64', 'shape': (1,), 'names': [k]}
    return result


def make_frame(obs, requested, sent, images, obs_t, act_t):
    frame = {'observation.state': np.array([obs[k] for k in KEYS], dtype=np.float32),
             'action': np.array([sent[k] for k in KEYS], dtype=np.float32),
             'telemetry.requested_action': np.array([requested[k] for k in KEYS], dtype=np.float32),
             'task': TASK}
    for role,(rgb,stamp) in images.items():
        frame['observation.images.'+role]=rgb
        frame['telemetry.camera_'+role+'_received_monotonic_s']=np.array([stamp],dtype=np.float64)
    for k,v in [('observation_monotonic_s',obs_t),('action_monotonic_s',act_t),('wall_time_s',time.time())]:
        frame['telemetry.'+k] = np.array([v],dtype=np.float64)
    return frame


class Hardware:
    def __init__(self):
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.teleoperators.so_leader.so_leader import SOLeader
        from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
        self.robot = SOFollower(SOFollowerRobotConfig(port='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61035080-if00',id='R12252801',use_degrees=True,max_relative_target=None,cameras={}))
        self.leader = SOLeader(SOLeaderTeleopConfig(port='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61036020-if00',id='R12252802',use_degrees=True))

    def connect(self):
        # Refuse occupied ports instead of contending with terminal teleoperation.
        for obj in [self.leader,self.robot]:
            r=subprocess.run(['fuser',obj.config.port],capture_output=True,timeout=2)
            if r.returncode==0:
                raise RuntimeError('机械臂串口被占用：先退出终端遥操作')
        for obj in [self.leader,self.robot]:
            obj.bus.connect()
            if not obj.bus.is_calibrated:
                raise RuntimeError('现场标定与保存文件不一致；已停止，未改写标定')
        self.leader.configure()
        # Seed the follower's goal from its current pose before torque is enabled.
        # No automatic move to a stored reset pose or leader pose on connection.
        current=self.robot.bus.sync_read('Present_Position',normalize=False)
        self.robot.bus.sync_write('Goal_Position',current,normalize=False)
        self.robot.configure()
        return self.robot.get_observation()

    def observe(self):
        return self.robot.get_observation()

    def step(self):
        requested=self.leader.get_action()
        sent=self.robot.send_action(requested)
        return requested,sent

    def close(self):
        errors=[]
        # Attempt every follower motor even if one fails to answer.
        if self.robot.bus.is_connected:
            for n in NAMES:
                try:self.robot.bus.disable_torque(n,num_retry=1)
                except Exception as e:errors.append(str(e))
            self.robot.bus.disconnect(disable_torque=False)
        if self.leader.bus.is_connected:
            self.leader.bus.disconnect(disable_torque=False)
        return errors


class Collection:
    def __init__(self,cfg):
        self.cfg=cfg
        self.root=Path(cfg['dataset_root'])
        self.session=Path(cfg['session_dir'])
        self.commands=queue.Queue()
        self.mode='connecting'
        self.frames=0
        self.target=cfg.get('target_successes',10)
        if type(self.target) is not int or not 1<=self.target<=10000:raise ValueError('目标条数无效')
        self.initial_validated=False
        self.saved=0
        self.successes=0
        self.annotations=[]
        self.dataset=None
        self.hardware=Hardware()
        self.future=None
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.last_heartbeat=time.monotonic()
        self.stop=False
        self.last_publish=0
        self.episode_start=0
        self.last_camera_t=None
        self.duplicate_frames=0
        self.error=None
        self.reference_pose=None
        self.last_obs=None

    def camera(self):
        images={}
        for role, camera in camera_configs(self.cfg).items():
            req=urllib.request.Request(self.cfg['monitor_url']+'/api/frame/'+urllib.parse.quote(camera['id'],safe=''))
            with urllib.request.urlopen(req,timeout=.8) as response:
                stamp=float(response.headers['X-Capture-Monotonic'])
                data=response.read()
            bgr=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2]!=(camera['height'],camera['width']):
                raise RuntimeError(role+' 相机帧解码或尺寸校验失败')
            images[role]=(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),stamp)
        now=time.monotonic()
        for role,(_,stamp) in images.items():
            if now-stamp>.2 or stamp>now:
                raise RuntimeError(role+' 相机帧超过 200 ms 未更新或时间异常，已停止录制')
        stamps=[stamp for _,stamp in images.values()]
        if max(stamps)-min(stamps)>.1:
            raise RuntimeError('双视角帧到达时间差超过 100 ms，已停止录制')
        return images

    def prepare_resume(self):
        if not self.cfg.get('resume'):return
        from collection import read_existing
        original,labels,info=read_existing(self.root)
        if self.cfg['cameras']!=original['cameras'] or self.cfg['fps']!=info['fps'] or original['task']!=TASK:
            raise ValueError('续录相机、帧率或任务不匹配')
        for key,spec in feature_spec(camera_configs(self.cfg)).items():
            expected=json.loads(json.dumps(spec))
            actual=info['features'].get(key,{})
            if any(actual.get(k)!=v for k,v in expected.items()):raise ValueError('续录数据字段不匹配：'+key)
        for obj in [self.hardware.robot,self.hardware.leader]:
            path=str(obj.calibration_fpath)
            if original['calibration_sha256'].get(path)!=hashlib.sha256(obj.calibration_fpath.read_bytes()).hexdigest():
                raise ValueError('标定文件已改变，不能直接续录原数据集')
        self.annotations=labels;self.saved=len(labels);self.successes=sum(e['human_success'] for e in labels)
        if self.target<=self.successes:raise ValueError('目标成功总条数必须大于已有成功数')
        self.reference_pose=labels[0]['reset_reference_pose'] if labels else None
        report_path=self.root/'meta/validation.json'
        if report_path.exists():
            report=json.loads(report_path.read_text())
            self.initial_validated=bool(report.get('passed') and len(report.get('valid_success_episodes',[]))>=10 and report.get('total_frames')==info['total_frames'])
        if self.successes>=10 and not self.initial_validated:
            raise ValueError('已有至少 10 条成功示范；请先完成当前数据集技术校验，再续录')

    def create_dataset(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        if self.cfg.get('resume'):
            self.dataset=LeRobotDataset.resume(self.cfg['repo_id'],root=self.root,image_writer_threads=4,vcodec='h264',encoder_threads=2)
            return
        self.dataset=LeRobotDataset.create(repo_id=self.cfg['repo_id'],root=self.root,fps=self.cfg['fps'],robot_type='so_follower',features=feature_spec(camera_configs(self.cfg)),use_videos=True,image_writer_threads=4,vcodec='h264',encoder_threads=2,metadata_buffer_size=1)
        atomic_json(self.root/'meta/benchmark.json',{**self.cfg,'task':TASK,'cube_dimensions_mm':[25,25,25],'lift_height_mm':50,'hold_seconds':1,'physical_measurements_verified':False,'camera_timestamp_definition':'host monotonic receipt of a complete JPEG, not sensor exposure time','action_definition':'joint target returned by official SOFollower.send_action; body degrees, gripper 0-100','timestamp_definition':'LeRobot nominal index/fps; telemetry holds measured host times','calibration_sha256':{str(o.calibration_fpath):hashlib.sha256(o.calibration_fpath.read_bytes()).hexdigest() for o in [self.hardware.robot,self.hardware.leader]}})

    def status(self,force=False):
        now=time.monotonic()
        if not force and now-self.last_publish<.3:return
        self.last_publish=now
        data={'mode':self.mode,'frames':self.frames,'elapsed_s':round(now-self.episode_start,1) if self.mode=='recording' else 0,'saved':self.saved,'successes':self.successes,'target':self.target,'dataset_root':str(self.root),'error':self.error,'last_episode':self.annotations[-1] if self.annotations else None,'joint_state':self.last_obs}
        atomic_json(self.session/'status.json',data)
        emit(**data)

    def finish(self,outcome,note=''):
        if self.mode!='recording':return
        if self.frames<2:
            self.dataset.clear_episode_buffer()
            self.mode='holding';self.frames=0
            self.error='本条不足两帧，未保存；请重新开始'
            return
        index=self.saved
        metadata={'episode_index':index,'outcome':outcome,'human_success':outcome=='success','note':note,'frames':self.frames,'duration_s':time.monotonic()-self.episode_start,'duplicate_camera_frames':self.duplicate_frames,'start_pose':self.episode_pose,'reset_reference_pose':self.reference_pose,'reset_max_joint_error':max(abs(self.episode_pose[k]-self.reference_pose[k]) for k in KEYS),'utc_finished':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        atomic_json(self.session/'pending_episode.json',metadata)
        self.mode='saving'
        self.future=self.pool.submit(self.save,metadata)

    def save(self,metadata):
        for name in ['validation.json','act_success_episodes.json']:
            path=self.root/'meta'/name
            if path.exists():path.rename(self.session/('previous_'+name))
        self.dataset.save_episode(parallel_encoding=False)
        # Close Parquet footers for immediate offline QA and replay; reopen only
        # on the NEXT record command. Robot stays at its last commanded target.
        self.dataset.finalize()
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        meta=LeRobotDatasetMetadata(self.cfg['repo_id'],root=self.root)
        ep=meta.episodes[metadata['episode_index']]
        metadata['videos']={}
        for role in camera_configs(self.cfg):
            key='observation.images.'+role
            path=meta.get_video_file_path(metadata['episode_index'],key)
            metadata['videos'][role]={'video_path':str(path),'video_from_s':ep[f'videos/{key}/from_timestamp'],'video_to_s':ep[f'videos/{key}/to_timestamp']}
        with (self.root/'meta/benchmark_episodes.jsonl').open('a') as f:
            f.write(json.dumps(metadata,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        return metadata

    def handle(self,cmd):
        action=cmd.get('action')
        if action=='heartbeat':self.last_heartbeat=time.monotonic();return
        if action=='stop':self.stop=True;return
        if self.mode=='saving':return
        if action=='follow' and self.mode=='holding':self.mode='reset';self.error=None
        elif action=='hold' and self.mode=='reset':self.mode='holding'
        elif action=='record' and self.mode in ['holding','reset']:
            if self.successes>=self.target:
                self.error=f'已达到目标 {self.target} 条成功示范；可结束会话后提高目标并续录';return
            if self.successes>=10 and not self.initial_validated:
                self.error='首批 10 条已完成，请结束会话并验证，再续录至目标总数';return
            self.camera()  # Require a fresh decodable frame before accepting.
            if self.dataset._is_finalized:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                self.dataset=LeRobotDataset.resume(self.cfg['repo_id'],root=self.root,image_writer_threads=4,vcodec='h264',encoder_threads=2)
            self.frames=0;self.duplicate_frames=0;self.last_camera_t=None
            self.episode_pose=self.hardware.observe()
            self.episode_start=time.monotonic();self.mode='recording';self.error=None
        elif action=='finish' and cmd.get('outcome') in ['success','failure','excluded']:
            self.finish(cmd['outcome'],str(cmd.get('note',''))[:1000])

    def run(self):
        try:
            self.prepare_resume()
            self.camera()
            current_pose=self.hardware.connect()
            if self.reference_pose is None:self.reference_pose=current_pose
            self.last_obs=current_pose
            self.create_dataset()
            self.mode='holding';self.status(True)
            while not self.stop:
                tick=time.monotonic()
                while not self.commands.empty():self.handle(self.commands.get_nowait())
                if self.stop:break
                if tick-self.last_heartbeat>10:raise RuntimeError('网页连接中断超过 10 秒，已退出遥操')
                if self.future and self.future.done():
                    metadata=self.future.result();self.future=None
                    self.annotations.append(metadata);self.saved+=1;self.successes+=metadata['human_success'];self.mode='holding'
                if self.mode in ['recording','reset']:
                    images=self.camera()
                    obs=self.hardware.observe();obs_t=time.monotonic();self.last_obs=obs
                    requested,sent=self.hardware.step();act_t=time.monotonic()
                    if self.mode=='recording':
                        self.dataset.add_frame(make_frame(obs,requested,sent,images,obs_t,act_t))
                        self.frames+=1
                        stamps={role:stamp for role,(_,stamp) in images.items()}
                        if self.last_camera_t and any(stamps[k]==self.last_camera_t[k] for k in stamps):self.duplicate_frames+=1
                        self.last_camera_t=stamps
                self.status()
                time.sleep(max(0,1/self.cfg['fps']-(time.monotonic()-tick)))
        except BaseException as e:
            self.error=f'{type(e).__name__}: {e}'
        finally:
            # Stop robot actuation before waiting on disk/video operations.
            try:
                cleanup=self.hardware.close()
                if cleanup:self.error=(self.error or '')+' 释放力矩未完全确认：'+'; '.join(cleanup)
            except Exception as e:self.error=(self.error or '')+' 断开异常：'+str(e)
            try:
                if self.mode=='recording':self.finish('aborted',self.error or '用户结束会话')
                if self.future:
                    item=self.future.result();self.annotations.append(item);self.saved+=1;self.successes+=item['human_success']
                if self.dataset:self.dataset.finalize()
            except Exception as e:self.error=(self.error or '')+' 保存异常：'+str(e)
            self.pool.shutdown(wait=True)
            self.mode='error' if self.error else 'closed';self.status(True)


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text())
    collector=Collection(cfg)
    def reader():
        for line in sys.stdin:
            try:collector.commands.put(json.loads(line))
            except json.JSONDecodeError:pass
        collector.commands.put({'action':'stop'})
    threading.Thread(target=reader,daemon=True).start()
    signal.signal(signal.SIGTERM,lambda *_:collector.commands.put({'action':'stop'}))
    collector.run()


if __name__=='__main__':main()
