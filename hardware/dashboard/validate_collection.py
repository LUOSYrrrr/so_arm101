"""Offline validation only; never imports robot drivers or opens serial ports."""
import json
import sys
from pathlib import Path
import av
import numpy as np
import pandas as pd


def validate(root):
    root=Path(root);issues=[];warnings=[];episodes=[];selected=[]
    info=json.loads((root/'meta/info.json').read_text())
    if info.get('codebase_version')!='v3.0':issues.append('数据格式不是 v3.0')
    files=sorted((root/'data').rglob('*.parquet'))
    if not files:return {'passed':False,'issues':['数据集还没有已保存帧'],'warnings':[],'episodes':[]}
    table=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    labels=[json.loads(s) for s in (root/'meta/benchmark_episodes.jsonl').read_text().splitlines() if s.strip()]
    if len(table)!=info['total_frames']:issues.append('info 总帧数与 Parquet 不一致')
    if len(labels)!=info['total_episodes']:issues.append('结果标注数量与 episode 总数不一致')
    decoded={}
    for label in labels:
        idx=label['episode_index'];frame=table[table.episode_index==idx].sort_values('frame_index');errors=[];warn=[]
        if len(frame)!=label['frames'] or len(frame)<2:errors.append('帧数不一致或不足')
        if not np.array_equal(frame.frame_index.to_numpy(),np.arange(len(frame))):errors.append('帧索引不连续')
        for col in ['observation.state','action','telemetry.requested_action']:
            a=np.stack(frame[col])
            if a.shape!=(len(frame),6) or not np.isfinite(a).all():errors.append(col+' 含异常数值或尺寸')
        nominal=frame.timestamp.to_numpy()
        if not np.allclose(nominal,np.arange(len(frame))/info['fps'],atol=1e-4):errors.append('名义时间戳异常')
        times=np.array([float(np.asarray(v).reshape(-1)[0]) for v in frame['telemetry.observation_monotonic_s']])
        act=np.array([float(np.asarray(v).reshape(-1)[0]) for v in frame['telemetry.action_monotonic_s']])
        if not (np.isfinite(times).all() and np.isfinite(act).all()):errors.append('实测时间戳含非有限值')
        if np.any(np.diff(times)<=0) or np.any(act<times):errors.append('实测时间戳顺序异常')
        duration=float(times[-1]-times[0]) if len(times)>1 else 0
        fps=(len(times)-1)/duration if duration>0 else 0
        if fps<info['fps']*.85:errors.append('实际平均采集频率低于目标的 85%')
        if len(times)>1 and np.max(np.diff(times))>3/info['fps']:errors.append('采集间隔超过三个周期')
        cameras={}
        video_features={k.removeprefix('observation.images.'):v for k,v in info['features'].items() if v.get('dtype')=='video'}
        media=label.get('videos',{'scene':label})
        for role,feature in video_features.items():
            col='telemetry.camera_'+role+'_received_monotonic_s'
            if role=='scene' and col not in frame:col='telemetry.camera_received_monotonic_s'
            if col not in frame:errors.append(role+' 相机时间戳缺失');continue
            cam=np.array([float(np.asarray(v).reshape(-1)[0]) for v in frame[col]])
            cameras[role]=cam
            if not np.isfinite(cam).all() or np.any(np.diff(cam)<0) or np.any(cam>times):errors.append(role+' 相机时间戳异常')
            if np.any(times-cam>.2):errors.append(role+' 相机帧年龄超过 200 ms')
            if len(cam)<2 or np.mean(np.diff(cam)==0)>.1:errors.append(role+' 相机重复时间戳超过 10%')
            entry=media.get(role)
            if not entry:errors.append(role+' 视频元数据缺失');continue
            video=(root/entry['video_path']).resolve()
            if not video.is_relative_to(root.resolve()):errors.append(role+' 视频路径越界')
            elif not video.exists():errors.append(role+' 视频缺失')
            else:
                if video not in decoded:
                    pts=[]
                    try:
                        with av.open(str(video)) as container:
                            for rgb in container.decode(video=0):
                                pts.append(float(rgb.time))
                                if [rgb.height,rgb.width,3]!=feature['shape']:raise ValueError('视频尺寸异常')
                        decoded[video]=pts
                    except Exception as e:decoded[video]=[];errors.append(role+' 视频解码失败：'+str(e))
                count=sum(entry['video_from_s']-1e-5<=t<entry['video_to_s']-1e-5 for t in decoded[video])
                if abs(count-len(frame))>1:errors.append(role+' 视频区间帧数与状态帧数不一致')
        skew_ms=0
        if len(cameras)>1:
            skew_ms=float(np.max(np.ptp(np.stack(list(cameras.values())),axis=0))*1000)
            if skew_ms>100:errors.append('双视角帧到达时间差超过 100 ms')
        if label.get('reset_max_joint_error',0)>5:warn.append('起始姿态相对参考姿态偏差超过 5（角度/夹爪百分点）')
        if label['human_success'] and not errors:selected.append(idx)
        episodes.append({'episode_index':idx,'outcome':label['outcome'],'frames':len(frame),'actual_fps':round(fps,2),'duration_s':round(label['duration_s'],2),'max_camera_skew_ms':round(skew_ms,2),'passed':not errors,'issues':errors,'warnings':warn})
        issues.extend(f'episode {idx}: {e}' for e in errors)
        warnings.extend(f'episode {idx}: {e}' for e in warn)
    warnings.append('技术校验不自动判定抓取成功；需回放逐条复核 5 cm 抬升与 1 秒保持。物体质量、生成区域、工作空间等实测信息仍需补齐。')
    report={'passed':not issues,'format':info.get('codebase_version'),'episodes':episodes,'total_frames':len(table),'valid_success_episodes':selected,'issues':issues,'warnings':warnings}
    (root/'meta/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (root/'meta/act_success_episodes.json').write_text(json.dumps(selected))
    return report


if __name__=='__main__':
    try:print(json.dumps(validate(sys.argv[1]),ensure_ascii=False))
    except Exception as e:print(json.dumps({'passed':False,'issues':[str(e)],'warnings':[],'episodes':[]},ensure_ascii=False));sys.exit(1)
