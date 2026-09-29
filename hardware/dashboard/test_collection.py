"""Hardware-free integration test: real v3 videos/Parquet, fake robot/camera."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import collection_worker as worker
from validate_collection import validate


class FakeHardware:
    def __init__(self, calibration):
        self.robot=SimpleNamespace(calibration_fpath=calibration)
        self.leader=SimpleNamespace(calibration_fpath=calibration)
        self.steps=0;self.closed=False
    def connect(self):return self.observe()
    def observe(self):return {k:float(i) for i,k in enumerate(worker.KEYS)}
    def step(self):
        self.steps+=1
        return ({k:float(i+1) for i,k in enumerate(worker.KEYS)},self.observe())
    def close(self):self.closed=True;return []


class Tests(unittest.TestCase):
    def test_v3_roundtrip_and_state_boundaries(self):
        with tempfile.TemporaryDirectory(prefix='so101_mock_test_') as folder:
            base=Path(folder);session=base/'session';session.mkdir();cal=base/'cal.json';cal.write_text('{}')
            fake=FakeHardware(cal)
            cfg={'dataset_root':str(base/'dataset'),'session_dir':str(session),'repo_id':'local/mock_only','task_type':'pick_and_lift','cameras':{'wrist':{'id':'a','height':48,'width':64},'third_person':{'id':'b','height':64,'width':80}},'fps':20,'monitor_url':'http://unused'}
            with patch.object(worker,'Hardware',return_value=fake),patch.object(worker,'emit'):
                c=worker.Collection(cfg)
                c.camera=lambda:{'wrist':(np.full((48,64,3),40,dtype=np.uint8),time.monotonic()),'third_person':(np.full((64,80,3),180,dtype=np.uint8),time.monotonic())}
                thread=threading.Thread(target=c.run);thread.start()
                def wait_mode(mode,timeout=30):
                    deadline=time.monotonic()+timeout
                    while c.mode!=mode:
                        if not thread.is_alive():raise AssertionError(c.error)
                        if time.monotonic()>deadline:raise AssertionError((c.mode,c.error))
                        c.commands.put({'action':'heartbeat'});time.sleep(.05)
                try:
                    wait_mode('holding');self.assertEqual(fake.steps,0)
                    # Manual reset actions are never dataset frames.
                    c.commands.put({'action':'follow'});wait_mode('reset');time.sleep(.15)
                    self.assertEqual(c.frames,0);self.assertGreater(fake.steps,0)
                    for outcome in ['success','excluded']:
                        c.commands.put({'action':'record'});wait_mode('recording');time.sleep(.65)
                        c.commands.put({'action':'finish','outcome':outcome});wait_mode('saving');wait_mode('holding')
                        count=fake.steps;time.sleep(.1);self.assertEqual(fake.steps,count)
                    self.assertEqual(c.saved,2);self.assertEqual(c.successes,1)
                finally:
                    c.commands.put({'action':'stop'});thread.join(30)
                self.assertFalse(thread.is_alive());self.assertTrue(fake.closed);self.assertIsNone(c.error)
                report=validate(base/'dataset')
                self.assertTrue(report['passed'],report)
                self.assertEqual(report['valid_success_episodes'],[0])
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                reader=LeRobotDataset('local/mock_only',root=base/'dataset',episodes=[0])
                sample=reader[0]
                self.assertEqual(tuple(sample['observation.images.wrist'].shape),(3,48,64))
                self.assertEqual(tuple(sample['observation.images.third_person'].shape),(3,64,80))
                self.assertLess(float(sample['observation.images.wrist'].mean()),float(sample['observation.images.third_person'].mean()))
                self.assertIn('max_camera_skew_ms',report['episodes'][0])
                self.assertEqual(tuple(sample['action'].shape),(6,))
                # A fresh process/session resumes without replacing existing episodes.
                import hashlib
                old_files={p:hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['data','videos'] for p in (base/'dataset'/folder).rglob('*') if p.is_file()}
                original_benchmark=(base/'dataset/meta/benchmark.json').read_bytes()
                session2=base/'session2';session2.mkdir()
                resumed=worker.Collection({**cfg,'session_dir':str(session2),'resume':True,'target_successes':2})
                resumed.camera=c.camera
                c=resumed
                thread=threading.Thread(target=c.run);thread.start()
                try:
                    wait_mode('holding');self.assertEqual(c.saved,2);self.assertEqual(c.successes,1)
                    c.commands.put({'action':'record'});wait_mode('recording');time.sleep(.65)
                    c.commands.put({'action':'finish','outcome':'success'});wait_mode('saving');wait_mode('holding')
                    self.assertEqual(c.saved,3);self.assertEqual(c.successes,2)
                    c.handle({'action':'record'});self.assertEqual(c.mode,'holding');self.assertIn('目标 2',c.error)
                    c.error=None
                finally:c.commands.put({'action':'stop'});thread.join(30)
                self.assertFalse(thread.is_alive());self.assertIsNone(c.error)
                for path,digest in old_files.items():self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
                self.assertEqual((base/'dataset/meta/benchmark.json').read_bytes(),original_benchmark)
                report=validate(base/'dataset');self.assertTrue(report['passed'],report)
                self.assertEqual(report['valid_success_episodes'],[0,2])
                import pandas as pd
                data=pd.concat([pd.read_parquet(p) for p in (base/'dataset/data').rglob('*.parquet')])
                self.assertTrue(np.all(np.stack(data['action'])!=np.stack(data['telemetry.requested_action'])))
                labels=[json.loads(line) for line in (base/'dataset/meta/benchmark_episodes.jsonl').read_text().splitlines()]
                third=base/'dataset'/labels[0]['videos']['third_person']['video_path']
                hidden=third.with_suffix('.hidden');third.rename(hidden)
                try:self.assertFalse(validate(base/'dataset')['passed'])
                finally:hidden.rename(third)
                # Corrupted timing cannot silently pass validation.
                file=next((base/'dataset/data').rglob('*.parquet'));d=pd.read_parquet(file)
                d.at[1,'telemetry.observation_monotonic_s']=0.0;d.to_parquet(file,index=False)
                self.assertFalse(validate(base/'dataset')['passed'])

    def test_each_camera_freshness_and_pair_skew(self):
        import io
        import cv2
        jpeg=cv2.imencode('.jpg',np.zeros((48,64,3),dtype=np.uint8))[1].tobytes()
        c=worker.Collection.__new__(worker.Collection)
        c.cfg={'monitor_url':'http://unused','cameras':{role:{'id':role,'height':48,'width':64} for role in ['wrist','third_person']}}
        def responses(stamps):
            result=[]
            for stamp in stamps:
                response=io.BytesIO(jpeg);response.headers={'X-Capture-Monotonic':str(stamp)};result.append(response)
            return result
        now=time.monotonic()
        with patch.object(worker.urllib.request,'urlopen',side_effect=responses([now,now-.3])):
            with self.assertRaisesRegex(RuntimeError,'third_person'):c.camera()
        now=time.monotonic()
        with patch.object(worker.urllib.request,'urlopen',side_effect=responses([now,now-.15])):
            with self.assertRaisesRegex(RuntimeError,'100 ms'):c.camera()

    def test_lost_browser_stops_without_following(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);cal=base/'cal.json';cal.write_text('{}');fake=FakeHardware(cal)
            with patch.object(worker,'Hardware',return_value=fake),patch.object(worker,'emit'):
                c=worker.Collection({'dataset_root':str(base/'dataset'),'session_dir':str(base),'fps':20})
                c.camera=lambda:(None,time.monotonic())
                c.create_dataset=lambda:None
                c.last_heartbeat=time.monotonic()-11
                c.run()
                self.assertEqual(fake.steps,0);self.assertTrue(fake.closed)
                self.assertIn('10 秒',c.error);self.assertEqual(c.mode,'error')

    def test_stale_camera_rejected_before_hardware_connect(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);cal=base/'cal.json';cal.write_text('{}');fake=FakeHardware(cal)
            with patch.object(worker,'Hardware',return_value=fake),patch.object(worker,'emit'):
                c=worker.Collection({'dataset_root':str(base/'dataset'),'session_dir':str(base)})
                c.camera=lambda:(_ for _ in ()).throw(RuntimeError('stale camera'))
                c.run();self.assertEqual(fake.steps,0);self.assertTrue(fake.closed);self.assertEqual(c.mode,'error')
                self.assertFalse((base/'dataset').exists())


if __name__=='__main__':unittest.main()
