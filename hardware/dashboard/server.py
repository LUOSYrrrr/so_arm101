#!/usr/bin/env python3
"""Local USB monitor and explicitly started collection subprocess.
Camera capture uses the current V4L2 format; no format/control setters.
"""
import argparse
import logging
from logging.handlers import RotatingFileHandler
import json
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, unquote
from collection import CollectionManager

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger('hardware_monitor')
CALIB = Path.home() / '.cache/huggingface/lerobot/calibration'
# User identified this exact device as the computer's built-in webcam.
EXCLUDED_CAMERAS = {'usb-Sonix_Technology_Co.__Ltd._USB2.0_HD_UVC_WebCam-video-index0'}


def devices(kind):
    if kind != 'camera':
        return sorted(p for p in Path('/dev/serial/by-id').glob('*') if p.exists())
    # Identical cameras without serial numbers collide in by-id. USB paths
    # distinguish ports; deduplicate the usb/usbv2 aliases by resolved node.
    excluded = {(Path('/dev/v4l/by-id') / name).resolve() for name in EXCLUDED_CAMERAS}
    unique = {}
    for p in sorted(Path('/dev/v4l/by-path').glob('*video-index0')):
        if p.exists() and p.resolve() not in excluded:
            unique.setdefault(p.resolve(), p)
    return list(unique.values())


def camera_format(path):
    r = subprocess.run(['v4l2-ctl', '-d', str(path), '--get-fmt-video', '--get-parm'], capture_output=True, text=True, timeout=3)
    if r.returncode:
        raise RuntimeError(r.stderr.strip() or '无法读取相机格式')
    size = re.search(r'Width/Height\s*:\s*(\d+)/(\d+)', r.stdout)
    fmt = re.search(r"Pixel Format\s*:\s*'([^']+)'", r.stdout)
    fps = re.search(r'Frames per second\s*:\s*([\d.]+)', r.stdout)
    return {'width': int(size[1]) if size else None, 'height': int(size[2]) if size else None,
            'format': fmt[1] if fmt else None, 'configured_fps': float(fps[1]) if fps else None}


class Camera:
    def __init__(self, path, attempt=0):
        self.attempt = attempt
        self.started = time.monotonic()
        self.path = path
        self.lock = threading.Lock()
        self.process = None
        self.stop_event = threading.Event()
        self.jpeg = None
        self.last = None
        self.times = deque(maxlen=120)
        self.info = {}
        self.error = None
        self.diagnostics = deque(maxlen=8)
        self.stderr_thread = None
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()

    def read_diagnostics(self):
        while True:
            chunk = self.process.stderr.read1(1024)
            if not chunk:
                return
            self.diagnostics.append(chunk.decode(errors='replace'))

    def capture(self):
        try:
            self.info = camera_format(self.path)
            if self.info['format'] != 'MJPG':
                raise RuntimeError('当前格式不是 MJPG；为保留配置，不自动切换格式')
            with self.lock:
                if self.stop_event.is_set():
                    return
                # No serial access, no camera format/control changes.
                self.process = subprocess.Popen(['v4l2-ctl', '-d', str(self.path), '--stream-mmap=3', '--stream-poll', '--stream-to=/dev/stdout'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.stderr_thread = threading.Thread(target=self.read_diagnostics, daemon=True)
                self.stderr_thread.start()
            buffer = bytearray()
            while not self.stop_event.is_set():
                chunk = self.process.stdout.read1(262144)
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b'\xff\xd8')
                    if start < 0:
                        if buffer and buffer[0] != 255:
                            self.diagnostics.append(bytes(buffer[:1200]).decode(errors='replace'))
                        buffer[:] = buffer[-1:]
                        break
                    end = buffer.find(b'\xff\xd9', start + 2)
                    if end < 0:
                        del buffer[:start]
                        break
                    frame = bytes(buffer[start:end+2])
                    del buffer[:end+2]
                    now = time.monotonic()
                    with self.lock:
                        self.jpeg, self.last = frame, now
                        self.times.append(now)
                if len(buffer) > 16 * 1024 * 1024:
                    raise RuntimeError('相机帧解析失败')
            if not self.stop_event.is_set():
                if self.stderr_thread:
                    self.stderr_thread.join(timeout=1)
                detail = ''.join(self.diagnostics).strip()[-1200:]
                if 'No space left on device' in detail:
                    self.error = 'USB 视频流资源不足：请将一台相机接到另一组 USB 接口（保持机位不变）。' + detail
                elif 'select timeout' in detail:
                    self.error = '相机等待帧超时：请检查 USB 连接，释放后重试。' + detail
                else:
                    self.error = '采集结束：' + (detail or '没有返回帧，请释放后重试')
        except Exception as e:
            self.error = str(e)
        finally:
            if self.error:
                LOG.error('%s attempt=%s %s', self.path, self.attempt, self.error)
            if self.process:
                if self.process.poll() is None:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.process.stdout.close()
                if self.stderr_thread:
                    self.stderr_thread.join(timeout=1)
                self.process.stderr.close()

    def close(self):
        self.stop_event.set()
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
        self.thread.join(timeout=4)

    def status(self):
        with self.lock:
            age = None if self.last is None else time.monotonic() - self.last
            fps = (len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times)>1 else 0
            return {**self.info, 'age_s': age, 'fps': round(fps, 1) if age is not None and age < 2 else 0,
                    'retry_attempt': self.attempt, 'live': age is not None and age < 2 and not self.error, 'error': self.error}


class Monitor:
    def __init__(self):
        self.lock = threading.RLock()
        self.cameras = {}
        self.enabled = True
        self.last_client = time.monotonic()
        threading.Thread(target=self.watch, daemon=True).start()

    def release(self):
        with self.lock:
            for camera in self.cameras.values():
                camera.close()
            self.cameras.clear()

    def watch(self):
        while True:
            time.sleep(2)
            with self.lock:
                if time.monotonic() - self.last_client > 15:
                    self.release()

    def status(self):
        with self.lock:
            self.last_client = time.monotonic()
            paths = devices('camera')
            for key in list(self.cameras):
                if key not in [p.name for p in paths]:
                    self.cameras.pop(key).close()
            if self.enabled:
                for path in paths:
                    if path.name not in self.cameras:
                        self.cameras[path.name] = Camera(path)
                    else:
                        old = self.cameras[path.name]
                        if (not old.thread.is_alive() and old.error and old.attempt < 2
                                and time.monotonic() - old.started > 10):
                            old.close()
                            self.cameras[path.name] = Camera(path, old.attempt + 1)
            cams = []
            for p in paths:
                camera = self.cameras.get(p.name)
                cams.append({'id': p.name, 'device': str(p.resolve()), 'path': str(p),
                             **(camera.status() if camera else {'live': False, 'fps': 0, 'age_s': None})})
            calibration = []
            for kind, ident in [('robots/so_follower', 'R12252801'), ('teleoperators/so_leader', 'R12252802')]:
                path = CALIB / kind / (ident + '.json')
                try:
                    data = json.loads(path.read_text())
                    valid = len(data) == 6 and all(v['range_min'] < v['range_max'] for v in data.values())
                except (OSError, ValueError, KeyError, TypeError):
                    valid = False
                calibration.append({'id': ident, 'role': '从臂' if kind.startswith('robots') else '主臂', 'file_valid': valid})
            return {'time': time.strftime('%H:%M:%S'), 'preview_enabled': self.enabled, 'cameras': cams,
                    'robots': [{'id': p.name, 'device': str(p.resolve()), 'path': str(p)} for p in devices('robot')], 'calibration': calibration}


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, content, mime='application/json; charset=utf-8'):
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/':
            self.reply(200, (ROOT/'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif path == '/api/status':
            snapshot = self.server.monitor.status()
            snapshot['collection'] = self.server.collection.status()
            self.reply(200, json.dumps(snapshot).encode())
        elif path.startswith('/api/replay/'):
            try: self.video(self.server.collection.media_path(int(path.rsplit('/', 1)[1]),parse_qs(urlsplit(self.path).query).get('view',['scene'])[0]))
            except (ValueError, OSError, StopIteration): self.reply(404, b'{}')
        elif path.startswith('/api/frame/'):
            key = unquote(path.removeprefix('/api/frame/'))
            with self.server.monitor.lock:
                camera = self.server.monitor.cameras.get(key)
                if camera and camera.status()['live']:
                    with camera.lock:
                        frame, stamp = camera.jpeg, camera.last
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame)))
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('X-Capture-Monotonic', str(stamp))
                    self.end_headers()
                    try: self.wfile.write(frame)
                    except (BrokenPipeError, ConnectionResetError): pass
                else:
                    self.reply(503, b'{"error":"no fresh frame"}')
        else:
            self.reply(404, b'{}')

    def video(self, path):
        size=path.stat().st_size
        start,end=0,size-1
        requested=self.headers.get('Range')
        if requested:
            match=re.fullmatch(r'bytes=(\d+)-(\d*)',requested)
            if not match:
                self.reply(416,b'{}');return
            start=int(match[1]);end=min(int(match[2]) if match[2] else end,end)
            if start>end or start>=size:
                self.reply(416,b'{}');return
        self.send_response(206 if requested else 200)
        self.send_header('Content-Type','video/mp4')
        self.send_header('Accept-Ranges','bytes')
        self.send_header('Content-Length',str(end-start+1))
        if requested:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as f:
            f.seek(start);remaining=end-start+1
            try:
                while remaining:
                    chunk=f.read(min(65536,remaining))
                    if not chunk:break
                    self.wfile.write(chunk);remaining-=len(chunk)
            except (BrokenPipeError,ConnectionResetError):pass

    def do_POST(self):
        # Only our own page can change capture ownership. No CORS access.
        if self.headers.get('Origin') != f'http://{self.headers.get("Host")}' or self.headers.get('X-Monitor') != '1':
            self.reply(403, b'{}')
            return
        path = urlsplit(self.path).path
        if path.startswith('/api/collection/'):
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<=length<=8192:raise ValueError('请求过大')
                data=json.loads(self.rfile.read(length) or b'{}')
                if not isinstance(data,dict):raise ValueError('请求格式错误')
                if path=='/api/collection/start': self.server.collection.start(data)
                elif path=='/api/collection/select': self.server.collection.select(data)
                elif path=='/api/collection/command': self.server.collection.command(data)
                elif path=='/api/collection/validate': self.server.collection.validate()
                else:raise ValueError('未知采集操作')
                self.reply(200,b'{"ok":true}')
            except (ValueError,OSError,subprocess.SubprocessError) as e:
                self.reply(400,json.dumps({'error':str(e)},ensure_ascii=False).encode())
            return
        if self.server.collection.running():
            self.reply(409,json.dumps({'error':'采集会话正在使用相机，请先结束会话'},ensure_ascii=False).encode());return
        if path not in ['/api/pause', '/api/resume']:
            self.reply(404, b'{}')
            return
        with self.server.monitor.lock:
            self.server.monitor.enabled = path.endswith('resume')
            if not self.server.monitor.enabled:
                self.server.monitor.release()
        self.reply(200, b'{"ok":true}')

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    handler = RotatingFileHandler(ROOT / 'capture.log', maxBytes=1_000_000, backupCount=2)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.monitor = Monitor()
    server.collection = CollectionManager(server.monitor, args.port)
    print(f'Hardware monitor: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server.collection.running():
            server.collection.command({'action':'stop'})
        server.monitor.enabled = False
        server.monitor.release()
        server.server_close()


if __name__ == '__main__':
    main()
