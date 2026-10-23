import base64
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import backend_api
from agent.video_preview import VideoPreviews, create_video_router


class VideoPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'dashcam.avi'
        self.source.write_bytes(b'original-video')
        self.files = {'video-test': self.source, 'text': self.root/'notes.txt'}
        self.files['text'].write_text('notes')
        app = FastAPI()
        app.include_router(create_video_router(self.files, self.root/'previews'))
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_missing_files_and_nonvideo_are_rejected(self):
        self.assertEqual(self.client.get('/api/files/missing/video-preview').status_code,404)
        self.assertEqual(self.client.get('/api/files/text/video-preview').status_code,415)

    def test_missing_ffmpeg_and_failed_transcode(self):
        with patch('agent.video_preview.shutil.which',return_value=None):
            self.assertEqual(self.client.get('/api/files/video-test/video-preview').status_code,503)
        with patch('agent.video_preview.shutil.which',return_value='/usr/bin/ffmpeg'), \
             patch('agent.video_preview.subprocess.run',side_effect=subprocess.CalledProcessError(1,'ffmpeg',stderr=b'/private/input/path')):
            response = self.client.get('/api/files/video-test/video-preview')
        self.assertEqual(response.status_code,422)
        self.assertNotIn('/private',response.text)
        self.assertFalse(list((self.root/'previews').glob('*.partial.mp4')))

    def test_timeout_cleans_partial_output(self):
        def timeout(command,**kwargs):
            Path(command[-1]).write_bytes(b'partial')
            raise subprocess.TimeoutExpired(command,120)
        with patch('agent.video_preview.shutil.which',return_value='/usr/bin/ffmpeg'), \
             patch('agent.video_preview.subprocess.run',side_effect=timeout):
            response = self.client.get('/api/files/video-test/video-preview')
        self.assertEqual(response.status_code,504)
        self.assertFalse(list((self.root/'previews').glob('*.partial.mp4')))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
    def test_real_transcode_range_seek_and_cache_preserve_original(self):
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
                        '-f','lavfi','-i','testsrc=size=160x90:rate=10:duration=1',
                        '-c:v','mjpeg','-threads','1',str(self.source)],check=True)
        original = self.source.read_bytes()
        response = self.client.get('/api/files/video-test/video-preview')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.headers['content-type'],'video/mp4')
        self.assertIn('inline',response.headers['content-disposition'])
        cached = next((self.root/'previews').glob('*.mp4'))
        probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                                'stream=codec_name,pix_fmt','-of','json',str(cached)],check=True,capture_output=True)
        self.assertEqual(json.loads(probe.stdout)['streams'][0],{'codec_name':'h264','pix_fmt':'yuv420p'})
        with patch('agent.video_preview.subprocess.run',side_effect=AssertionError('Cached preview must not re-encode')):
            ranged = self.client.get('/api/files/video-test/video-preview',headers={'Range':'bytes=10-29'})
        self.assertEqual(ranged.status_code,206)
        self.assertEqual(ranged.content,response.content[10:30])
        self.assertEqual(ranged.headers['content-range'],f'bytes 10-29/{len(response.content)}')
        self.assertEqual(self.source.read_bytes(),original)


class ChatVideoUploadTests(unittest.TestCase):
    def test_upload_urls_in_stream_and_json_and_history_mime(self):
        result={'messages':[AIMessage(content='<Conclusion>视频已收到</Conclusion>')]}
        class Handle:
            def invoke(self,*args,**kwargs): return result
            def stream(self,*args,**kwargs): yield {'type':'final','response':result}
            def review_answer_and_artifacts(self,raw,*args,**kwargs): return raw
        with tempfile.TemporaryDirectory() as tmp:
            for endpoint in ('/api/chat','/api/chat/stream'):
                session=backend_api.ChatSession('video-upload',Path(tmp),handle=Handle())
                with patch.object(backend_api,'get_session',return_value=session), \
                     patch.object(backend_api.db,'create_session'), \
                     patch.object(backend_api.db,'save_chat_message'), \
                     patch.object(backend_api,'maybe_generate_session_title'):
                    client=TestClient(backend_api.app)
                    try:
                        response=client.post(endpoint,data={'session_id':'video-upload','message':'分析视频'},
                                             files={'files':('drive.mp4',b'video-bytes','video/mp4')})
                        self.assertEqual(response.status_code,200)
                        if endpoint.endswith('/stream'):
                            blocks=response.text.split('\n\n')
                            self.assertTrue(blocks[0].startswith('event: uploaded\n'))
                            files=json.loads(blocks[0].split('data: ',1)[1])['files']
                        else:
                            files=response.json()['uploaded_files']
                        self.assertEqual(files[0]['name'],'drive.mp4')
                        ranged=client.get(files[0]['url'],headers={'Range':'bytes=0-4'})
                        self.assertEqual(ranged.status_code,206)
                        self.assertEqual(ranged.content,b'video')
                    finally: client.close()
            with patch.object(backend_api,'TEMP_BASE',Path(tmp)):
                restored=backend_api._serialize_message({'id':1,'role':'user','attachment_files':[
                    {'name':'drive.mp4','mime_type':'video/mp4','data_base64':base64.b64encode(b'video').decode()}]})
            self.assertEqual(restored['attachments'][0]['mime_type'],'video/mp4')


if __name__=='__main__':
    unittest.main()
