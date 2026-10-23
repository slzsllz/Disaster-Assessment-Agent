import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

import backend_api


class TrafficVideoArtifactTests(unittest.TestCase):
    def test_preview_survives_empty_review_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            preview=Path(tmp)/'clip_accident_preview.mp4'
            preview.write_bytes(b'preview')
            empty='<Artifacts>{"display":[],"download":[]}</Artifacts>'
            images,files=backend_api.choose_frontend_artifacts(empty,[str(preview)])
            self.assertEqual(images,[])
            self.assertEqual(files,[str(preview)])
            missing=str(Path(tmp)/'missing_accident_preview.mp4')
            self.assertEqual(backend_api.choose_frontend_artifacts(empty,[missing]),([],[]))
            selected=f'<Artifacts>{json.dumps({"display":[preview.name],"download":[]})}</Artifacts>'
            self.assertEqual(backend_api.choose_frontend_artifacts(selected,[str(preview)]),([],[str(preview)]))

    def test_both_chat_endpoints_return_and_persist_processed_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'clip_accident_preview.mp4';path.write_bytes(b'processed-video')
            result={'messages':[
                ToolMessage(content=json.dumps({'task':'traffic_accident_anticipation','outputs':{'video_path':str(path)}}),tool_call_id='video-test'),
                AIMessage(content='<Conclusion>分析完成</Conclusion>'),
            ]}
            class Handle:
                def invoke(self,*args,**kwargs): return result
                def stream(self,*args,**kwargs): yield {'type':'final','response':result}
                def review_answer_and_artifacts(self,*args,**kwargs):
                    return '<Conclusion>文件说明</Conclusion><Artifacts>{"display":[],"download":[]}</Artifacts>'
            for endpoint in ('/api/chat','/api/chat/stream'):
                session=backend_api.ChatSession('processed-video',Path(tmp),handle=Handle())
                with patch.object(backend_api,'get_session',return_value=session), \
                     patch.object(backend_api.db,'create_session'), \
                     patch.object(backend_api.db,'save_chat_message') as save, \
                     patch.object(backend_api,'maybe_generate_session_title'):
                    client=TestClient(backend_api.app)
                    try:
                        response=client.post(endpoint,data={'session_id':'processed-video','message':'交通检测'})
                        self.assertEqual(response.status_code,200)
                        if endpoint.endswith('/stream'):
                            done=next(b for b in response.text.split('\n\n') if b.startswith('event: done\n'))
                            data=json.loads(done.split('data: ',1)[1])
                        else: data=response.json()
                        self.assertEqual(data['files'][0]['name'],path.name)
                        self.assertEqual(data['images'],[])
                        self.assertEqual(client.get(data['files'][0]['url']).content,b'processed-video')
                        persisted=save.call_args.kwargs['attachments']
                        self.assertEqual(persisted,[{'name':path.name,'path':str(path)}])
                    finally:client.close()


if __name__=='__main__': unittest.main()
