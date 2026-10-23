import test from 'node:test'
import assert from 'node:assert/strict'
import { isVideoFile, videoPreviewUrl } from '../src/utils/mediaFiles.js'

test('upload and history videos are recognized using MIME and filename', () => {
  for (const file of [{name:'dashcam.MP4'}, {name:'clip.avi',type:'application/octet-stream'},
    {name:'clip',mime_type:'video/webm'}, {name:'clip',type:'video/mp4; codecs=avc1'}, {name:'clip.mov'}]) {
    assert.equal(isVideoFile(file),true,JSON.stringify(file))
  }
  for (const file of [{name:'curve.csv'}, {name:'preview.jpg',type:'image/jpeg'}, {name:'summary.json'}, {}]) {
    assert.equal(isVideoFile(file),false)
  }
})
test('compatible previews only use registered backend file URLs', () => {
  assert.equal(videoPreviewUrl('/api/files/abc-123'),'/api/files/abc-123/video-preview')
  for (const url of ['blob:test','https://example.com/video.mp4','/api/files/../secret','/tmp/video.avi','']) {
    assert.equal(videoPreviewUrl(url),'')
  }
})
