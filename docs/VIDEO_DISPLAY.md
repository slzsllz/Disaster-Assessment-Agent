# 前端视频展示

选择视频后，输入区显示本地播放器；发送后，视频保留在用户消息中。
历史消息与模型返回的视频文件也可直接播放。支持原生播放/暂停、音量、进度跳转、全屏，
不自动播放；进入规则页面或移除组件时暂停，切换会话时释放本地 Blob URL。
图片和其他文件继续使用原有展示方式。

视频识别使用 `type` / `mime_type`，并兼容 MP4、WebM、MOV、AVI、MKV 等文件扩展名。
能否直接播放取决于浏览器支持的实际编码，不只取决于扩展名。

若浏览器无法解码，可在视频下点击“生成兼容预览”。本地视频需先发送取得服务端文件地址。
后端 `GET /api/files/{file_id}/video-preview` 仅接受已注册的视频文件，使用 FFmpeg 生成
H.264/AAC MP4，最长等待 120 秒，分辨率最高 1280×720，缓存复用且支持 HTTP Range。
兼容预览只用于播放，模型仍读取上传原文件，“下载原视频”也始终指向原文件。
该功能展示原始视频及工具实际产出的视频，不生成事故框或把曲线图转换成检测视频。

运行环境需要 FFmpeg（含 libx264/AAC 编码器）；当前机器已提供，Docker 后端镜像已加入安装。
没有 FFmpeg 时仍可直接播放浏览器支持的视频、下载原文件。
流式聊天新增 `uploaded` 事件，非流式响应新增 `uploaded_files`，用于给当前消息补充服务器地址；
历史附件继续通过数据库恢复。

部署后重启后端并刷新前端；Docker 部署需重新构建镜像。

验证：

```bash
python -m unittest discover -s tests -p test_video_preview.py -v
cd frontend/chatDisaster
node tests/mediaFiles.test.js
npm run build
```
