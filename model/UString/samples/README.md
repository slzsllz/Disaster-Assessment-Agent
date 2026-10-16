输入行车记录仪 MP4/AVI 等 OpenCV 可解码视频，或单段 DAD/VGG16 特征 NPZ。
未附真实事故样本；项目 tmp/ustring_video_smoke/synthetic.avi 仅用于合成冒烟测试。
默认仅处理前 100 个采样帧，可用 start_time 选择片段。
NPZ: data (T,20,4096), det (T,19,4或6), 1<=T<=100。
