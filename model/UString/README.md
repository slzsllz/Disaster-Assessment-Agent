# UString

按 `pest_yolo` 范式封装的交通事故提前预测工具，支持直接输入行车记录仪视频，
也支持 DAD/VGG16 特征 NPZ。统一入口为 `infer.py:predict()`。

## 目录

```text
UString/
  infer.py                 # 统一接口、CLI、结果文件
  video_features.py        # 视频采样、区域采样、VGG16 提取
  model/                   # 推理网络、Cascade R-CNN torchvision 适配
  weights/
    cascade_kitti.pth       # 已下载的 KITTI Cascade R-CNN 原始权重
    vgg16-397923af.pth      # 从本机 torch 缓存复制，无需联网下载
    ustring/               # 保留四份已下载的事故预测权重
  samples/                 # 输入说明
  output/                  # CLI 默认输出
  requirements.txt
  LICENSE
```

本次导入的原始训练、评测、旧演示、下载说明和 Git 元数据，已移至项目
`tmp/ustring_backup/reimport/`；源码压缩备份为其中的 `source.tar.gz`。
旧 output 下重复的权重也移入备份。现有模型文件内容未修改。

## 安装与使用

```bash
pip install -r model/UString/requirements.txt
python agent/tools/TrafficAccidentAnticipation.py \
  --video /path/to/dashcam.mp4 --temp_dir tmp/ustring_run \
  --device cuda:0 --fps 20 --max-frames 100 --allow-legacy-checkpoint
```

直接调用模型：

```python
from model.UString.infer import predict

result = predict(
    {"video": "/path/to/dashcam.mp4"},
    output_dir="model/UString/output",
    device="cuda:0",
    fps=20.0,
    start_time=0.0,
    max_frames=100,
    allow_legacy_checkpoint=True,
)
```

`--features /path/to/clip.npz` 可以替代 `--video`，两者互斥。
模型接口对应 `{"features": "/path/to/clip.npz"}`。
CUDA 不可用时回退 CPU。CPU 完整视频推理较慢。

## 视频输入及采样

流程：OpenCV 解码 → KITTI Cascade R-CNN 目标框 → 每帧 19 个区域 →
VGG16 整帧和区域特征 → UString 事故分数与不确定性。

- 默认 DAD/VGG16-new 权重，20 FPS、最多 100 帧。
- `start_time` 是原视频秒数偏移，`max_frames` 范围 1–100。
- **每次只分析选定片段，不能把结果解释为完整长视频的结论**。
  长视频可分段调用，片段之间重置循环状态。
- 源视频 FPS 低于请求 FPS 时使用源 FPS，不重复帧、不虚构更高采样率。
- 短视频仅使用实际解码帧，不补帧；空视频、越界起点明确报错。
- 时间戳按照源视频帧索引和 FPS 元数据计算，适用于固定帧率视频；可变帧率视频应先转为固定帧率。
- `feature_batch_size` 默认 4，内存不足可设为 1。
- 可通过 `detector_weights` 和 `vgg_weights` 覆盖本地模型路径。

原始 NPZ 要求 `data=(T,20,4096)`、`det=(T,19,4或6)`，`1<=T<=100`。
第 0 个特征为整帧，其余 19 个与框一一对应；框前四列是像素 xyxy。
缓存附带 `timestamps_s` 和 `fps`。VGG16 保留原 demo 的 Resize(256)、
CenterCrop(224)、ToTensor 预处理及倒数第二层 4096 维特征，不额外做 Normalize。

## 权重与可靠性限制

四份现有 UString checkpoint 都没有 `rnn.*` 参数：原实现使用普通 list 保存图 GRU，
未注册到 PyTorch，因而未进入标准模型参数保存和优化器参数列表。
新版修正为 ModuleList，但不能由现有权重恢复缺失参数。

默认 `allow_legacy_checkpoint=False`，缺失参数明确报错。
显式开启实验兼容模式后使用固定种子随机初始化循环参数，返回 `status=experimental`，
列出缺失参数，结果图也标注实验状态。**该模式可用于工具链联调，不能当成可靠事故预测。**
正式使用需要与结构匹配的完整 checkpoint，并经过数据集验证。

检测器用当前 PyTorch/torchvision 实现加载原有权重，参数严格匹配；移除了旧 MMDetection、
mmcv 和自定义 CUDA 扩展依赖。适配保留原配置的 RGB 均值/方差、1696×512 保比例缩放、
五种 anchor ratio、三级 cascade、background-first 类别顺序及 inclusive box 坐标。
ROIAlign 坐标参照 [MMDetection v1.1.0 实现](https://github.com/open-mmlab/mmdetection/blob/v1.1.0/mmdet/ops/roi_align/src/roi_align_kernel.cu)。
尚未与原运行环境进行检测输出或准确率基准对比，不能声称完全复现论文精度。

分数不是校准概率，首次越阈值时间不是事故发生时刻或经过验证的提前量。
没有本地评测指标，`reliability.metric` 留空。

## 工具与输出

MCP server：`TrafficAccidentAnticipation`，函数：`anticipate_traffic_accident`。
已注册到 `agent/config.json`、`config_deepseek.json`、`config_qwen3.json`。
使用项目现有 FastMCP；输入 `video_path` 或 `features_path`，只能选一个。

共享结果 schema：`model_name/disaster_type/task_type/status/outputs/meta/reliability`。
输出逐帧分数、不确定性、阈值时间、CSV、PNG；视频输入额外输出 NPZ 缓存和首帧框预览。
工具额外写入 `summary.json`。首帧框是特征提取候选区域，不是事故定位框。

## 验证

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -p test_ustring.py -v
```

已用真实权重完成 CPU 上两帧合成视频的完整工具链测试；合成输入仅验证运行和接口，
不验证真实交通事故预测能力。测试结果在项目 `tmp/ustring_video_smoke/`。

## 引用

Wentao Bao, Qi Yu, Yu Kong. Uncertainty-based Traffic Accident Anticipation with
Spatio-Temporal Relational Learning. ACM Multimedia, 2020. 保留上游 MIT LICENSE。
