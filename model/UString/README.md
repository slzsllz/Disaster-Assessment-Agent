# UString

按 `pest_yolo` 范式封装的交通事故提前预测工具，支持直接输入行车记录仪视频，
也支持 DAD/VGG16 特征 NPZ。统一入口为 `infer.py:predict()`。

## 目录

```text
UString/
  infer.py                 # 统一接口、CLI、结果文件
  video_features.py        # 视频采样、区域采样、VGG16 提取
  video_preview.py         # 处理后采样片段、候选框、分数与不确定性叠加
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
  --device cuda:0 --model-profile auto
```

直接调用模型：

```python
from model.UString.infer import predict

result = predict(
    {"video": "/path/to/dashcam.mp4"},
    output_dir="model/UString/output",
    device="cuda:0",
    model_profile="auto",
    start_time=0.0,
)
```

`--features /path/to/clip.npz` 可以替代 `--video`，两者互斥。
模型接口对应 `{"features": "/path/to/clip.npz"}`。
CUDA 不可用时回退 CPU。CPU 完整视频推理较慢。

## 视频输入及采样

流程：OpenCV 解码 → KITTI Cascade R-CNN 目标框 → 每帧 19 个区域 →
VGG16 整帧和区域特征 → UString 事故分数与不确定性。

- `model_profile=auto` 根据输入来源匹配模型，不通过分数高低选择权重：
  原始视频复用上游 **CCD Demo** 的 KITTI/VGG16 特征流程，因此默认使用
  `final_model_crash_vgg16.pth`（CCD），10 FPS、最多 50 帧。
  新视频特征缓存写入 `model_profile=ccd`，再次输入 NPZ 时自动保持 CCD。
  旧无标记 NPZ 兼容 DAD/VGG16-new 默认（20 FPS、最多 100 帧）；如果已知它来自
  CCD 视频提取，应显式指定 `model_profile=ccd`，不能仅凭同为 4096 维就认为域相同。
- 可显式选择 `model_profile=ccd|dad|a3d`；指定 `weights_path` 始终优先保留用户权重。
  auto 模式下，已识别的显式权重按内容指纹确定采样默认配置；显式传入的 fps/max_frames 优先于配置默认值。
  `meta.model_selection` 记录选择依据、实际 checkpoint 模型域、是否覆盖权重及域不匹配警告。
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
缓存附带 `timestamps_s`、`fps`、`model_profile` 和 `feature_pipeline`。VGG16 保留原 demo 的 Resize(256)、
CenterCrop(224)、ToTensor 预处理及倒数第二层 4096 维特征，不额外做 Normalize。

## 权重与可靠性限制

四份现有 UString checkpoint 都没有 `rnn.*` 参数：原实现使用普通 list 保存图 GRU，
未注册到 PyTorch，因而未进入标准模型参数保存和优化器参数列表。
新版修正为 ModuleList。当前对这四份随附权重按文件内容 SHA256 识别后，自动按照
上游 `main.py` / `demo.py` 的默认种子 **123** 和模型构造顺序，重建原本未保存、未优化的
循环层；无须传入 `allow_legacy_checkpoint=True`，视频上传可直接运行。
不是只初始化一个孤立 GRU：必须先消耗 `phi_x` 与两个 encoder 的初始化随机数，再构造循环层。
循环初始化固定为 123，用户 `seed` 仍控制采样与特征提取，不会改变该兼容层。
嵌套 RNG 上下文保证不影响调用方随机状态。

兼容结果返回 **`status=legacy_compatible`**，`reliability.level=unvalidated`，图中标注
`legacy compatibility, unvalidated`。`meta.checkpoint_compatibility` 包含模式、权重 SHA256、
种子、循环参数来源；`missing_checkpoint_keys` 保留文件原本缺失的键。
这是在当前 PyTorch 运行时重建上游初始化行为，**不是恢复训练参数，也未验证与旧 PyTorch 1.x
环境或论文精度完全一致**。准确率仍须用真实数据集评估。

未知、被修改或只缺部分循环参数的权重不会自动通过兼容检查；文件同名也不能绕过校验。
对这些文件，默认仍明确报错。只有显式开启 `allow_legacy_checkpoint=True` 才使用用户种子
保留随机循环层并返回 `status=experimental`；仅用于实验联调。
完整结构权重直接加载，返回 `status=success`，不覆盖已有循环参数。
较早的 `dad_vgg16` 文件还使用 `enc_layer1/2` 命名；仅对其已识别的文件映射为
`enc_gcn1/2`，映射记录在 `renamed_keys` 中。

识别的随附发行文件（SHA256）：

| 文件 | SHA256 |
| --- | --- |
| final_model_dad_vgg16-new.pth | `85158182a55add46fe47cf7a4d0bcd7a316f8a6aa6f18b908d720a0e88af4b53` |
| final_model_dad_vgg16.pth | `88f82c3ec9f82945b1fdb5101928e96a517534c2e27b0e0932988fda2756ac41` |
| final_model_a3d_vgg16.pth | `1afbf5b95eb604cc78d7a81f286d2112b9feb7445b2b48ced22c2d0001fb16d8` |
| final_model_crash_vgg16.pth | `d1a5c8954f4012bf036d8efb49718bb04baaa5dc102d2730a35d0c1e7342a776` |

上游依据：[循环层列表与模型构造顺序](https://github.com/Cogito2012/UString/blob/master/src/Models.py)、
[默认种子](https://github.com/Cogito2012/UString/blob/master/main.py)、
[Demo 种子设置](https://github.com/Cogito2012/UString/blob/master/demo.py)。

检测器用当前 PyTorch/torchvision 实现加载原有权重，参数严格匹配；移除了旧 MMDetection、
mmcv 和自定义 CUDA 扩展依赖。适配保留原配置的 RGB 均值/方差、1696×512 保比例缩放、
五种 anchor ratio、三级 cascade、background-first 类别顺序及 inclusive box 坐标。
ROIAlign 坐标参照 [MMDetection v1.1.0 实现](https://github.com/open-mmlab/mmdetection/blob/v1.1.0/mmdet/ops/roi_align/src/roi_align_kernel.cu)。
尚未与原运行环境进行检测输出或准确率基准对比，不能声称完全复现论文精度。

分数不是校准概率，首次越阈值时间不是事故发生时刻或经过验证的提前量。
没有本地评测指标，`reliability.metric` 留空。
`threshold_exceeded=false` 仅表示未超过分数阈值，不是“没有事故”。
`assessment` 明确返回 `accident_presence=undetermined`、`accident_ruled_out=false`，
即使超过阈值也需要画面与标注复核。单个事故样本或合成输入不能证明准确率或预警提前量。

2026-10-23 对用户 `data/000001.mp4` 的诊断与修复见
[视频误判诊断](../../docs/TRAFFIC_000001_DIAGNOSIS.md)。

## 工具与输出

MCP server：`TrafficAccidentAnticipation`，函数：`anticipate_traffic_accident`。
已注册到 `agent/config.json`、`config_deepseek.json`、`config_qwen3.json`。
使用项目现有 FastMCP；输入 `video_path` 或 `features_path`，只能选一个。

共享结果 schema：`model_name/disaster_type/task_type/status/outputs/meta/reliability`。
输出逐帧分数、不确定性、阈值时间、CSV、PNG；视频输入额外输出 NPZ 缓存、首帧框预览及
**`outputs.video_path`：`*_accident_preview.mp4` 处理后预览视频**。
工具额外写入 `summary.json`。首帧框是特征提取候选区域，不是事故定位框。

处理后视频使用 H.264 / yuv420p 和 faststart，聊天助手消息中可直接播放、拖动和下载。
每帧显示源视频时间戳、候选区域、事故分数、阈值、两种不确定性及截至当前帧的分数曲线。
顶部持续标记 `UNVALIDATED` 和兼容/实验模式；候选框明确标为 ROI，不能解释为事故定位框。
没有检测目标时的整帧回退不绘制为目标框；不会根据高分给车辆编造事故类别。
仅输出实际分析的采样片段，按有效采样 FPS 播放，不补充未分析帧；不保留音频。
纯特征 NPZ 输入无法还原原视频，`meta.video_preview.status=not_available`。
缺少 FFmpeg、编码器或源视频解码变化时，保留数值与图表，并通过
`meta.video_preview.status=error` 和 `reason` 明确说明，不伪造视频路径。
后端会保留已生成的交通预览，避免被展示复核的空附件选择过滤；历史助手消息也能恢复视频。
需要 FFmpeg/libx264，Docker 后端已包含该依赖。旧分析记录需重新运行才能产生该视频。

## 验证

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -p test_ustring.py -v
```

2026-10-23 验证：20 项 UString 测试、7 项系统提示词回归及 2 项处理后视频附件测试通过。
使用真实权重，通过 MCP 视频入口在 CPU 上完成两帧合成视频推理，未传实验开关，
返回 `legacy_compatible`；CSV、PNG、NPZ、预览图、H.264 处理后视频和 summary 均生成，
视频/缓存推理分数完全一致。处理后视频帧数与实际采样数一致。
本机冒烟测试约 20 秒，结果位于项目 `tmp/ustring_processed_video/`。
合成输入仅验证运行和接口，不验证真实交通事故预测能力。

## 引用

Wentao Bao, Qi Yu, Yu Kong. Uncertainty-based Traffic Accident Anticipation with
Spatio-Temporal Relational Learning. ACM Multimedia, 2020. 保留上游 MIT LICENSE。
