# 000001.mp4 交通事故漏报诊断

## 已核对的输入与分析范围

- 用户指定文件：`data/000001.mp4`，SHA256 为 `877b6804d709acd31a8e7c4843590d674d3877fffab3d8c076332462787d2179`。
- MPEG-4 编码，1280×720，10 FPS，50 帧，5 秒。
- 按文件 SHA256 找到的历史运行结果均覆盖 50 帧，时间戳 0.0–4.9 秒。
  因此本次并非默认窗口漏掉了后半段，也不是视频解码失败。
- 逐段抽帧可见约 3–4 秒黑色车辆横向进入本车路径并近距离接触/遮挡的过程。
  原图抽帧在 `tmp/traffic_000001_diagnosis/contact_sheet.jpg`，不根据模型分数否定这些画面。

## 已确认的问题

原始视频入口实现了上游 CCD Demo 的特征流程：KITTI Cascade R-CNN 候选区域、
未额外 Normalize 的 VGG16 fc7 线性输出、每帧 19 个区域；但封装默认加载了
`final_model_dad_vgg16-new.pth`。同为 4096 维并不意味着这些特征与 DAD 训练输入兼容。

[上游 README](https://github.com/Cogito2012/UString#-train-and-test) 明确说明端到端 Demo
使用 CCD 模型；[demo.py](https://github.com/Cogito2012/UString/blob/master/demo.py) 提供该视频流程，
[DataLoader.py](https://github.com/Cogito2012/UString/blob/master/src/DataLoader.py) 中 CrashDataset
对应 CCD，使用 50 帧、10 FPS。该匹配依据来自实现和数据来源，不是为此样本挑最高分。

同时，`threshold_exceeded=false` 原本仅表示模型分数未达到 0.5，不能被回答成
“没有事故”或“是否发生事故：否”。旧接口未单独表达这个区别，容易造成结果误读。

## 对照证据

固定同一份 50 帧特征、种子 0、10 次 Bayesian 采样、阈值 0.5，仅替换已有 checkpoint，CPU 复算：

| 权重 | 最高分 | 最低分 | 是否越过 0.5 |
| --- | ---: | ---: | --- |
| DAD/VGG16-new（旧默认） | 1.3373e-14 | 7.7235e-24 | 否 |
| DAD/VGG16 | 0 | 0 | 否 |
| A3D/VGG16 | 1.8063e-5 | 8.7372e-10 | 否 |
| CCD/crash/VGG16（匹配视频流程） | 1.0 | 0.83591 | 是 |

原 GPU 历史结果的 DAD 最高分约 1.337319e-14，CPU 约 1.337327e-14，
差异仅是数值精度，不能解释模型为何输出近零。图边权也与上游公式核对过，最大差约 2.98e-8。
VGG 截取层与未 Normalize 处理均与上游 Demo 一致，不能简单改归一化来让分数变高。

诊断控制还包括：重复第一帧特征 50 次，CCD 仍可达到 1.0；无事故语义的两帧合成输入，
CCD 最高分约 0.00210。这些是人工控制输入，不是带标注的真实正常视频评测集。
前者尤其说明，高分或本例首次越阈值不能证明模型正确定位了事故时刻或具有可靠预警提前量。

完整对照记录：`tmp/traffic_000001_diagnosis/checkpoint_comparison.json` 和 `controls.json`。

## 实施修复与复跑

1. 新增 `model_profile=auto|ccd|dad|a3d`。auto 的原始视频使用 CCD、10 FPS、50 帧默认；
   新缓存保存 CCD 标记。无标记旧 NPZ 保持旧 DAD 兼容行为，可根据已知来源显式指定模型域。
2. 保留用户显式权重、FPS 和帧数设置；记录实际选择理由及模型/特征域不匹配提示。
   没有修改阈值、反转分类或通过分数挑选权重。
3. 新增 `assessment`：阈值只描述模型信号，`accident_presence=undetermined`、
   `accident_ruled_out=false`；工具说明及系统 Markdown 提示词同步限制“无事故”结论。

真实 MCP 视频入口重新完整处理 `data/000001.mp4`，自动选择 CCD，GPU 执行约 34 秒：

- 共 50 帧，10 FPS，覆盖 0.0–4.9 秒，原视频 SHA256 不变。
- 最低分约 0.835936，最高分 1.0，首次越阈值为 0.0 秒。
- 新生成的特征缓存再次通过 MCP 输入，自动沿用 CCD，分数逐项完全一致。
- 生成 H.264 预览视频，保留 `legacy_compatible` / `unvalidated` 标记。

结果位于 `tmp/traffic_000001_diagnosis/fixed_run/video/`，其中：
`000001_accident_preview.mp4`、`000001_accident_curve.csv`、`summary.json`。
`fixed_result.json` 保存了此次完整结果。

当前修复解决了默认输入域不匹配和结论误读，不能据这一个事故样本宣称已恢复论文精度。
循环参数仍为上游种子初始化的兼容恢复，检测器也未经原运行时准确率对照。
可靠性评估还需要有标注的事故与正常视频集，报告漏报、误报和提前量；不得通过调低阈值
或只报告有利样本代替评测。

重启后端并使用新对话重新分析，才能加载更新后的工具描述和系统提示词。
