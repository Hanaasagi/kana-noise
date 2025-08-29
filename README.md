## kana-noise

*这是一个使用预训练模型的 Demo*

---

用于在 VTuber 录播视频中自动定位“尖叫/惊叫”与多类“副语言（Paralinguistic）事件”，如: 咂嘴（口腔点击）、叹气、哼声、低语、清嗓、咳嗽、喘气、呻吟、笑声等，并导出包含时间段与批量裁剪脚本的结果，支持长视频、可复用缓存与并发处理。

核心技术与组件: 
- 人声分离: Demucs（MDX 系列，支持 `cpu/cuda/mps`）
- 声活性: WebRTC VAD
- 启发式评分: RMS/谱质心/谱滚降/谱平坦度/ZCR 的 z-score 加权
- 事件分类（可选）: PANNs（AudioSet CNN14 多标签分类）。目前 PANNs 仅支持 `cpu/cuda`
- 字幕: 使用 whisper.cpp 生成字幕
- 工程化: 按步骤的“签名缓存”、结构化产物目录、并发与交互选择器

代码结构（主要）: 
- `scripts/extract.py`: 命令行入口
- `src/kana/pipeline/`: 流水线步骤实现（extract/demucs/vad/score/intersect/panns/paralinguistic/export）
- `runs/`: 每个视频的输出目录（见下）
- `models/panns/`: PANNs 模型（`Cnn14_mAP=0.431.pth`）
- `models/whisper/`: whisper.cpp 模型。可以使用 `misc/download-ggml-model.sh` 进行下载

## 实现细节

整体流程: 
1) 抽音（extract）
- 从视频提取 44.1k 立体声（供 Demucs）与 16k 单声道（检测回退）。
- 支持 `--preview-minutes N` 仅处理前 N 分钟用于快速预览。

2) 人声分离（demucs）
- 使用 `--demucs-model`（如 `mdx_q`）与 `--demucs-two-stems vocals` 提速，仅分离人声。
- Device: `--demucs-device cpu|cuda|mps`（Apple Silicon 推荐 `mps`）。
- 产物: `demucs/<model>/<basename>/{vocals.wav,no_vocals.wav}`；将 `vocals.wav` 下混为 `vocals_16k.wav` 供检测。

3) 声活性（vad）
- 在 `vocals_16k.wav`（若无则 `extracted_16k.wav`）上运行 WebRTC VAD 定位“有人声”窗口。
- 主要参数: `--vad-aggr`（0–3）`--vad-min-ms` `--vad-merge-ms`。

4) 尖叫候选（score）
- 计算帧级特征: RMS（dB）、谱质心、90% 谱滚降、谱平坦度、ZCR，并做 z-score 归一与加权+平滑；根据 `--score-thr/--min-sec/--merge-gap` 产出尖叫候选事件。

5) 与 VAD 相交（intersect）
- 将候选与 VAD 人声窗相交，并按 `--pad-sec` 对边界加余量后合并相邻事件。

6) 尖叫复核（panns，可选）
- `--use-panns` 启用后，在事件窗上用 PANNs 的 scream/yell/shout 置信度二次确认，并以 `Singing/Music` 做抑制。事件分数为 `max(启发式, PANNs 置信度)`。

7) 副语言事件（paralinguistic，可选）
- 规则通道: 
  - click（口腔瞬态/咂嘴）: 短窗瞬态+高平坦度+高质心、极短时长。
  - murmur（嘟囔）: 低能量有声、低谐波稳定性、短中时长。
  - sigh（叹气）: 较长、平坦度偏高、质心偏低。
- PANNs 通道（`--quirks-panns`）: 
  - 在 VAD 窗内以 1.5s 窗/0.75s 步长滑动，若片段短则补零到 1.5s；对以下标签取最大置信度，阈值 `--quirks-panns-thr`: 
  - `hum/whisper/breathing/throat_clearing/cough/gasp/groan/sigh/laugh`
  - 与规则通道结果按类别合并。

8) 导出（export）
- 临时策略: 导出层强制最小时长 `0.5s`，降低噪声片段（位于导出步骤注释说明）。
- 生成: 
  - 汇总 CSV: `export/events.csv`，列: `type,type_zh,start,end,start_text,end_text,score`（`start_text/end_text` 为 `mm:ss`）。
  - 按类别的脚本: `export/cut_<type>.sh`
  - 分类剪辑: `export/clips/<type>/mm-ss.mp4`

分数 score 说明: 
- 用于同类事件的排序与筛选，不建议跨类比较。
- 尖叫: 未启用 PANNs 时为启发式分数；启用后为 `max(启发式, PANNs)`。
- 副语言 PANNs 类: PANNs 置信度（0–1）。规则类为无量纲启发式分数。

## runs 目录说明

`runs/<video_stem>/`: 
- `audio/`: `extracted_44k.wav`、`extracted_16k.wav`、`demucs/<model>/<basename>/{vocals.wav,no_vocals.wav}`、`vocals_16k.wav`
- `vad/segments.json`: VAD 人声窗列表
- `score/{times.npy,scores.npy,events.json}`: 尖叫帧时间/分数与候选
- `events/events_vad.json`: 与 VAD 相交后的尖叫事件
- `panns/refined.json`: 尖叫经 PANNs 复核的结果
- `paralinguistic/refined.json`: 副语言事件，`{"categories": {cat: [(s,e,score), ...], ...}}`
- `export/events.csv`、`export/cut_<type>.sh`、`export/clips/<type>/*.mp4`
- `meta/*.json`: 各步骤签名（signature），用于缓存
- `subs/vocals.srt`: whisper.cpp 生成字幕
- `logs/demucs.log`: Demucs 详细输出
- `logs/whisper.log`: whisper.cpp 日志

## 使用方法

因为一些原因，本项目向下 Pin 在 Python 3.11 版本

依赖安装: 
```bash
uv sync
```

处理全部视频（示例）: 
```bash
uv run python scripts/extract.py \
  -d videos -r runs --all --reuse --log-level INFO \
  --separate-vocals --demucs-model mdx_q --demucs-device mps --demucs-two-stems vocals \
  --vad-aggr 2 --vad-min-ms 150 --vad-merge-ms 200 \
  --score-thr 0.40 --min-sec 0.20 --merge-gap 0.30 --pad-sec 0.20 \
  --use-panns --panns-thr 0.25 \
  --extract-quirks --quirks-panns --quirks-panns-thr 0.30 \
  --gen-subs --subs-language zh
```

预览模式（前 10 分钟）用于调试: 
```bash
uv run python scripts/extract.py -d videos -r runs --preview-minutes 10 --reuse \
  --separate-vocals --demucs-model mdx_q --demucs-device mps --demucs-two-stems vocals \
  --vad-aggr 2 --vad-min-ms 150 --vad-merge-ms 200 \
  --score-thr 0.40 --min-sec 0.20 --merge-gap 0.30 --pad-sec 0.20 \
  --gen-subs --subs-language zh
```

通过 Makefile: 
- `make run` 与 `make run-preview` 已内置常用参数，可直接使用。

## 工程缓存

- 每步运行后写入 `meta/<step>.json`，包含: 关键输入（存在/大小/mtime）、参数字典与状态。签名一致且产物存在时跳过计算。
- 变更传播: 上游产物/参数变化会使下游失效，在下一次运行重算。
- Demucs Device 独立: 签名不包含 `device`（cpu/cuda/mps），保证不同 Device 重复运行不会触发不必要重算；`model/two-stems/jobs` 等仍入签名以保证一致性。

- 交互选择器: 未提供 `--all` 与 `--file` 时，会列出 `videos/` 下的 `.mp4` 供交互选择。
- 配置文件: `--config config.json` 支持将 JSON 作为默认参数加载，CLI 显式参数覆盖之。

## 现阶段问题

1. demucs 分离人声耗时很长，mba m4 下 2 小时时长视频需要提取 30 分钟以上
2. PANNs 不支持 MPS，需要使用 CPU
3. 对于 `click/hum/murmur/` 分类识别并不准确

## 感谢下面的项目

- https://github.com/facebookresearch/demucs
- https://github.com/adefossez/demucs
- https://github.com/qiuqiangkong/panns_inference
- https://github.com/ggml-org/whisper.cpp
