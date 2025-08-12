## 工程化批处理（推荐）

新的批处理与缓存管线会将每个视频产物写入 `runs/<video_stem>/`，各步骤带签名缓存（参数+输入信息），可复用、断点续跑。

目录结构（每视频）：
- `runs/<video>/audio/`: `extracted_44k.wav`, `extracted_16k.wav`, `demucs/<model>/<basename>/vocals.wav`, 以及 `vocals_16k.wav`
- `runs/<video>/vad/segments.json`
- `runs/<video>/score/{times.npy,scores.npy,events.json}`
- `runs/<video>/events/events_vad.json`
- `runs/<video>/panns/refined.json`
- `runs/<video>/export/{screams.csv,cut_screams.sh,clips/}`
- `runs/<video>/meta/*.json`（每步签名） 和 `runs/<video>/logs/`

### 批处理命令

对 `videos/` 下所有 `.mp4` 并发处理（带缓存复用）：
```bash
uv run python scripts/extract.py \
  -d videos -r runs --reuse --num-workers 2 \
  --preview-minutes 10 --log-level INFO \
  --separate-vocals --demucs-model mdx_q --demucs-device cpu --demucs-two-stems vocals --demucs-jobs 0 \
  --vad-aggr 2 --vad-min-ms 150 --vad-merge-ms 200 \
  --score-thr 0.40 --min-sec 0.20 --merge-gap 0.30 --pad-sec 0.20 \
  --use-panns --panns-thr 0.25 \
  --extract-quirks --quirks-panns --quirks-panns-thr 0.30
```

说明：
- `--reuse`：复用已完成步骤（签名一致则跳过）
- `--num-workers`：并行处理视频数
- `--preview-minutes`：只处理前 N 分钟（产物签名包含该参数，和全量不互相复用）
- Demucs 仍会生成 `no_vocals.wav`，本项目不删除，默认仅使用 `vocals.wav` 下混的 `vocals_16k.wav` 作为检测输入

后续可在 `runs/<video>/export/` 下查看 `screams.csv` 与 `cut_screams.sh`。

### 组件说明
- 抽音（extract）：输出 `extracted_44k.wav` 与 `extracted_16k.wav`
- 人声分离（demucs）：输出 `demucs/<model>/<basename>/vocals.wav`，并生成 `vocals_16k.wav`（默认也会有 `no_vocals.wav`，不删除）
- VAD（vad）：输出声活性片段 `vad/segments.json`
- 启发式评分（score）：输出 `score/times.npy`、`score/scores.npy`、`score/events.json`
- 事件相交（intersect）：`events/events_vad.json`
 - PANNs 细化（panns，可禁用）：`panns/refined.json`
 - 副语言事件（paralinguistic，可选）：`paralinguistic/refined.json`（类别：sigh/hum/whisper/breathing/throat_clearing/cough/gasp/groan + click/murmur 规则通道）
 - 导出（export）：
   - 汇总 CSV：`export/events.csv`（列：type,start,end,score）
   - 分类脚本：`export/cut_<type>.sh`
   - 分类剪辑：`export/clips/<type>/*.mp4`

### 小贴士
- 首次使用 Demucs 与 PANNs 可能下载权重，macOS 如提示缺少 `wget`：`brew install wget`。
- 若 Demucs 速度仍慢：优先 `--demucs-two-stems vocals`，选用 `--demucs-model mdx_q`，如有 GPU 加 `--demucs-device cuda`。
- 若 PANNs 误杀较多，降低 `--panns-thr`；若误报歌声，保持开启 `--use-panns` 并提高 `--panns-thr`。

## 规格文档

详见 `SPEC.md`，包含需求、架构、实现细节与扩展点。



