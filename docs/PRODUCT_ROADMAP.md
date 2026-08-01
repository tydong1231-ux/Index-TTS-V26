# VoiceBook Studio 产品与技术开发路线图

## 1. 产品目标

把现有 IndexTTS 的单段与多人语音能力，包装成面向长篇小说的本地生产工作台：建立项目目录并放入章节 TXT 后，系统完成章节识别、人物与旁白拆分、全书人物统一、音色匹配、逐段 TTS、章节组装、质量复核和导出。

产品同时提供：

1. **可控模式**：每一阶段都能人工确认、修改、锁定和单独重跑。
2. **自动模式**：选中的多章按依赖关系串行执行，异常时暂停或按策略继续。

核心原则是“自动化流水线 + 可审计中间产物 + 可局部重跑”，而不是不可解释的一键黑箱。

## 2. 对现有仓库的判断

当前 `app/webui.py` 已经包含可复用的后端雏形：

- 音色库和参考音频读取。
- `角色：台词` 多人脚本解析。
- 逐句调用 IndexTTS 并按顺序拼接 WAV。
- 小说项目、章节任务和项目级角色表。
- OpenAI-compatible LLM 结构化拆分、分块记忆、覆盖率检查和遗漏旁白修复。
- 多章串行预处理和合成。
- WAV、可选 MP3 与 manifest 输出。

因此后续不应重写 TTS 核心，而应将单文件 Gradio 工作台拆成稳定的领域层、任务层和独立前端。

主要缺口：

- 稳定的项目、章节、片段和人物 ID。
- 人物别名、人物合并、跨章节一致性和年龄变体。
- 带全局唯一性约束的音色匹配。
- 可暂停、取消、重试和恢复的持久化队列。
- 中间产物版本、依赖失效和局部重跑规则。
- 音频 QA、失败诊断和规范化导出。

## 3. 产品信息架构

### 项目概览

展示总章节、各阶段完成度、待确认人物、低置信度台词、失败片段、最近运行和下一步行动。

### 章节流水线

每章统一使用：

`Imported → Analyzed → Script Confirmed → Voices Ready → Generated → QA Passed → Exported`

每个阶段保存输入版本、输出版本、运行 ID、开始/结束时间和错误。单章可运行、确认、回退、重跑；多章可按依赖串行执行。重跑上游后，下游结果标记为过期，但不立即删除旧产物。

### 角色与音色

项目级人物表是全书一致性的唯一来源。每个人物拥有不可变 `character_id`、主名称、别名、身份线索、年龄段、人物变体、锁定音色、候选音色、匹配解释、出现章节和修改历史。

旁白作为特殊人物管理，可按整书或分卷复用。

### 脚本工作台

每个片段保存 `segment_id`、原文字符区间、章节顺序、人物 ID、文本、情绪、停顿、置信度、确认状态和生成历史。支持低置信度筛选、人物修改、片段拆分/合并、原文覆盖率提示和单句重生。

### 生成队列

默认按原文顺序逐段生成。章节是父任务，片段是子任务；支持失败重试、暂停、恢复、取消和应用重启后的任务恢复。

### 导出与 QA

每章输出音频、manifest、人物音色映射、脚本 JSON、可读 TXT/SRT、模型版本和校验结果。QA 检查空音频、时长异常、削波、静音比例、片段缺失、顺序错误和文件损坏。

## 4. 数据模型

第一版建议 SQLite 保存索引、关系和任务状态，文本和音频保存在项目目录。

```text
Project
  id, name, root_path, settings_version, created_at

Chapter
  id, project_id, order_no, title, source_path, source_hash, status

Character
  id, project_id, canonical_name, parent_character_id, variant_label,
  traits_json, voice_id, voice_locked, confirmed

CharacterAlias
  id, character_id, alias, source_chapter_id, confidence

Voice
  id, name, audio_path, metadata_json, embedding_path, enabled

Segment
  id, chapter_id, order_no, source_start, source_end,
  character_id, text, emotion, pause_ms, confidence, confirmed, version

Artifact
  id, entity_type, entity_id, stage, version, path, content_hash, run_id

Run
  id, project_id, type, status, config_json, progress, error,
  started_at, finished_at

RunItem
  id, run_id, chapter_id, segment_id, status, attempt, output_path, error
```

不要用文件名作为业务主键。

## 5. LLM 预处理

后端提供统一 `LLMProvider`：

```text
list_models()
analyze_chapter(input, schema, context)
repair_analysis(input, previous_output, validation_errors)
```

支持 OpenAI 官方 API、OpenAI-compatible `base_url`、自定义模型 ID，以及从 `/v1/models` 动态读取模型。API Key 不写入项目 JSON 或浏览器 localStorage。

推荐两阶段分析：

1. 识别人名、别名、年龄阶段、身份和说话关系。
2. 基于项目人物表输出严格顺序的 Segment。

随后执行确定性校验：原文覆盖、字符区间、顺序、重复、遗漏、未知人物、别名冲突和片段长度。仅对失败块进行修复调用。

LLM 可以提议新人物、合并或年龄变体，但不能静默修改已锁定人物。

## 6. AI 音色匹配

音色文件应带可解析名称或结构化元数据，例如：

```text
少年音1__male__teen__clear__energetic.wav
御姐音2__female__adult__cool__authoritative.wav
老年男1__male__elder__warm__slow.wav
```

匹配因素包括性别表达、年龄、性格、音高、质感、能量、语速和角色重要性。

```text
score =
  0.28 * gender_match +
  0.22 * age_match +
  0.18 * temperament_match +
  0.12 * pitch_match +
  0.10 * pace_match +
  0.10 * role_importance_quality
  - reuse_penalty
  - conflict_penalty
```

AI 一键匹配必须进行全局分配，而不是逐个人物贪心选择：

- 已锁定音色先加入约束。
- 主角和高频人物优先获得区分度高的音色。
- 默认尽量不复用同一音色。
- 音色不足时才允许次要人物复用，并避免同场人物声音相近。
- 基础人物与童年/老年变体应“相似但可辨认”。

第一版可使用加权打分与匈牙利算法完成一对一分配，再进入带惩罚的复用轮次。UI 展示推荐理由、匹配度和冲突。

## 7. TTS 生成策略

### 默认模式

每个 Segment 独立生成并保存文本 hash、人物、音色、参考音频 hash、参数、模型版本、输出文件和重试次数。该模式最容易缓存、恢复、局部替换和可靠回装。

### 优先性能优化

在实现长音频回切之前，先做：

- 缓存每个音色的 speaker conditioning/embedding。
- 同一人物片段连续调度，但仍逐 Segment 输出。
- 对相邻且情绪一致的短片段做有限合并并保留边界。
- 相同输入 hash 直接复用缓存。

### Plan B：按角色批量

第一版应定义为**按角色分组调度与共享 conditioning**，而不是把全书某人物台词拼成长音频再盲切。只有 TTS 原生批量返回独立音频或提供可靠声学边界时，才启用真正的批量推理与回装。

### 章节组装

组装器按 manifest 的 `order_no` 工作，不依赖文件名排序；统一采样率、声道和响度，应用句后与场景停顿，并把每个片段的起止时间写回 manifest。

## 8. 推荐系统架构

```text
Frontend: React + TypeScript + Vite
API: FastAPI
Persistence: SQLite + SQLAlchemy/SQLModel
Job Engine: persistent local queue + worker process
TTS Adapter: current IndexTTS2 wrapper
Audio: ffmpeg + soundfile/wave
Events: WebSocket or Server-Sent Events
```

本地 Windows 第一版可由浏览器访问本机 FastAPI，稳定后再使用 Tauri 包装。TTS 模型进程保持常驻，API 请求不重复加载模型。

服务边界：

```text
Frontend
  Project / Chapter / Character / Script / Queue / Export

API Service
  ProjectService / AnalysisService / CharacterService
  VoiceService / RunService / ExportService

Worker
  LLM jobs / TTS jobs / Assembly jobs / QA jobs

Adapters
  OpenAI-compatible LLM / IndexTTS2 / ffmpeg / filesystem
```

所有耗时接口返回 `run_id`，通过 SSE/WebSocket 汇报状态。

## 9. 开发阶段

### Phase 0 — 交互确认

当前静态原型。确认六个主页面、章节阶段、项目级人物表、脚本审核方式，以及默认模式与 Plan B 的产品表述。

### Phase 1 — 项目骨架与持久化

- React/TypeScript 工程。
- FastAPI、SQLite、项目目录扫描。
- 项目、章节、人物、片段和运行模型。
- TXT 导入、自然排序、source hash 和文件变更检测。
- SSE/WebSocket 状态通道。

验收：应用重启后项目、状态和运行历史完整恢复。

### Phase 2 — LLM 分析与脚本工作台

- Provider 配置与动态模型列表。
- 分块、结构化输出、校验和局部修复。
- 人物别名提议与跨章人物注册表。
- 逐句编辑、低置信度队列和版本保存。

验收：每个 Segment 都能定位回原文；覆盖异常不能静默通过。

### Phase 3 — 角色与音色

- 音色导入、命名解析、试听和元数据编辑。
- 人物锁定、合并、别名和年龄变体。
- 全局 AI 匹配与冲突解释。

验收：同一 `character_id` 全书默认使用同一 `voice_id`；变体关系明确。

### Phase 4 — 持久化生成队列

- IndexTTS adapter 与常驻 worker。
- Segment 级任务、缓存、重试、暂停、恢复和取消。
- 分章串行批处理、日志和错误分类。

验收：中途退出后可恢复；修改一句只重生该句并重新组装章节。

### Phase 5 — 组装、QA 与导出

- manifest 稳定组装。
- 响度、静音、损坏和时长异常检测。
- 章节播放器与文本时间定位。
- MP3/M4B/SRT/工程包导出。

### Phase 6 — 性能与高级自动化

- speaker conditioning 缓存。
- 按角色分组调度。
- 有条件的短句合并。
- 自动 QA 后只把异常送人工。
- 多 GPU worker 和优先级。

## 10. 第一版明确不做

- 不做无法暂停的一键黑箱。
- 不用文件名关联人物或片段。
- 不在浏览器保存明文 API Key。
- 不把全角色文本合成长音频后盲切。
- 不让 LLM 覆盖人工锁定人物或音色。
- 不同时支持大量 TTS 引擎和复杂云部署。

## 11. 给 AI 编程工具的落地顺序

每个 PR 完成一个可验证的垂直切片：

1. 建立 `frontend/`、`server/`、`worker/` 和共享 schema。
2. 项目创建、TXT 导入、章节列表和 SQLite 持久化。
3. mock run + SSE 真实进度。
4. LLM 分析一个章节并保存 versioned script。
5. 人物注册表和手动音色绑定。
6. 单个 Segment 的 IndexTTS 生成。
7. 章节队列与 manifest 组装。
8. 多章批量、AI 自动匹配和高级优化。

每一步保留 mock adapter，使前端开发不依赖 GPU 和模型权重。