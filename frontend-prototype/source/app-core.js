const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];

const DEMO_CHAPTERS = [
  {n:'01', title:'雨夜来信', words:'7,812 字', status:'done', output:'18:42', stages:['done','done','done','done','done']},
  {n:'02', title:'没有名字的人', words:'9,104 字', status:'done', output:'21:16', stages:['done','done','done','done','done']},
  {n:'03', title:'旧站台', words:'8,420 字', status:'attention', output:'待生成', stages:['done','attention','pending','pending','pending']},
  {n:'04', title:'被划掉的日期', words:'8,996 字', status:'ready', output:'待生成', stages:['done','done','done','pending','pending']},
  {n:'05', title:'河对岸的灯', words:'10,132 字', status:'running', output:'生成中', stages:['done','done','done','current','pending']},
  {n:'06', title:'周岚的录音', words:'7,566 字', status:'attention', output:'待生成', stages:['done','attention','pending','pending','pending']},
];
const DEMO_CHARACTERS = [
  {name:'旁白', note:'全书旁白 · 1,482 段', voice:'沉稳男声 03', score:'已确认', initial:'旁'},
  {name:'林默', note:'男主角 · 684 段 · 青年', voice:'青年男声 07', score:'已确认', initial:'林'},
  {name:'周岚', note:'女主角 · 511 段 · 冷静克制', voice:'御姐音 02', score:'待确认', initial:'周'},
  {name:'陈伯', note:'配角 · 92 段 · 60–70 岁', voice:'老年男声 01', score:'94% 匹配', initial:'陈'},
  {name:'林默 · 童年', note:'年龄变体 · 18 段', voice:'少年音 04', score:'待确认', initial:'林'},
];
const DEMO_SCRIPT = [
  {id:'0016', speaker:'旁白', text:'雾从河面漫上来时，旧站台只剩下最后一盏灯。', conf:98, emotion:'平静', attention:false},
  {id:'0017', speaker:'旁白', text:'林默站在候车线外，听见铁轨深处传来并不存在的轰鸣。', conf:96, emotion:'低落', attention:false},
  {id:'0018', speaker:'周岚', text:'你还是来了。', conf:72, emotion:'平静', attention:true},
  {id:'0019', speaker:'旁白', text:'周岚从柱子后走出来，手里攥着那封已经被雨水泡皱的信。', conf:89, emotion:'平静', attention:true},
  {id:'0020', speaker:'林默', text:'我只是想知道，当年是谁把名字划掉了。', conf:93, emotion:'低落', attention:false},
];
const stageNames = ['原文','拆分','音色','生成','检查'];
const stageKeys = ['source','script','voices','generate','qa'];
const selected = new Set();
let chapters = [...DEMO_CHAPTERS], characters = [...DEMO_CHARACTERS], scriptRows = [...DEMO_SCRIPT], availableVoices = [];
let activeStage = 'script', currentEpisode = chapters[2] || chapters[0], layer = null;
const backend = {online:false, projectId:null, project:null, currentChapterId:null, activeJobId:null, projects:[]};
function icon(id){return `<svg><use href="#${id}"/></svg>`} function chapterKey(c){return c.backendId||c.n}
function toast(message,error=false){const el=$('#toast');$('span',el).textContent=message;el.classList.toggle('error',error);el.classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>el.classList.remove('show'),2600)}
function formatDuration(seconds){if(!seconds&&seconds!==0)return '';const total=Math.round(seconds),h=Math.floor(total/3600),m=Math.floor((total%3600)/60),s=total%60;return h?`${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`:`${m}:${String(s).padStart(2,'0')}`}
function apiJobSettings(){return {api_base_url:$('#llm-base-url')?.value.trim()||undefined,api_key:$('#llm-api-key')?.value||undefined,model:$('#llm-model')?.value.trim()||undefined,endpoint:'chat_completions',reasoning_effort:'low',generation_mode:$('#generation-mode')?.value||'ordered'}}
function mapStageState(v){return v==='running'?'current':v==='error'||v==='attention'?'attention':v==='done'?'done':'pending'}
function mapChapter(c){const status=c.status==='done'?'done':c.status==='running'?'running':c.status==='attention'?'attention':c.stages?.script==='done'?'ready':'new';return {backendId:c.id,n:String(c.index||0).padStart(2,'0'),title:c.title,words:`${Number(c.source_chars||0).toLocaleString()} 字`,status,output:c.duration_seconds?formatDuration(c.duration_seconds):status==='running'?'生成中':c.mp3_path||c.audio_path?'已生成':'待生成',stages:stageKeys.map(k=>mapStageState(c.stages?.[k])),segmentCount:c.segment_count||0,roleCount:c.role_count||0}}
function mapRole(r){const score=r.voice_locked?'已确认':r.voice_score!=null?`${r.voice_score}% 匹配`:r.voice?'待确认':'待处理';const age=r.age_stage&&r.age_stage!=='adult'?` · ${r.age_stage}`:'';return {backendName:r.name,name:r.name,note:`${r.count||0} 段${age}${r.note?` · ${r.note}`:''}`,voice:r.voice||'尚未选择',score,initial:(r.name||'?').slice(0,1)}}
function mapScript(s){return (s?.segments||[]).map(x=>({id:String(x.id),speaker:x.speaker||'旁白',text:x.text||'',conf:Number(x.confidence??90),emotion:x.emotion||'默认',attention:Boolean(x.needs_review)}))}
async function initializeBackend(){try{const health=await VoiceBookAPI.health();backend.online=true;$('#engine-title').textContent=health.tts.ready?'IndexTTS 已就绪':'IndexTTS 待配置';$('#engine-meta').textContent=health.tts.ready?`${health.tts.fp16?'FP16':'FP32'} · API 已连接`:`缺少 ${health.tts.missing_files.length} 个模型文件`;$('#engine-dot').classList.toggle('offline',!health.tts.ready);const config=await VoiceBookAPI.config();$('#llm-base-url').value=config.api_base_url||$('#llm-base-url').value;$('#llm-model').value=config.model||$('#llm-model').value;const voices=await VoiceBookAPI.listVoices();availableVoices=(voices.voices||[]).map(v=>v.name);const payload=await VoiceBookAPI.listProjects();backend.projects=payload.projects||[];renderProjectLinks();if(backend.projects.length)await loadProject(backend.projects[0].id);else{$('#project-eyebrow').innerHTML='<span class="live-dot"></span>后端已连接 · 请新建项目';renderChapters()}}catch(error){backend.online=false;$('#engine-title').textContent='原型演示模式';$('#engine-meta').textContent='后端未连接';$('#engine-dot').classList.add('offline');toast(`后端未连接，继续使用演示数据：${error.message}`,true);renderChapters()}}
async function loadProject(projectId){const project=await VoiceBookAPI.getProject(projectId);backend.projectId=project.id;backend.project=project;chapters=(project.chapters||[]).map(mapChapter);characters=Object.values(project.roles||{}).map(mapRole);selected.clear();applyProjectToUI(project);renderChapters();renderVoiceDrawer();renderEpisodeVoices();renderProjectLinks();try{const jobs=(await VoiceBookAPI.listJobs(project.id)).jobs||[];const active=jobs.find(j=>['queued','running','cancelling'].includes(j.status));if(active){backend.activeJobId=active.id;pollJob(active.id)}else $('#active-run').style.display='none'}catch(_){$('#active-run').style.display='none'}}
function applyProjectToUI(project){$('#project-title').textContent=project.name;$('#project-eyebrow').innerHTML=`<span class="live-dot"></span>${project.chapters.length} 集 · 最后更新 ${new Date(project.updated_at).toLocaleString()}`;$('#breadcrumb').innerHTML=`<span>项目</span>${icon('i-chevron')}<b>${project.name}</b>`;$('#project-path').textContent=project.source_path||'使用上传文件管理';$('#llm-base-url').value=project.settings?.api_base_url||$('#llm-base-url').value;$('#llm-model').value=project.settings?.model||$('#llm-model').value;$('#generation-mode').value=project.settings?.generation_mode||'ordered';$('#interval-ms').value=project.settings?.interval_ms??450;const complete=chapters.filter(c=>c.status==='done').length,progress=chapters.length?Math.round(complete/chapters.length*100):0;$('.summary-primary strong').textContent=`${progress}%`;$('.summary-primary .summary-progress-row span').textContent=`${complete} / ${chapters.length} 集已完成`;$('.summary-primary .progress-track i').style.width=`${progress}%`;const attention=chapters.filter(c=>c.status==='attention').length,confirmed=characters.filter(c=>c.voice!=='尚未选择').length,summary=$$('.summary-item');if(summary[0]){$('b',summary[0]).textContent=`${confirmed} / ${characters.length} 已分配`;$('em',summary[0]).textContent=`${Math.max(0,characters.length-confirmed)} 个等待处理`}if(summary[1])$('b',summary[1]).textContent=`${attention} 个需要检查`;if(summary[2])$('b',summary[2]).textContent=`${complete} 集已生成`}
