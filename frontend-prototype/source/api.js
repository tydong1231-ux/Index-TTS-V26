class VoiceBookAPIClient {
  constructor(base = '') { this.base = base; }
  async request(path, options = {}) {
    const response = await fetch(`${this.base}${path}`, {...options, headers: options.body instanceof FormData ? (options.headers || {}) : {'Content-Type':'application/json', ...(options.headers || {})}});
    if (response.status === 204) return null;
    const contentType = response.headers.get('content-type') || '';
    const payload = contentType.includes('application/json') ? await response.json() : await response.text();
    if (!response.ok) { const message = typeof payload === 'object' ? payload.detail || JSON.stringify(payload) : payload; throw new Error(message || `HTTP ${response.status}`); }
    return payload;
  }
  health(){return this.request('/api/health')} config(){return this.request('/api/config')}
  listProjects(){return this.request('/api/projects')} getProject(id){return this.request(`/api/projects/${encodeURIComponent(id)}`)}
  createProject(data){return this.request('/api/projects',{method:'POST',body:JSON.stringify(data)})}
  updateProjectSettings(id,data){return this.request(`/api/projects/${encodeURIComponent(id)}/settings`,{method:'PATCH',body:JSON.stringify(data)})}
  scanProject(id){return this.request(`/api/projects/${encodeURIComponent(id)}/scan`,{method:'POST'})}
  async uploadChapters(id,files){const body=new FormData();[...files].forEach(file=>body.append('files',file));return this.request(`/api/projects/${encodeURIComponent(id)}/chapters/upload`,{method:'POST',body})}
  getChapter(pid,cid){return this.request(`/api/projects/${encodeURIComponent(pid)}/chapters/${encodeURIComponent(cid)}`)}
  updateSegment(pid,cid,sid,data){return this.request(`/api/projects/${encodeURIComponent(pid)}/chapters/${encodeURIComponent(cid)}/segments/${encodeURIComponent(sid)}`,{method:'PATCH',body:JSON.stringify(data)})}
  confirmStage(pid,cid,stage){return this.request(`/api/projects/${encodeURIComponent(pid)}/chapters/${encodeURIComponent(cid)}/confirm/${encodeURIComponent(stage)}`,{method:'POST'})}
  autoMatch(id,overwrite=false){return this.request(`/api/projects/${encodeURIComponent(id)}/roles/auto-match`,{method:'POST',body:JSON.stringify({overwrite_unlocked:overwrite})})}
  updateRole(pid,name,data){return this.request(`/api/projects/${encodeURIComponent(pid)}/roles/${encodeURIComponent(name)}`,{method:'PATCH',body:JSON.stringify(data)})}
  listVoices(){return this.request('/api/voices')}
  createJob(id,data){return this.request(`/api/projects/${encodeURIComponent(id)}/jobs`,{method:'POST',body:JSON.stringify(data)})}
  listJobs(id){return this.request(`/api/projects/${encodeURIComponent(id)}/jobs`)}
  getJob(pid,jid){return this.request(`/api/projects/${encodeURIComponent(pid)}/jobs/${encodeURIComponent(jid)}`)}
  cancelJob(pid,jid){return this.request(`/api/projects/${encodeURIComponent(pid)}/jobs/${encodeURIComponent(jid)}/cancel`,{method:'POST'})}
  listModels(data){return this.request('/api/models',{method:'POST',body:JSON.stringify(data)})}
}
window.VoiceBookAPI = new VoiceBookAPIClient();
