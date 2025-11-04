export class RealTime {
  constructor({ baseUrl=null }={}){
    this.baseUrl = baseUrl;
    this.handlers = new Map();
    this.state = { qpm:0, tfs:0, acc:.9, latency:80 };
    this._init();
  }
  on(evt, cb){
    if(!this.handlers.has(evt)) this.handlers.set(evt, new Set());
    this.handlers.get(evt).add(cb);
  }
  emit(evt, data){ (this.handlers.get(evt)||[]).forEach(cb=>cb(data)); }

  _init(){
    if (this.baseUrl) {
      this._trySSE() || this._tryWS() || this._simulate();
    } else {
      this._simulate();
    }
  }
  _trySSE(){
    try{
      const url = this.baseUrl.replace(/\/$/,'') + '/api/events';
      const es = new EventSource(url);
      es.onmessage = (e)=>{
        const msg = JSON.parse(e.data);
        if(msg.type==='metrics') this._update(msg.payload);
      };
      es.onerror = ()=>{ es.close(); };
      return true;
    }catch{ return false; }
  }
  _tryWS(){
    try{
      const url = this.baseUrl.replace(/^http/,'ws').replace(/\/$/,'') + '/ws';
      const ws = new WebSocket(url);
      ws.onmessage = (e)=>{
        const msg = JSON.parse(e.data);
        if(msg.type==='metrics') this._update(msg.payload);
      };
      ws.onerror = ()=>{ ws.close(); };
      return true;
    }catch{ return false; }
  }
  _simulate(){
    // Simulación suave con ruido
    setInterval(()=>{
      const drift = (v, step, min, max)=>Math.max(min, Math.min(max, v + (Math.random()-.5)*step));
      this.state.qpm = Math.round(drift(this.state.qpm || 120, 14, 40, 240));
      this.state.tfs = Math.round(drift(this.state.tfs || 18, 4, 0, 60));
      this.state.acc = drift(this.state.acc || .92, .02, .75, .99);
      this.state.latency = drift(this.state.latency || 80, 10, 30, 260);
      this._update(this.state);
    }, 1500);
  }
  _update(payload){
    this.emit('metrics', payload);
  }
}
