import { RealTime } from './realtime.js';

const yearEl = document.getElementById('year');
if (yearEl) yearEl.textContent = new Date().getFullYear();

const ob = new IntersectionObserver((entries)=>{
  entries.forEach(e => {
    if (e.isIntersecting) e.target.classList.add('visible');
  });
},{threshold:0.15});
document.querySelectorAll('.reveal').forEach(el=>ob.observe(el));

const links = document.querySelectorAll('a[data-link]');
const sections = Array.from(links).map(a=>document.querySelector(a.getAttribute('href'))).filter(Boolean);
const navOb = new IntersectionObserver((entries)=>{
  entries.forEach(e=>{
    if(e.isIntersecting){
      const id = '#'+e.target.id;
      links.forEach(l=>l.toggleAttribute('aria-current', l.getAttribute('href')===id));
    }
  });
},{threshold:0.5});
sections.forEach(s=>navOb.observe(s));

const themeBtn = document.getElementById('themeToggle');
themeBtn?.addEventListener('click', ()=>{
  const html = document.documentElement;
  const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
});
const saved = localStorage.getItem('theme'); if(saved) document.documentElement.setAttribute('data-theme', saved);

// Real-time hookup
const rt = new RealTime({ baseUrl: window.__API_BASE__ });
// KPIs en index
rt.on('metrics', m=>{
  const byId = id=>document.getElementById(id);
  byId('kpi-queries') && (byId('kpi-queries').textContent = m.qpm.toString());
  byId('kpi-tfs') && (byId('kpi-tfs').textContent = m.tfs.toString());
  byId('kpi-acc') && (byId('kpi-acc').textContent = (m.acc*100).toFixed(1)+'%');

  byId('d-kpi-queries') && (byId('d-kpi-queries').textContent = m.qpm.toString());
  byId('d-kpi-tfs') && (byId('d-kpi-tfs').textContent = m.tfs.toString());
  byId('d-kpi-acc') && (byId('d-kpi-acc').textContent = (m.acc*100).toFixed(1)+'%');
  byId('d-kpi-lat') && (byId('d-kpi-lat').textContent = m.latency.toFixed(0));
});

// Sparklines simples
function drawSpark(id, data, color='#7c5cff'){
  const c = document.getElementById(id); if(!c) return;
  const ctx = c.getContext('2d'); const w=c.width, h=c.height;
  ctx.clearRect(0,0,w,h);
  const max = Math.max(...data), min = Math.min(...data);
  const n = data.length; const pad = 6;
  ctx.lineWidth = 2; ctx.strokeStyle = color; ctx.beginPath();
  data.forEach((v,i)=>{
    const x = pad + (i*(w-2*pad))/(n-1||1);
    const y = h - pad - ((v-min)/(max-min||1))*(h-2*pad);
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  });
  ctx.stroke();
}

const charts = {
  qpm: { id:'sp-queries', data:[] },
  tfs: { id:'sp-tfs', data:[] },
  acc: { id:'sp-acc', data:[] },
  lat: { id:'sp-lat', data:[] },
};

rt.on('metrics', m=>{
  charts.qpm.data.push(m.qpm); if(charts.qpm.data.length>40) charts.qpm.data.shift();
  charts.tfs.data.push(m.tfs); if(charts.tfs.data.length>40) charts.tfs.data.shift();
  charts.acc.data.push(m.acc*100); if(charts.acc.data.length>40) charts.acc.data.shift();
  charts.lat.data.push(m.latency); if(charts.lat.data.length>40) charts.lat.data.shift();
  drawSpark(charts.qpm.id, charts.qpm.data, '#7c5cff');
  drawSpark(charts.tfs.id, charts.tfs.data, '#21d4fd');
  drawSpark(charts.acc.id, charts.acc.data, '#19d27c');
  drawSpark(charts.lat.id, charts.lat.data, '#ff8a00');
});
