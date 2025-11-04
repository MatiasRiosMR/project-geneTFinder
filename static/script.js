document.addEventListener('DOMContentLoaded', function(){
  const hero = document.getElementById('hero');
  const appCard = document.getElementById('appCard');
  const loader = document.getElementById('loader');
  const seqForm = document.getElementById('seqForm');
  const clearBtn = document.getElementById('clearBtn');
  const predictBtn = document.getElementById('predictBtn');
  const resultPanel = document.getElementById('resultPanel');
  const resultBadge = document.getElementById('resultBadge');
  const resultText = document.getElementById('resultText');
  const resultIcon = document.getElementById('resultIcon');
  const barFill = document.getElementById('barFill');
  const percentText = document.getElementById('percentText');
  const newPredBtn = document.getElementById('newPredBtn');

  // Clear textarea
  if(clearBtn){
    clearBtn.addEventListener('click', () => {
      const ta = document.getElementById('sequence');
      if(ta) ta.value = '';
      ta && ta.focus();
    });
  }

  // AJAX submit to /api/predict
  if(seqForm){
    seqForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const ta = document.getElementById('sequence');
      const seq = ta && ta.value.trim();
      if(!seq){
        ta && (ta.style.animation = 'shake .35s');
        setTimeout(()=>{ if(ta) ta.style.animation=''; }, 400);
        return;
      }
      // show loader
      loader.classList.remove('hidden');
      loader.setAttribute('aria-hidden','false');
      predictBtn.disabled = true;
      resultPanel.classList.add('hidden');

      try{
        const resp = await fetch('/api/predict', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({sequence: seq})
        });
        const data = await resp.json();
        if(!resp.ok){
          alert(data.error || 'Error en la predicción');
        } else {
          const isTF = data.class === 1;
          const conf = Math.max(0, Math.min(1, Number(data.confidence)));
          resultText.textContent = isTF ? 'Factor de Transcripción (TF)' : 'No-TF';
          resultIcon.textContent = isTF ? '✔' : '✖';
          resultBadge.className = 'badge ' + (isTF ? 'tf' : 'not-tf');
          barFill.style.width = (conf*100) + '%';
          percentText.textContent = (conf*100).toFixed(2) + '%';
          resultPanel.classList.remove('hidden');
          resultPanel.style.animation = 'fadeUp .45s ease both';
          // actualizar sidebar inmediatamente
          fetchStats();
        }
      } catch(err){
        console.error(err);
        alert('Error al conectar con el servidor');
      } finally {
        loader.classList.add('hidden');
        loader.setAttribute('aria-hidden','true');
        predictBtn.disabled = false;
      }
    });
  }

  if(newPredBtn){
    newPredBtn.addEventListener('click', () => {
      resultPanel.classList.add('hidden');
      const ta = document.getElementById('sequence');
      ta && ta.focus();
    });
  }

  // Stats polling
  async function fetchStats(){
    try{
      const resp = await fetch('/api/stats');
      if(resp.status === 200){
        const data = await resp.json();
        document.getElementById('totalCount') && (document.getElementById('totalCount').textContent = data.total);
        document.getElementById('tfCount') && (document.getElementById('tfCount').textContent = data.tf);
        document.getElementById('noTfCount') && (document.getElementById('noTfCount').textContent = data.no_tf);
        const hist = document.getElementById('historyList');
        if(hist){
          hist.innerHTML = '';
          (data.history || []).forEach(item => {
            const li = document.createElement('li');
            li.className = 'history-item';
            li.innerHTML = `<span class="time">${new Date(item.time).toLocaleString()}</span> <strong>${item.class===1?'TF':'No-TF'}</strong> <span class="conf">(${(item.confidence*100).toFixed(1)}%)</span><div class="hseq">${item.seq}</div>`;
            hist.appendChild(li);
          });
        }
      }
    }catch(err){
      // no mostrar error al usuario para polling
    }
  }

  // Iniciar polling cuando appCard visible
  let statsTimer = null;
  function startStatsPolling(){
    fetchStats();
    if(statsTimer) clearInterval(statsTimer);
    statsTimer = setInterval(fetchStats, 4000);
  }
  function stopStatsPolling(){
    if(statsTimer) clearInterval(statsTimer);
    statsTimer = null;
  }

  // Si autostart (login redirige con autostart=1), appCard ya visible: iniciar polling
  if(appCard && !appCard.classList.contains('hidden')){
    startStatsPolling();
  }

  // Mostrar appCard si URL contiene autostart=1 (caso directo sin reload)
  const params = new URLSearchParams(window.location.search);
  if(params.get('autostart') === '1' && appCard){
    if(hero) hero.style.display = 'none';
    appCard.classList.remove('hidden');
    startStatsPolling();
  }

  window.addEventListener('beforeunload', () => stopStatsPolling());

  // small CSS shake keyframe injection
  const style = document.createElement('style');
  style.innerHTML = '@keyframes shake{0%{transform:translateX(0)}25%{transform:translateX(-6px)}50%{transform:translateX(6px)}75%{transform:translateX(-4px)}100%{transform:translateX(0)}}';
  document.head.appendChild(style);

  // Manejo actualizado del modal info
  const infoBtn = document.getElementById('infoBtn');
  const infoSection = document.getElementById('info');
  const closeInfoBtn = document.getElementById('closeInfoBtn');
  const closeInfoBtn2 = document.getElementById('closeInfoBtn2');

  function openInfo(){
    if(!infoSection) return;
    infoSection.classList.remove('hidden');
    infoSection.setAttribute('aria-hidden','false');
    document.body.style.overflow = 'hidden';
    setTimeout(()=>{ (closeInfoBtn || closeInfoBtn2)?.focus(); }, 200);
  }
  function closeInfo(){
    if(!infoSection) return;
    infoSection.classList.add('hidden');
    infoSection.setAttribute('aria-hidden','true');
    document.body.style.overflow = '';
    // devolver foco al boton info si existe
    infoBtn && infoBtn.focus();
  }

  infoBtn && infoBtn.addEventListener('click', (e)=>{ e.preventDefault(); openInfo(); });
  closeInfoBtn && closeInfoBtn.addEventListener('click', (e)=>{ e.preventDefault(); closeInfo(); });
  closeInfoBtn2 && closeInfoBtn2.addEventListener('click', (e)=>{ e.preventDefault(); closeInfo(); });
  document.addEventListener('keydown', (e)=>{ if(e.key==='Escape') closeInfo(); });

  // File handling
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const seqTextarea = document.getElementById('sequence');

  function showDropZone() {
    dropZone.classList.add('active');
  }
  function hideDropZone() {
    dropZone.classList.remove('active');
  }
  function highlightDropZone() {
    dropZone.classList.add('highlight');
  }
  function unhighlightDropZone() {
    dropZone.classList.remove('highlight');
  }

  // Mostrar dropZone al arrastrar sobre la ventana
  window.addEventListener('dragenter', showDropZone);
  dropZone.addEventListener('dragleave', hideDropZone);
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    highlightDropZone();
  });

  // Procesar archivo
  async function handleFile(file) {
    try {
      const text = await file.text();
      // Extraer secuencia del FASTA (asume formato simple)
      const lines = text.split('\n');
      const sequence = lines
        .filter(line => !line.startsWith('>'))
        .join('')
        .trim();
      
      if (sequence) {
        seqTextarea.value = sequence;
        hideDropZone();
      }
    } catch (err) {
      alert('Error al leer el archivo FASTA');
      console.error(err);
    }
  }

  // Drop handlers
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    unhighlightDropZone();
    hideDropZone();
    
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.fasta') || file.name.endsWith('.fa'))) {
      handleFile(file);
    }
  });

  // File input change
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFile(file);
    }
  });

  // Función para mostrar mensaje de función en desarrollo
  function showDevelopmentMessage(e) {
    e.preventDefault();
    alert('Función en desarrollo. Contactarse con servicio técnico: support@genetfinder.com');
  }

  // Agregar manejadores a botones sin funcionalidad
  document.addEventListener('DOMContentLoaded', function() {
    const devButtons = [
        'exportBtn',
        'apiBtn',
        'downloadBtn',
        'copyResultBtn'
    ];

    devButtons.forEach(btnId => {
        const btn = document.getElementById(btnId);
        if (btn) {
            btn.addEventListener('click', showDevelopmentMessage);
        }
    });
  });
});
