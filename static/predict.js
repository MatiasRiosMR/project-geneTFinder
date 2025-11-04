class PredictApp {
    constructor() {
        this.form = document.getElementById('seqForm');
        this.sequence = document.getElementById('sequence');
        this.resultPanel = document.getElementById('resultPanel');
        this.loader = document.getElementById('loader');
        this.setupListeners();
        this.startStatsUpdater();
    }

    setupListeners() {
        // Form submit
        this.form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.predict();
        });

        // File upload
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        }

        // Clear button
        const clearBtn = document.getElementById('clearBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearForm());
        }

        // Copy sequence
        const copyBtn = document.getElementById('copySeqBtn');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => this.copySequence());
        }
    }

    async predict() {
        if (!this.sequence.value.trim()) {
            this.showToast('Ingresa una secuencia', 'error');
            return;
        }

        this.loader.classList.remove('hidden');
        
        try {
            const formData = new FormData(this.form);
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Error en la predicción');
            
            const result = await response.text();
            this.resultPanel.innerHTML = result;
            this.resultPanel.classList.remove('hidden');
            this.showToast('Predicción completada', 'success');

        } catch (error) {
            this.showToast('Error al procesar la predicción', 'error');
            console.error(error);
        } finally {
            this.loader.classList.add('hidden');
        }
    }

    handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            this.sequence.value = e.target.result;
            this.showToast('Archivo FASTA cargado', 'success');
        };
        reader.onerror = () => this.showToast('Error al leer el archivo', 'error');
        reader.readAsText(file);
    }

    clearForm() {
        this.sequence.value = '';
        this.resultPanel.classList.add('hidden');
        this.sequence.focus();
    }

    copySequence() {
        const seq = this.sequence.value.trim();
        if (!seq) {
            this.showToast('No hay secuencia para copiar', 'error');
            return;
        }
        navigator.clipboard.writeText(seq)
            .then(() => this.showToast('Secuencia copiada', 'success'))
            .catch(() => this.showToast('Error al copiar', 'error'));
    }

    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        if (!toast) return;
        toast.textContent = message;
        toast.className = `toast show ${type}`;
        setTimeout(() => toast.className = 'toast', 3000);
    }

    startStatsUpdater() {
        const updateStats = async () => {
            try {
                const response = await fetch('/stats_json');
                const data = await response.json();
                this.updateStatistics(data);
            } catch (error) {
                console.warn('Error actualizando estadísticas:', error);
            }
        };

        updateStats();
        setInterval(updateStats, 3000);
    }

    updateStatistics(data) {
        const setText = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        };

        setText('totalCount', data.total);
        setText('tfCount', data.tf);
        setText('noTfCount', data.no_tf);
        setText('avgConfidence', (data.avg_confidence * 100).toFixed(1) + '%');
        setText('connectedCount', `Conectados: ${data.connected_users || 0}`);

        this.updateHistory(data.history || []);
    }

    updateHistory(history) {
        const historyList = document.getElementById('historyList');
        if (!historyList) return;

        historyList.innerHTML = '';
        history.forEach(h => {
            const li = document.createElement('li');
            const time = new Date(h.time).toLocaleString();
            const cls = h.class === 1 ? 'TF' : 'No-TF';
            li.innerHTML = `
                <span>${time}</span> — 
                <span>${h.seq}</span> — 
                <span>${cls}</span> — 
                <span>${(h.confidence * 100).toFixed(1)}%</span>
            `;
            historyList.appendChild(li);
        });
    }
}

// Inicializar app cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    new PredictApp();
});
