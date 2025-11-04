class AdminDashboard {
    constructor() {
        this.updateInterval = 3000;
        this.initialize();
    }

    initialize() {
        this.startRealTimeUpdates();
        this.setupEventListeners();
    }

    startRealTimeUpdates() {
        this.updateDashboard();
        setInterval(() => this.updateDashboard(), this.updateInterval);
    }

    async updateDashboard() {
        try {
            const response = await fetch('/admin/system_stats');
            const data = await response.json();
            this.updateMetrics(data);
            this.updateActiveUsers(data.users);
            this.updateActivityStream(data.recent_activity);
            this.updateSystemStatus(data);
        } catch (error) {
            console.warn('Error actualizando dashboard:', error);
        }
    }

    updateMetrics(data) {
        this.setElementText('totalPredictions', data.stats.total);
        this.setElementText('activeUsersCount', data.active_users);
        this.setElementText('modelAccuracy', (data.model_accuracy * 100).toFixed(1) + '%');
        this.setElementText('memoryUsage', data.memory_usage);
    }

    updateActiveUsers(users) {
        const container = document.getElementById('activeUsersList');
        if (!container) return;

        container.innerHTML = Object.entries(users)
            .filter(([, user]) => user.connected)
            .map(([username, user]) => `
                <div class="user-item">
                    <div class="user-info">
                        <strong>${username}</strong>
                        <small class="device-info">${user.device || 'Dispositivo desconocido'}</small>
                    </div>
                    <span class="status-badge online">Online</span>
                </div>
            `).join('');
    }

    updateActivityStream(activities) {
        const container = document.getElementById('activityStream');
        if (!container) return;

        container.innerHTML = activities.map(activity => `
            <div class="activity-item">
                <span class="activity-time">${new Date(activity.time).toLocaleString()}</span>
                <span class="activity-description">${activity.description}</span>
            </div>
        `).join('');
    }

    setElementText(id, text) {
        const element = document.getElementById(id);
        if (element) element.textContent = text;
    }

    setupEventListeners() {
        const exportBtn = document.getElementById('exportDataBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                window.location.href = '/admin/export_data';
            });
        }

        const reloadModelBtn = document.getElementById('reloadModelBtn');
        if (reloadModelBtn) {
            reloadModelBtn.addEventListener('click', async () => {
                if (!confirm('¿Recargar modelo? Esto puede tomar unos segundos.')) return;
                try {
                    const response = await fetch('/reload_model', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'}
                    });
                    const data = await response.json();
                    if (data.ok) {
                        alert('Modelo recargado exitosamente');
                        location.reload();
                    } else {
                        alert('Error al recargar el modelo: ' + data.msg);
                    }
                } catch (error) {
                    alert('Error en la operación');
                }
            });
        }
    }
}

// Inicializar dashboard cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    new AdminDashboard();
});
