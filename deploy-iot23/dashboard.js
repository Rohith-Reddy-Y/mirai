/* IoT-23 Dashboard Logic */
document.addEventListener('DOMContentLoaded', () => {
    renderStats();
    renderPipeline();
    renderTable();
    renderCharts();
    renderFeatures();
});

/* ── Helpers ── */
function fmt(n, d = 4) { if (n === null || n === undefined) return '—'; return Number(n).toFixed(d); }
function fmtPct(n) { if (n === null || n === undefined) return '—'; return (n * 100).toFixed(2) + '%'; }
function fmtInt(n) { if (n === null || n === undefined) return '—'; return Number(n).toLocaleString(); }
function metricBadge(val, thresholds = [0.95, 0.8]) {
    if (val === null || val === undefined) return '<span class="badge badge-gray">N/A</span>';
    const cls = val >= thresholds[0] ? 'badge-green' : val >= thresholds[1] ? 'badge-amber' : 'badge-red';
    return `<span class="badge ${cls}">${fmtPct(val)}</span>`;
}
function statusBadge(s) {
    if (s === 'trained') return '<span class="badge badge-green">Trained</span>';
    if (s === 'single_class_train') return '<span class="badge badge-amber">Single Class</span>';
    return '<span class="badge badge-red">Error</span>';
}

/* ── Stats ── */
function renderStats() {
    const trained = DATASETS.filter(d => d.status === 'trained');
    const totalRows = DATASETS.reduce((a, d) => a + d.trainRows + d.valRows + d.testRows, 0);
    const testAucs = trained.filter(d => d.test && d.test.rocAuc !== null).map(d => d.test.rocAuc);
    const avgAuc = testAucs.length ? testAucs.reduce((a, b) => a + b, 0) / testAucs.length : 0;
    const best = trained.reduce((a, d) => (d.test && d.test.rocAuc !== null && d.test.rocAuc > (a.test?.rocAuc || 0)) ? d : a, trained[0]);
    document.getElementById('stat-datasets').textContent = '23';
    document.getElementById('stat-trained').textContent = trained.length;
    document.getElementById('stat-rows').textContent = (totalRows / 1e6).toFixed(1) + 'M';
    document.getElementById('stat-auc').textContent = fmtPct(avgAuc);
    document.getElementById('stat-best').textContent = best ? best.id.replace('dataset', '#') : '—';
}

/* ── Pipeline ── */
function renderPipeline() {
    const el = document.getElementById('pipeline');
    el.innerHTML = PIPELINE_STEPS.map((s, i) =>
        (i > 0 ? '<span class="pipe-arrow">→</span>' : '') +
        `<div class="pipe-step fade-in" style="animation-delay:${i * 0.08}s">
      <div class="pipe-icon">${s.icon}</div>
      <div class="pipe-title">${s.title}</div>
      <div class="pipe-desc">${s.desc}</div>
    </div>`
    ).join('');
}

/* ── Table ── */
let sortCol = 'id', sortDir = 1;
function renderTable() {
    const search = (document.getElementById('searchInput')?.value || '').toLowerCase();
    const filter = document.getElementById('statusFilter')?.value || 'all';
    let data = [...DATASETS];
    if (search) data = data.filter(d => d.id.includes(search));
    if (filter !== 'all') data = data.filter(d => d.status === filter);
    data.sort((a, b) => {
        let va = a[sortCol], vb = b[sortCol];
        if (sortCol === 'testAuc') { va = a.test?.rocAuc ?? -1; vb = b.test?.rocAuc ?? -1 }
        if (sortCol === 'testAcc') { va = a.test?.acc ?? -1; vb = b.test?.acc ?? -1 }
        if (sortCol === 'trainRows') { va = a.trainRows; vb = b.trainRows }
        if (typeof va === 'string') return sortDir * va.localeCompare(vb);
        return sortDir * ((va ?? -1) - (vb ?? -1));
    });
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = data.map(d => {
        const ta = d.test?.rocAuc; const tAcc = d.test?.acc;
        const totalTrain = d.classTrain ? (d.classTrain[0] || 0) + (d.classTrain[1] || 0) : d.trainRows;
        const malPct = d.classTrain && d.classTrain[1] ? ((d.classTrain[1] / totalTrain) * 100).toFixed(1) + '%' : '0%';
        return `<tr onclick="toggleDetail('${d.id}')">
      <td style="font-weight:600;color:var(--cyan)">${d.id.replace('dataset', 'Dataset ')}</td>
      <td>${statusBadge(d.status)}</td>
      <td>${fmtInt(d.trainRows)}</td>
      <td>${d.features}</td>
      <td>${malPct}</td>
      <td>${metricBadge(ta)}</td>
      <td>${metricBadge(tAcc)}</td>
      <td>${d.test?.cm ? fmtInt(d.test.cm.tp) : '—'}</td>
      <td>${d.test?.cm ? fmtInt(d.test.cm.fp) : '—'}</td>
    </tr>
    <tr class="detail-panel" id="detail-${d.id}"><td colspan="9">${renderDetail(d)}</td></tr>`;
    }).join('');
}

function renderDetail(d) {
    if (!d.test && !d.train) return '<div class="detail-content"><div class="detail-card"><h4>No Model</h4><p style="color:var(--text-dim);font-size:.85rem">This dataset was skipped (' + d.status.replace(/_/g, ' ') + ').</p></div></div>';
    let html = '<div class="detail-content">';
    // Metrics card
    const splits = [{ name: 'Train', m: d.train }, { name: 'Validation', m: d.val }, { name: 'Test', m: d.test }];
    splits.forEach(s => {
        if (!s.m) return;
        html += `<div class="detail-card"><h4>${s.name} Metrics</h4>`;
        if (s.m.note === 'single_class') {
            html += `<div class="detail-row"><span class="dl">Note</span><span class="dv badge badge-amber">Single Class</span></div>`;
            html += `<div class="detail-row"><span class="dl">Accuracy</span><span class="dv">${fmtPct(s.m.acc)}</span></div>`;
        } else {
            html += `<div class="detail-row"><span class="dl">ROC-AUC</span><span class="dv">${fmt(s.m.rocAuc)}</span></div>`;
            html += `<div class="detail-row"><span class="dl">PR-AUC</span><span class="dv">${fmt(s.m.prAuc)}</span></div>`;
            html += `<div class="detail-row"><span class="dl">Accuracy</span><span class="dv">${fmtPct(s.m.acc)}</span></div>`;
            if (s.m.f1 !== undefined) html += `<div class="detail-row"><span class="dl">Best F1</span><span class="dv">${fmt(s.m.f1)}</span></div>`;
            if (s.m.cm) {
                html += '<div class="cm-grid">';
                html += `<div class="cm-cell cm-tp">TP: ${fmtInt(s.m.cm.tp)}</div>`;
                html += `<div class="cm-cell cm-fp">FP: ${fmtInt(s.m.cm.fp)}</div>`;
                html += `<div class="cm-cell cm-fn">FN: ${fmtInt(s.m.cm.fn)}</div>`;
                html += `<div class="cm-cell cm-tn">TN: ${fmtInt(s.m.cm.tn)}</div>`;
                html += '</div>';
            }
        }
        html += '</div>';
    });
    // Meta card
    if (d.scalePos !== undefined) {
        html += `<div class="detail-card"><h4>Training Config</h4>`;
        html += `<div class="detail-row"><span class="dl">Scale Pos Weight</span><span class="dv">${d.scalePos}</span></div>`;
        html += `<div class="detail-row"><span class="dl">Chosen Threshold</span><span class="dv">${d.threshold}</span></div>`;
        html += `<div class="detail-row"><span class="dl">Feature Count</span><span class="dv">${d.features}</span></div>`;
        html += `<div class="detail-row"><span class="dl">Total Rows</span><span class="dv">${fmtInt(d.trainRows + d.valRows + d.testRows)}</span></div>`;
        html += '</div>';
    }
    html += '</div>';
    return html;
}

function toggleDetail(id) {
    const el = document.getElementById('detail-' + id);
    if (!el) return;
    const wasOpen = el.classList.contains('show');
    document.querySelectorAll('.detail-panel').forEach(p => p.classList.remove('show'));
    document.querySelectorAll('tr.expanded').forEach(t => t.classList.remove('expanded'));
    if (!wasOpen) {
        el.classList.add('show');
        el.previousElementSibling?.classList.add('expanded');
    }
}

function sortTable(col) {
    if (sortCol === col) sortDir *= -1; else { sortCol = col; sortDir = 1; }
    document.querySelectorAll('th').forEach(th => {
        const icon = th.querySelector('.sort-icon');
        if (icon) icon.textContent = th.dataset.col === col ? (sortDir > 0 ? '▲' : '▼') : '⇅';
    });
    renderTable();
}

/* ── Charts ── */
function renderCharts() {
    const trained = DATASETS.filter(d => d.status === 'trained' && d.test && d.test.rocAuc !== null);
    const labels = trained.map(d => d.id.replace('dataset', 'D'));
    const aucs = trained.map(d => d.test.rocAuc);
    const accs = trained.map(d => d.test.acc);

    // ROC-AUC Bar Chart
    new Chart(document.getElementById('aucChart'), {
        type: 'bar',
        data: {
            labels, datasets: [{
                label: 'Test ROC-AUC', data: aucs,
                backgroundColor: aucs.map(v => v >= 0.95 ? 'rgba(6,182,212,0.7)' : v >= 0.8 ? 'rgba(245,158,11,0.7)' : 'rgba(239,68,68,0.7)'),
                borderColor: aucs.map(v => v >= 0.95 ? '#06b6d4' : v >= 0.8 ? '#f59e0b' : '#ef4444'), borderWidth: 1, borderRadius: 6
            }]
        },
        options: {
            responsive: true, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => 'ROC-AUC: ' + c.raw.toFixed(4) } } },
            scales: {
                y: { beginAtZero: false, min: 0.3, grid: { color: 'rgba(42,58,92,0.5)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            }
        }
    });

    // Accuracy Bar Chart
    new Chart(document.getElementById('accChart'), {
        type: 'bar',
        data: {
            labels, datasets: [{
                label: 'Test Accuracy', data: accs,
                backgroundColor: 'rgba(168,85,247,0.6)', borderColor: '#a855f7', borderWidth: 1, borderRadius: 6
            }]
        },
        options: {
            responsive: true, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => 'Accuracy: ' + (c.raw * 100).toFixed(2) + '%' } } },
            scales: {
                y: { beginAtZero: false, min: 0.3, grid: { color: 'rgba(42,58,92,0.5)' }, ticks: { color: '#94a3b8', callback: v => (v * 100) + '%' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            }
        }
    });

    // Dataset Size Chart
    const allLabels = DATASETS.map(d => d.id.replace('dataset', 'D'));
    const sizes = DATASETS.map(d => d.trainRows);
    new Chart(document.getElementById('sizeChart'), {
        type: 'bar',
        data: {
            labels: allLabels, datasets: [{
                label: 'Training Rows', data: sizes,
                backgroundColor: 'rgba(20,184,166,0.6)', borderColor: '#14b8a6', borderWidth: 1, borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y', responsive: true, plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: c => c.raw.toLocaleString() + ' rows' } }
            },
            scales: {
                x: { grid: { color: 'rgba(42,58,92,0.5)' }, ticks: { color: '#94a3b8', callback: v => (v / 1e6).toFixed(0) + 'M' } },
                y: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10 } } }
            }
        }
    });

    // Class Distribution Doughnut
    const topTrained = DATASETS.filter(d => d.classTrain && d.classTrain[1] !== undefined).slice(0, 8);
    const doughData = topTrained.map(d => ({
        label: d.id.replace('dataset', 'D'),
        benign: d.classTrain[0] || 0,
        malicious: d.classTrain[1] || 0
    }));
    new Chart(document.getElementById('classChart'), {
        type: 'bar',
        data: {
            labels: doughData.map(d => d.label), datasets: [
                { label: 'Benign', data: doughData.map(d => d.benign / (d.benign + d.malicious) * 100), backgroundColor: 'rgba(59,130,246,0.7)', borderRadius: 4 },
                { label: 'Malicious', data: doughData.map(d => d.malicious / (d.benign + d.malicious) * 100), backgroundColor: 'rgba(239,68,68,0.7)', borderRadius: 4 }
            ]
        },
        options: {
            responsive: true, plugins: {
                legend: { labels: { color: '#94a3b8' } },
                tooltip: { callbacks: { label: c => c.dataset.label + ': ' + c.raw.toFixed(1) + '%' } }
            },
            scales: {
                x: { stacked: true, grid: { display: false }, ticks: { color: '#94a3b8' } },
                y: { stacked: true, max: 100, grid: { color: 'rgba(42,58,92,0.5)' }, ticks: { color: '#94a3b8', callback: v => v + '%' } }
            }
        }
    });
}

/* ── Features ── */
function renderFeatures() {
    const icons = { duration: '⏱️', local_orig: '📡', local_resp: '📡', missed_bytes: '❌', orig_bytes: '📤', orig_ip_bytes: '📤', orig_pkts: '📦', proto_icmp: '🌐', proto_tcp: '🌐', proto_udp: '🌐', resp_bytes: '📥', resp_ip_bytes: '📥', resp_pkts: '📦', 'service_-': '⚙️', service_dns: '🔍', service_http: '🌍' };
    const el = document.getElementById('features');
    el.innerHTML = FEATURES.map((f, i) =>
        `<div class="feature-tag fade-in" style="animation-delay:${i * 0.05}s">
      <span class="f-icon">${icons[f] || '📊'}</span>${f}
    </div>`
    ).join('');
}

/* ═══════════════════════════════════════════
   LIVE DETECTION
   ═══════════════════════════════════════════ */
let _selectedFile = null;
let _detectResults = null;
let _detectCharts = {};

// Upload zone interactions
(function initDetect() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('detectFile');
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; handleFileSelect(); }
    });
    input.addEventListener('change', handleFileSelect);
})();

function handleFileSelect() {
    const input = document.getElementById('detectFile');
    if (!input.files.length) return;
    _selectedFile = input.files[0];
    document.getElementById('detectFileName').textContent = '📄 ' + _selectedFile.name + ' (' + (_selectedFile.size / 1024).toFixed(1) + ' KB)';
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('detectControls').style.display = 'block';
}

function clearDetection() {
    _selectedFile = null;
    _detectResults = null;
    document.getElementById('detectFile').value = '';
    document.getElementById('uploadZone').style.display = '';
    document.getElementById('detectControls').style.display = 'none';
    document.getElementById('detectResults').style.display = 'none';
    document.getElementById('detectProgress').style.display = 'none';
    // Destroy charts
    Object.values(_detectCharts).forEach(c => c.destroy());
    _detectCharts = {};
}

async function runDetection() {
    if (!_selectedFile) return;
    const btn = document.getElementById('btnDetect');
    btn.disabled = true; btn.textContent = '⏳ Detecting...';
    const prog = document.getElementById('detectProgress');
    const fill = document.getElementById('progressFill');
    const txt = document.getElementById('progressText');
    prog.style.display = 'block';
    fill.style.width = '20%'; txt.textContent = 'Uploading file...';

    const form = new FormData();
    form.append('file', _selectedFile);
    form.append('threshold', document.getElementById('detectThreshold').value);
    form.append('combine', document.getElementById('detectCombine').value);

    try {
        fill.style.width = '40%'; txt.textContent = 'Running ensemble detection...';
        const resp = await fetch('/api/predict', { method: 'POST', body: form });
        if (!resp.ok) throw new Error('Server error: ' + resp.status);
        fill.style.width = '80%'; txt.textContent = 'Processing results...';
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        fill.style.width = '100%'; txt.textContent = '✅ Detection complete!';
        _detectResults = data;
        setTimeout(() => { prog.style.display = 'none'; renderDetectResults(data); }, 500);
    } catch (err) {
        fill.style.width = '100%'; fill.style.background = 'var(--red)';
        txt.textContent = '❌ Error: ' + err.message;
    }
    btn.disabled = false; btn.textContent = '🔍 Run Detection';
}

function renderDetectResults(data) {
    document.getElementById('detectResults').style.display = 'block';
    document.getElementById('det-total').textContent = data.total.toLocaleString();
    document.getElementById('det-benign').textContent = data.benign.toLocaleString();
    document.getElementById('det-malicious').textContent = data.malicious.toLocaleString();
    const tl = data.threat_pct;
    const det_threat = document.getElementById('det-threat');
    det_threat.textContent = tl.toFixed(1) + '%';
    det_threat.className = 'value ' + (tl > 50 ? 'detect-malicious' : tl > 20 ? '' : 'detect-benign');
    document.getElementById('tab-ben-count').textContent = data.benign.toLocaleString();
    document.getElementById('tab-mal-count').textContent = data.malicious.toLocaleString();

    // Destroy old charts
    Object.values(_detectCharts).forEach(c => c.destroy());
    _detectCharts = {};

    // Pie chart
    _detectCharts.pie = new Chart(document.getElementById('detectPieChart'), {
        type: 'doughnut',
        data: {
            labels: ['Benign (Allowed)', 'Malicious (Dropped)'],
            datasets: [{
                data: [data.benign, data.malicious], backgroundColor: ['rgba(34,197,94,0.7)', 'rgba(239,68,68,0.7)'],
                borderColor: ['#22c55e', '#ef4444'], borderWidth: 2, hoverOffset: 8
            }]
        },
        options: { responsive: true, plugins: { legend: { labels: { color: '#94a3b8' } } } }
    });

    // Histogram
    const bins = 20;
    const benHist = histBins(data.prob_distribution.benign, bins);
    const malHist = histBins(data.prob_distribution.malicious, bins);
    _detectCharts.hist = new Chart(document.getElementById('detectHistChart'), {
        type: 'bar',
        data: {
            labels: benHist.labels,
            datasets: [
                { label: 'Benign', data: benHist.counts, backgroundColor: 'rgba(34,197,94,0.5)', borderColor: '#22c55e', borderWidth: 1, borderRadius: 3 },
                { label: 'Malicious', data: malHist.counts, backgroundColor: 'rgba(239,68,68,0.5)', borderColor: '#ef4444', borderWidth: 1, borderRadius: 3 }
            ]
        },
        options: {
            responsive: true, plugins: { legend: { labels: { color: '#94a3b8' } } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 9 } } },
                y: { grid: { color: 'rgba(42,58,92,0.5)' }, ticks: { color: '#94a3b8' } }
            }
        }
    });

    // Model chart
    const ms = data.model_stats.filter(m => m.flagged > 0).sort((a, b) => b.flagged - a.flagged);
    _detectCharts.model = new Chart(document.getElementById('detectModelChart'), {
        type: 'bar',
        data: {
            labels: ms.map(m => m.name.replace('dataset', 'D')),
            datasets: [{
                label: 'Packets Flagged', data: ms.map(m => m.flagged),
                backgroundColor: ms.map((_, i) => `rgba(6,182,212,${0.4 + i * 0.04})`),
                borderColor: '#06b6d4', borderWidth: 1, borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y', responsive: true, plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(42,58,92,0.5)' }, ticks: { color: '#94a3b8' } },
                y: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10 } } }
            }
        }
    });

    // Show allowed tab by default
    switchTab('allowed');
    document.getElementById('detectResults').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function histBins(arr, n) {
    const labels = [], counts = new Array(n).fill(0);
    for (let i = 0; i < n; i++) labels.push((i / n).toFixed(2));
    arr.forEach(v => { const idx = Math.min(Math.floor(v * n), n - 1); counts[idx]++; });
    return { labels, counts };
}

function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', i === (tab === 'allowed' ? 0 : 1)));
    if (!_detectResults) return;
    const rows = _detectResults.results.filter(r => tab === 'allowed' ? r.verdict === 'BENIGN' : r.verdict === 'MALICIOUS');
    const thead = document.getElementById('detectTHead');
    const tbody = document.getElementById('detectTBody');
    thead.innerHTML = '<tr><th>#</th><th>Verdict</th><th>Probability</th><th>Top Model</th><th>Confidence</th></tr>';
    tbody.innerHTML = rows.slice(0, 200).map(r =>
        `<tr><td>${r.index}</td><td><span class="badge ${r.verdict === 'BENIGN' ? 'badge-green' : 'badge-red'}">${r.verdict === 'BENIGN' ? '✅ ALLOWED' : '❌ DROPPED'}</span></td>` +
        `<td>${(r.probability * 100).toFixed(2)}%</td><td style="color:var(--cyan)">${r.top_model.replace('dataset', 'D')}</td>` +
        `<td>${(r.top_model_conf * 100).toFixed(2)}%</td></tr>`
    ).join('');
}

function downloadResults(type) {
    if (!_detectResults) return;
    const rows = _detectResults.results.filter(r => type === 'allowed' ? r.verdict === 'BENIGN' : r.verdict === 'MALICIOUS');
    const header = 'index,verdict,probability,top_model,top_model_conf\n';
    const csv = header + rows.map(r => `${r.index},${r.verdict},${r.probability},${r.top_model},${r.top_model_conf}`).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = type === 'allowed' ? 'allowed_packets.csv' : 'dropped_packets.csv';
    a.click();
}

