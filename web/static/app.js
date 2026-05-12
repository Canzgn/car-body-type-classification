/* app.js — Araba Gövde Tipi Sınıflandırıcı Frontend */

const dropZone      = document.getElementById('dropZone');
const fileInput     = document.getElementById('fileInput');
const previewWrapper = document.getElementById('previewWrapper');
const previewImage  = document.getElementById('previewImage');
const predictBtn    = document.getElementById('predictBtn');
const resetBtn      = document.getElementById('resetBtn');
const btnText       = document.getElementById('btnText');
const btnLoader     = document.getElementById('btnLoader');
const resultPanel   = document.getElementById('resultPanel');
const resultClass   = document.getElementById('resultClass');
const resultConf    = document.getElementById('resultConf');
const errorMsg      = document.getElementById('errorMsg');

let currentFile = null;
let barChart    = null;

// ── Drop Zone ──────────────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});

fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// ── Dosya İşle ────────────────────────────────────────────────────────────
function handleFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImage.src = e.target.result;
    dropZone.style.display        = 'none';
    previewWrapper.style.display  = 'flex';
    resultPanel.style.display     = 'none';
    hideError();
  };
  reader.readAsDataURL(file);
}

// ── Sıfırla ───────────────────────────────────────────────────────────────
resetBtn.addEventListener('click', reset);

function reset() {
  currentFile = null;
  fileInput.value = '';
  previewWrapper.style.display = 'none';
  dropZone.style.display       = 'flex';
  resultPanel.style.display    = 'none';
  hideError();
  if (barChart) { barChart.destroy(); barChart = null; }
}

// ── Tahmin Yap ────────────────────────────────────────────────────────────
predictBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  setLoading(true);
  hideError();

  try {
    const formData = new FormData();
    formData.append('file', currentFile);

    const res = await fetch('/predict', { method: 'POST', body: formData });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || 'Sunucu hatası');
    }

    const data = await res.json();
    showResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

// ── Sonuçları Göster ──────────────────────────────────────────────────────
function showResults(data) {
  resultClass.textContent = data.prediction;
  resultConf.textContent  = `Güven: ${data.confidence_percent}`;
  resultPanel.style.display = 'block';

  const labels = data.all_predictions.map(p => p.class_label);
  const values = data.all_predictions.map(p => +(p.probability * 100).toFixed(2));
  const colors = data.all_predictions.map(p =>
    p.class_key === data.class_key ? '#2563eb' : '#bfdbfe'
  );

  if (barChart) barChart.destroy();

  barChart = new Chart(document.getElementById('barChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Olasılık (%)',
        data: values,
        backgroundColor: colors,
        borderRadius: 5,
        borderSkipped: false,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.parsed.x.toFixed(2)}%` },
        },
      },
      scales: {
        x: {
          max: 100,
          ticks: { callback: v => v + '%' },
          grid: { color: 'rgba(0,0,0,.06)' },
        },
        y: {
          grid: { display: false },
        },
      },
    },
  });
}

// ── Yardımcı ──────────────────────────────────────────────────────────────
function setLoading(on) {
  predictBtn.disabled         = on;
  btnText.style.display       = on ? 'none'         : 'inline';
  btnLoader.style.display     = on ? 'inline-block' : 'none';
}

function showError(msg) {
  errorMsg.textContent    = `Hata: ${msg}`;
  errorMsg.style.display  = 'block';
  resultPanel.style.display = 'block';
  resultClass.textContent = '—';
  resultConf.textContent  = '—';
}

function hideError() {
  errorMsg.style.display = 'none';
}
