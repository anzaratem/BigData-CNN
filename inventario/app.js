// ============================================================================
// APP.JS - DETECCIÓN DE OBJETOS CON TFLITE + VISUALIZACIONES
// ============================================================================

// Configuración
const CONFIG = {
  MODEL_URL: './modelo.tflite',
  IMG_SIZE: 640,
  CONF_THRESHOLD: 0.25,
  IOU_THRESHOLD: 0.45,
  CLASSES: ['CPU', 'Mesa', 'Mouse', 'Pantalla', 'Silla', 'Teclado'],
  ICONS: ['🖥️', '🪑', '🖱️', '💻', '💺', '⌨️']
};

// Variables globales
let model = null;
let selectedFile = null;
let pieChart = null;
let barChart = null;
let allDetections = [];

// Elementos del DOM
const fileInput = document.getElementById('fileInput');
const btnRun = document.getElementById('btnRun');
const statusEl = document.getElementById('status');
const imgEl = document.getElementById('img');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const countsBody = document.getElementById('countsBody');
const modelStatus = document.getElementById('modelStatus');
const totalObjects = document.getElementById('totalObjects');
const avgConfidence = document.getElementById('avgConfidence');
const detectionCount = document.getElementById('detectionCount');
const imageSize = document.getElementById('imageSize');
const resultsContainer = document.getElementById('resultsContainer');
const detectionsList = document.getElementById('detectionsList');

// ============================================================================
// INICIALIZACIÓN
// ============================================================================

async function init() {
  updateStatus('🔄 Cargando modelo TFLite...', 'loading');
  modelStatus.textContent = 'Cargando...';
  
  try {
    if (typeof tflite === 'undefined') {
      throw new Error('TFLite no está cargado. Verifica tu conexión.');
    }
    
    model = await tflite.loadTFLiteModel(CONFIG.MODEL_URL);
    
    updateStatus('✅ Modelo listo. Selecciona una imagen del salón.', 'success');
    modelStatus.textContent = 'YOLOv8 ✓';
    fileInput.disabled = false;
    
  } catch (error) {
    console.error('Error:', error);
    updateStatus(`❌ Error: ${error.message}`, 'error');
    modelStatus.textContent = 'Error ✗';
  }
}

// ============================================================================
// UTILIDADES
// ============================================================================

function updateStatus(message, type = '') {
  const icon = {
    loading: '<i class="fas fa-circle-notch fa-spin"></i>',
    success: '<i class="fas fa-check-circle"></i>',
    error: '<i class="fas fa-exclamation-circle"></i>'
  }[type] || '';
  
  statusEl.innerHTML = `${icon} ${message}`;
  statusEl.className = `status-message ${type}`;
}

// ============================================================================
// EVENTOS
// ============================================================================

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  
  selectedFile = file;
  
  const reader = new FileReader();
  reader.onload = (event) => {
    imgEl.src = event.target.result;
    imgEl.style.display = 'block';
    imgEl.onload = () => {
      imageSize.textContent = `${imgEl.naturalWidth} x ${imgEl.naturalHeight} px`;
    };
    
    if (model) {
      btnRun.disabled = false;
      updateStatus('✅ Imagen cargada. Haz clic en "Detectar Objetos"', 'success');
    }
  };
  reader.readAsDataURL(file);
});

btnRun.addEventListener('click', detectObjects);

// ============================================================================
// DETECCIÓN
// ============================================================================

async function detectObjects() {
  updateStatus('🔍 Detectando objetos...', 'loading');
  btnRun.disabled = true;
  
  try {
    const img = await loadImage(imgEl.src);
    const inputTensor = preprocessImage(img);
    const outputTensor = model.predict(inputTensor);
    const outputData = await outputTensor.data();
    
    const detections = processOutput(outputData, img.width, img.height);
    allDetections = detections;
    
    inputTensor.dispose();
    outputTensor.dispose();
    
    if (detections.length === 0) {
      updateStatus('⚠️ No se detectaron objetos', 'error');
      resultsContainer.style.display = 'block';
      drawImage(img);
      updateAllStats([], img);
    } else {
      drawDetections(img, detections);
      updateAllStats(detections, img);
      resultsContainer.style.display = 'block';
      updateStatus(`✅ Detectados ${detections.length} objetos`, 'success');
    }
    
  } catch (error) {
    console.error('Error:', error);
    updateStatus(`❌ Error: ${error.message}`, 'error');
  } finally {
    btnRun.disabled = false;
  }
}

// ============================================================================
// PROCESAMIENTO
// ============================================================================

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function preprocessImage(img) {
  let tensor = tf.browser.fromPixels(img);
  tensor = tf.image.resizeBilinear(tensor, [CONFIG.IMG_SIZE, CONFIG.IMG_SIZE]);
  tensor = tensor.toFloat().div(255.0);
  tensor = tensor.expandDims(0);
  return tensor;
}

function processOutput(data, imgWidth, imgHeight) {
  const detections = [];
  const numClasses = CONFIG.CLASSES.length;
  const stride = 4 + 1 + numClasses;
  const numBoxes = Math.floor(data.length / stride);
  
  for (let i = 0; i < numBoxes; i++) {
    const offset = i * stride;
    
    const x = data[offset + 0];
    const y = data[offset + 1];
    const w = data[offset + 2];
    const h = data[offset + 3];
    const objectness = data[offset + 4];
    
    let maxClassProb = 0;
    let maxClassIdx = 0;
    
    for (let c = 0; c < numClasses; c++) {
      const classProb = data[offset + 5 + c];
      if (classProb > maxClassProb) {
        maxClassProb = classProb;
        maxClassIdx = c;
      }
    }
    
    const confidence = objectness * maxClassProb;
    
    if (confidence > CONFIG.CONF_THRESHOLD) {
      const x1 = (x - w / 2) * imgWidth;
      const y1 = (y - h / 2) * imgHeight;
      const x2 = (x + w / 2) * imgWidth;
      const y2 = (y + h / 2) * imgHeight;
      
      if (x1 >= 0 && y1 >= 0 && x2 <= imgWidth && y2 <= imgHeight) {
        detections.push({
          bbox: [x1, y1, x2, y2],
          class: maxClassIdx,
          score: confidence
        });
      }
    }
  }
  
  return applyNMS(detections);
}

function applyNMS(detections) {
  if (detections.length === 0) return [];
  
  detections.sort((a, b) => b.score - a.score);
  
  const selected = [];
  const suppressed = new Set();
  
  for (let i = 0; i < detections.length; i++) {
    if (suppressed.has(i)) continue;
    
    selected.push(detections[i]);
    
    for (let j = i + 1; j < detections.length; j++) {
      if (suppressed.has(j)) continue;
      
      const iou = calculateIoU(detections[i].bbox, detections[j].bbox);
      
      if (iou > CONFIG.IOU_THRESHOLD) {
        suppressed.add(j);
      }
    }
  }
  
  return selected;
}

function calculateIoU(box1, box2) {
  const x1 = Math.max(box1[0], box2[0]);
  const y1 = Math.max(box1[1], box2[1]);
  const x2 = Math.min(box1[2], box2[2]);
  const y2 = Math.min(box1[3], box2[3]);
  
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
  const area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
  const union = area1 + area2 - intersection;
  
  return intersection / union;
}

// ============================================================================
// VISUALIZACIÓN
// ============================================================================

function drawImage(img) {
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);
}

function drawDetections(img, detections) {
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);
  
  ctx.strokeStyle = '#0000FF';
  ctx.lineWidth = 3;
  ctx.font = 'bold 20px Inter';
  
  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.bbox;
    const w = x2 - x1;
    const h = y2 - y1;
    
    // Box azul
    ctx.strokeRect(x1, y1, w, h);
    
    // Label con confianza
    const label = `${det.class} ${(det.score * 100).toFixed(0)}%`;
    const textWidth = ctx.measureText(label).width;
    
    // Fondo
    ctx.fillStyle = '#0000FF';
    ctx.fillRect(x1, y1 - 28, textWidth + 16, 28);
    
    // Texto
    ctx.fillStyle = 'white';
    ctx.fillText(label, x1 + 8, y1 - 8);
  });
}

// ============================================================================
// ESTADÍSTICAS
// ============================================================================

function updateAllStats(detections, img) {
  // Actualizar header stats
  totalObjects.textContent = detections.length;
  
  const avgConf = detections.length > 0
    ? (detections.reduce((sum, d) => sum + d.score, 0) / detections.length * 100).toFixed(1)
    : 0;
  avgConfidence.textContent = `${avgConf}%`;
  
  detectionCount.textContent = `${detections.length} objetos`;
  
  // Actualizar tabla
  updateInventoryTable(detections);
  
  // Actualizar gráficos
  updateCharts(detections);
  
  // Actualizar lista de detecciones
  updateDetectionsList(detections);
}

function updateInventoryTable(detections) {
  const counts = Array(CONFIG.CLASSES.length).fill(0);
  const confidences = Array(CONFIG.CLASSES.length).fill([]);
  
  detections.forEach(det => {
    counts[det.class]++;
    if (!confidences[det.class]) confidences[det.class] = [];
    confidences[det.class].push(det.score);
  });
  
  countsBody.innerHTML = '';
  let total = 0;
  
  CONFIG.CLASSES.forEach((name, idx) => {
    const count = counts[idx];
    total += count;
    
    const avgConf = confidences[idx] && confidences[idx].length > 0
      ? (confidences[idx].reduce((a, b) => a + b, 0) / confidences[idx].length * 100).toFixed(1)
      : 0;
    
    const status = count > 0
      ? '<span style="color: var(--success)">✓ Detectado</span>'
      : '<span style="color: #94a3b8">- Sin detectar</span>';
    
    const row = document.createElement('tr');
    row.innerHTML = `
      <td><strong>${idx}</strong></td>
      <td>${CONFIG.ICONS[idx]} ${name}</td>
      <td><strong>${count}</strong></td>
      <td><strong>${avgConf}%</strong></td>
      <td>${status}</td>
    `;
    countsBody.appendChild(row);
  });
  
  const totalRow = document.createElement('tr');
  totalRow.style.background = 'linear-gradient(135deg, #f8fafc, #e2e8f0)';
  totalRow.style.fontWeight = 'bold';
  totalRow.innerHTML = `
    <td colspan="2"><strong>TOTAL</strong></td>
    <td><strong>${total}</strong></td>
    <td colspan="2">-</td>
  `;
  countsBody.appendChild(totalRow);
}

function updateCharts(detections) {
  const counts = Array(CONFIG.CLASSES.length).fill(0);
  const confidences = Array(CONFIG.CLASSES.length).fill([]);
  
  detections.forEach(det => {
    counts[det.class]++;
    if (!confidences[det.class]) confidences[det.class] = [];
    confidences[det.class].push(det.score * 100);
  });
  
  // Pie Chart
  const pieCtx = document.getElementById('pieChart');
  if (pieChart) pieChart.destroy();
  
  pieChart = new Chart(pieCtx, {
    type: 'doughnut',
    data: {
      labels: CONFIG.CLASSES.map((c, i) => `${CONFIG.ICONS[i]} ${c}`),
      datasets: [{
        data: counts,
        backgroundColor: [
          '#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4'
        ]
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { position: 'bottom' },
        title: { display: false }
      }
    }
  });
  
  // Bar Chart
  const barCtx = document.getElementById('barChart');
  if (barChart) barChart.destroy();
  
  const avgConfs = confidences.map(arr =>
    arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0
  );
  
  barChart = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: CONFIG.CLASSES.map((c, i) => `${CONFIG.ICONS[i]} ${c}`),
      datasets: [{
        label: 'Confianza Promedio (%)',
        data: avgConfs,
        backgroundColor: '#6366f1'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 100
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function updateDetectionsList(detections) {
  detectionsList.innerHTML = '';
  
  detections.forEach((det, idx) => {
    const item = document.createElement('div');
    item.className = 'detection-item';
    item.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <strong>${CONFIG.ICONS[det.class]} ${CONFIG.CLASSES[det.class]} #${idx + 1}</strong>
          <div style="font-size: 0.85em; color: #64748b; margin-top: 5px;">
            Confianza: <strong>${(det.score * 100).toFixed(1)}%</strong>
          </div>
        </div>
        <div style="text-align: right; font-size: 0.75em; color: #94a3b8;">
          <div>x: ${det.bbox[0].toFixed(0)}, y: ${det.bbox[1].toFixed(0)}</div>
          <div>${(det.bbox[2] - det.bbox[0]).toFixed(0)}×${(det.bbox[3] - det.bbox[1]).toFixed(0)} px</div>
        </div>
      </div>
    `;
    detectionsList.appendChild(item);
  });
}

// ============================================================================
// EXPORTAR CSV
// ============================================================================

function exportToCSV() {
  if (!allDetections || allDetections.length === 0) {
    alert('No hay detecciones para exportar');
    return;
  }
  
  let csv = 'Código,Objeto,Confianza(%),X1,Y1,X2,Y2\n';
  
  allDetections.forEach(det => {
    csv += `${det.class},${CONFIG.CLASSES[det.class]},${(det.score * 100).toFixed(2)},`;
    csv += `${det.bbox[0].toFixed(0)},${det.bbox[1].toFixed(0)},`;
    csv += `${det.bbox[2].toFixed(0)},${det.bbox[3].toFixed(0)}\n`;
  });
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'inventario_detecciones.csv';
  a.click();
  window.URL.revokeObjectURL(url);
}

// Hacer función global
window.exportToCSV = exportToCSV;

// ============================================================================
// INICIAR
// ============================================================================

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}