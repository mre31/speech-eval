// Turkish Speech Evaluation Frontend Application

let mediaRecorder = null;
let audioChunks = [];
let audioBlob = null;
let audioContext = null;
let analyser = null;
let animationId = null;

let currentEvaluationData = null;
let selectedWordIndex = null;

document.addEventListener('DOMContentLoaded', () => {
  initPresets();
  initRecording();
  initFileUpload();
  initEvaluateButton();
});

// Load and render practice sentence presets
async function initPresets() {
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    const container = document.getElementById('presetChips');
    container.innerHTML = '';

    data.presets.forEach((preset, index) => {
      const chip = document.createElement('div');
      chip.className = 'preset-chip';
      chip.textContent = preset.title;
      chip.title = preset.text;
      chip.addEventListener('click', () => {
        document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        document.getElementById('targetText').value = preset.text;
      });
      container.appendChild(chip);
    });
  } catch (err) {
    console.error('Presets could not be loaded:', err);
  }
}

// Microphone recording & live waveform visualizer
function initRecording() {
  const recordBtn = document.getElementById('recordBtn');
  const recordText = document.getElementById('recordStatus');
  const canvas = document.getElementById('waveformCanvas');
  const ctx = canvas.getContext('2d');
  const player = document.getElementById('audioPlayer');
  const previewBox = document.getElementById('audioPreview');

  let isRecording = false;

  recordBtn.addEventListener('click', async () => {
    if (!isRecording) {
      // Start Recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        // Audio Context for Visualizer
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        canvas.style.display = 'block';
        drawWaveform(analyser, ctx, canvas);

        mediaRecorder.ondataavailable = e => {
          if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
          const mimeType = mediaRecorder.mimeType || 'audio/webm';
          audioBlob = new Blob(audioChunks, { type: mimeType });
          const audioUrl = URL.createObjectURL(audioBlob);
          player.src = audioUrl;
          previewBox.style.display = 'block';
          document.getElementById('evalBtn').disabled = false;

          // Stop visualizer
          if (animationId) cancelAnimationFrame(animationId);
          canvas.style.display = 'none';
          stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;
        recordBtn.classList.add('recording');
        recordText.textContent = 'Kaydı Durdurmak İçin Tıkla';
      } catch (err) {
        alert('Mikrofon erişimi sağlanamadı: ' + err.message);
      }
    } else {
      // Stop Recording
      mediaRecorder.stop();
      isRecording = false;
      recordBtn.classList.remove('recording');
      recordText.textContent = 'Kaydı Başlatmak İçin Tıkla';
    }
  });
}

function drawWaveform(analyser, ctx, canvas) {
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function render() {
    animationId = requestAnimationFrame(render);
    analyser.getByteTimeDomainData(dataArray);

    ctx.fillStyle = 'rgba(10, 14, 23, 0.4)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#6366f1';
    ctx.beginPath();

    const sliceWidth = canvas.width * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = v * canvas.height / 2;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);

      x += sliceWidth;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }
  render();
}

// File Upload
function initFileUpload() {
  const dropZone = document.getElementById('uploadDropZone');
  const fileInput = document.getElementById('fileInput');
  const player = document.getElementById('audioPlayer');
  const previewBox = document.getElementById('audioPreview');

  dropZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.style.borderColor = '#6366f1';
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'var(--border-subtle)';
  });

  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border-subtle)';
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  function handleFile(file) {
    audioBlob = file;
    player.src = URL.createObjectURL(file);
    previewBox.style.display = 'block';
    document.getElementById('uploadFileName').textContent = file.name;
    document.getElementById('evalBtn').disabled = false;
  }
}

// Evaluate submit handler
function initEvaluateButton() {
  const evalBtn = document.getElementById('evalBtn');
  const loadingBox = document.getElementById('loadingBox');
  const resultsSection = document.getElementById('resultsSection');

  evalBtn.addEventListener('click', async () => {
    if (!audioBlob) {
      alert('Lütfen önce bir ses kaydedin veya yükleyin.');
      return;
    }

    evalBtn.disabled = true;
    loadingBox.style.display = 'block';
    resultsSection.style.display = 'none';

    const formData = new FormData();
    const isFile = (audioBlob instanceof File);
    const filename = isFile ? audioBlob.name : ((mediaRecorder && mediaRecorder.mimeType && mediaRecorder.mimeType.includes('webm')) ? 'speech.webm' : 'speech.wav');
    formData.append('file', audioBlob, filename);
    const targetText = document.getElementById('targetText').value.trim();
    if (targetText) {
      formData.append('target_text', targetText);
    }

    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Değerlendirme başarısız oldu.');
      }

      const data = await response.json();
      currentEvaluationData = data;
      renderResults(data);
    } catch (err) {
      alert('Hata: ' + err.message);
    } finally {
      loadingBox.style.display = 'none';
      evalBtn.disabled = false;
    }
  });
}

// Render Results View
function renderResults(data) {
  const resultsSection = document.getElementById('resultsSection');
  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth' });

  // 1. Overall Score & Circular Gauge Animation
  const scoreNumEl = document.getElementById('overallScoreNumber');
  const circleProgress = document.getElementById('circleProgress');
  const statusBadge = document.getElementById('statusBadge');
  const heroDesc = document.getElementById('heroDescription');

  const score = data.overall_score;
  scoreNumEl.textContent = Math.round(score);

  // SVG dashoffset: 440 is 0%, 0 is 100%
  const offset = 440 - (440 * (score / 100));
  circleProgress.style.strokeDashoffset = offset;

  if (score >= 80) {
    circleProgress.style.stroke = '#10b981';
    statusBadge.className = 'status-badge badge-excellent';
    statusBadge.textContent = 'Standart Türkçe Telaffuz';
    heroDesc.textContent = 'Konuşmanız standart İstanbul Türkçesi fonetik kurallarına ve doğal ses dağılımına çok yakın.';
  } else if (score >= 65) {
    circleProgress.style.stroke = '#f59e0b';
    statusBadge.className = 'status-badge badge-good';
    statusBadge.textContent = 'Hafif Şive / Bölgesel Telaffuz Sapması';
    heroDesc.textContent = 'Genel olarak anlaşılır, ancak bazı ünlü boylarında veya ünsüz boğumlanma yerlerinde standarttan sapmalar mevcut.';
  } else {
    circleProgress.style.stroke = '#ef4444';
    statusBadge.className = 'status-badge badge-needs-work';
    statusBadge.textContent = 'Belirgin Şive / Fonetik Ayrışma';
    heroDesc.textContent = 'Standart Türkçe referanslarından belirgin fonetik ve akustik ayrışmalar tespit edildi.';
  }

  // 2. Subscore Cards
  updateSubscore('subscorePronunciation', 'progressPronunciation', data.subscores.pronunciation);
  updateSubscore('subscoreVowels', 'progressVowels', data.subscores.vowels);
  updateSubscore('subscoreConsonants', 'progressConsonants', data.subscores.consonants);
  updateSubscore('subscoreRhythm', 'progressRhythm', data.subscores.rhythm);
  updateSubscore('subscoreIntonation', 'progressIntonation', data.subscores.intonation);

  // 3. Problematic Phonemes
  const probBox = document.getElementById('problematicBox');
  const probList = document.getElementById('problematicList');
  probList.innerHTML = '';

  if (data.problematic_phonemes && data.problematic_phonemes.length > 0) {
    probBox.style.display = 'block';
    data.problematic_phonemes.forEach(p => {
      const li = document.createElement('li');
      li.className = 'alert-item';
      li.innerHTML = `
        <span class="phoneme-pill">${p.phoneme}</span>
        <strong>Skor: ${p.score}</strong>
        <span>— ${p.diagnostic}</span>
      `;
      probList.appendChild(li);
    });
  } else {
    probBox.style.display = 'none';
  }

  // 4. Word Chips
  renderWordChips(data.word_scores);

  // 5. Phonemes Table
  renderPhonemesTable(data.word_scores, null);
}

function updateSubscore(valId, barId, score) {
  document.getElementById(valId).textContent = Math.round(score);
  const bar = document.getElementById(barId);
  bar.style.width = score + '%';
  if (score >= 80) bar.style.background = 'var(--success)';
  else if (score >= 65) bar.style.background = 'var(--warning)';
  else bar.style.background = 'var(--danger)';
}

function renderWordChips(words) {
  const container = document.getElementById('wordChips');
  container.innerHTML = '';

  // "Tümünü Göster" Chip
  const allChip = document.createElement('div');
  allChip.className = 'word-card active';
  allChip.innerHTML = `<span class="word-name">Tümü</span>`;
  allChip.addEventListener('click', () => {
    document.querySelectorAll('.word-card').forEach(w => w.classList.remove('active'));
    allChip.classList.add('active');
    selectedWordIndex = null;
    renderPhonemesTable(words, null);
  });
  container.appendChild(allChip);

  words.forEach(w => {
    const chip = document.createElement('div');
    chip.className = 'word-card';

    let badgeClass = 'badge-excellent';
    if (w.score < 65) badgeClass = 'badge-needs-work';
    else if (w.score < 80) badgeClass = 'badge-good';

    chip.innerHTML = `
      <span class="word-name">${w.word}</span>
      <span class="status-badge ${badgeClass}">${Math.round(w.score)}</span>
    `;

    chip.addEventListener('click', () => {
      document.querySelectorAll('.word-card').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      selectedWordIndex = w.word_index;
      renderPhonemesTable(words, w.word_index);
    });

    container.appendChild(chip);
  });
}

function renderPhonemesTable(words, filterWordIndex) {
  const tbody = document.getElementById('phonemesTableBody');
  tbody.innerHTML = '';

  words.forEach(w => {
    if (filterWordIndex !== null && w.word_index !== filterWordIndex) {
      return;
    }

    w.phonemes.forEach(p => {
      const row = document.createElement('tr');

      let scoreColor = 'var(--success)';
      if (p.score < 65) scoreColor = 'var(--danger)';
      else if (p.score < 80) scoreColor = 'var(--warning)';

      const isWarning = p.score < 75;

      row.innerHTML = `
        <td><span class="phoneme-pill">${p.key}</span></td>
        <td><strong>${p.grapheme}</strong> <span style="color:var(--text-dim); font-size:0.8rem">(${w.word})</span></td>
        <td><strong style="color: ${scoreColor}">${p.score}</strong></td>
        <td>${p.embedding_score}</td>
        <td>${p.acoustic_score}</td>
        <td>${Math.round(p.duration * 1000)} ms</td>
        <td class="diagnostic-text ${isWarning ? 'diagnostic-warning' : ''}">${p.diagnostic}</td>
      `;
      tbody.appendChild(row);
    });
  });
}
