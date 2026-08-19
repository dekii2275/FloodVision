// FloodVision Interactive Calibration & Depth Estimator Client JS

let currentImagePath = "";
let currentImage = new Image();
let canvas, ctx;

// 3 Virtual Gauges state (normalized coordinates 0.0 - 1.0)
let gauges = [
  {
    id: "G1",
    name: "Cột mốc Lề Trái",
    color: "#06b6d4",
    base: { x: 0.25, y: 0.95 },
    top: { x: 0.25, y: 0.38 },
    max_height_cm: 80.0
  },
  {
    id: "G2",
    name: "Cột mốc Lòng Đường",
    color: "#f59e0b",
    base: { x: 0.50, y: 0.98 },
    top: { x: 0.50, y: 0.35 },
    max_height_cm: 100.0
  },
  {
    id: "G3",
    name: "Cột mốc Lề Phải",
    color: "#ec4899",
    base: { x: 0.75, y: 0.95 },
    top: { x: 0.75, y: 0.38 },
    max_height_cm: 80.0
  }
];

// Dragging state
let draggedPoint = null; // { gaugeIndex, type: 'base' | 'top' | 'body', startX, startY, origBase, origTop }
const HANDLE_RADIUS = 11;

// Click-to-place state
let placingState = null; // { gaugeIndex, step: 1 | 2 }

window.addEventListener("DOMContentLoaded", () => {
  canvas = document.getElementById("gaugeCanvas");
  ctx = canvas.getContext("2d");

  initCanvasEvents();
  initControls();
  initSliders();
  fetchSampleImages();
});

function initControls() {
  // Presets
  document.getElementById("btnPresetDefault").addEventListener("click", () => {
    applyPreset([
      { base: { x: 0.25, y: 0.95 }, top: { x: 0.25, y: 0.38 } },
      { base: { x: 0.50, y: 0.98 }, top: { x: 0.50, y: 0.35 } },
      { base: { x: 0.75, y: 0.95 }, top: { x: 0.75, y: 0.38 } }
    ]);
  });

  document.getElementById("btnPresetNarrow").addEventListener("click", () => {
    applyPreset([
      { base: { x: 0.35, y: 0.95 }, top: { x: 0.35, y: 0.40 } },
      { base: { x: 0.50, y: 0.98 }, top: { x: 0.50, y: 0.38 } },
      { base: { x: 0.65, y: 0.95 }, top: { x: 0.65, y: 0.40 } }
    ]);
  });

  document.getElementById("btnPresetPerspective").addEventListener("click", () => {
    applyPreset([
      { base: { x: 0.15, y: 0.96 }, top: { x: 0.15, y: 0.30 } },
      { base: { x: 0.50, y: 0.92 }, top: { x: 0.50, y: 0.42 } },
      { base: { x: 0.85, y: 0.88 }, top: { x: 0.85, y: 0.48 } }
    ]);
  });

  // Quick place buttons
  document.querySelectorAll(".btn-place").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const gIdx = parseInt(e.currentTarget.getAttribute("data-gauge"));
      startPlacingMode(gIdx);
    });
  });

  document.getElementById("btnCancelPlace").addEventListener("click", cancelPlacingMode);

  // File Upload
  const fileInput = document.getElementById("fileUploadInput");
  document.getElementById("btnUploadTrigger").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileUpload);

  // Sample Image Selector
  document.getElementById("sampleSelect").addEventListener("change", (e) => {
    if (e.target.value) {
      loadImage(e.target.value);
    }
  });

  // Threshold slider
  const thSlider = document.getElementById("thresholdSlider");
  const thText = document.getElementById("thresholdValText");
  if (thSlider && thText) {
    thSlider.addEventListener("input", (e) => {
      thText.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  // Analysis Button
  document.getElementById("btnAnalyze").addEventListener("click", runAnalysis);

  // Modal zoom
  const modal = document.getElementById("imageModal");
  modal.addEventListener("click", () => modal.style.display = "none");
}

function applyPreset(presetCoords) {
  for (let i = 0; i < 3; i++) {
    gauges[i].base = { ...presetCoords[i].base };
    gauges[i].top  = { ...presetCoords[i].top };
  }
  syncSlidersFromGauges();
  renderCanvas();
}

function startPlacingMode(gaugeIndex) {
  placingState = { gaugeIndex, step: 1 };
  document.querySelectorAll(".btn-place").forEach((b, idx) => {
    b.classList.toggle("active", idx === gaugeIndex);
  });
  document.getElementById("btnCancelPlace").style.display = "inline-flex";
  document.getElementById("canvasHint").innerHTML = `
    🎯 <strong>Đang đặt mốc ${gauges[gaugeIndex].id}:</strong> Bước 1 - Hãy click chuột vào điểm <strong>CHÂN MỐC (0 cm - Mặt đường)</strong> trên ảnh.
  `;
}

function cancelPlacingMode() {
  placingState = null;
  document.querySelectorAll(".btn-place").forEach((b) => b.classList.remove("active"));
  document.getElementById("btnCancelPlace").style.display = "none";
  document.getElementById("canvasHint").innerHTML = `
    💡 <strong>Hướng dẫn:</strong> Bấm và kéo thân thước để di chuyển cả cột mốc, hoặc kéo 2 đầu mút <strong>Base (0 cm - Chân mốc)</strong> và <strong>Top (Đỉnh thước)</strong>.
  `;
}

function initSliders() {
  const mapInputs = [
    { prefix: "g1", index: 0 },
    { prefix: "g2", index: 1 },
    { prefix: "g3", index: 2 },
  ];

  mapInputs.forEach(({ prefix, index }) => {
    const xSlider = document.getElementById(`${prefix}_x_slider`);
    const xNum = document.getElementById(`${prefix}_x_num`);
    const bySlider = document.getElementById(`${prefix}_by_slider`);
    const byNum = document.getElementById(`${prefix}_by_num`);
    const tySlider = document.getElementById(`${prefix}_ty_slider`);
    const tyNum = document.getElementById(`${prefix}_ty_num`);
    const maxCm = document.getElementById(`${prefix}_max_cm`);

    // X Sync
    xSlider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      xNum.value = val;
      gauges[index].base.x = val / 100.0;
      gauges[index].top.x = val / 100.0;
      renderCanvas();
    });
    xNum.addEventListener("input", (e) => {
      const val = Math.max(0, Math.min(100, parseFloat(e.target.value) || 0));
      xSlider.value = val;
      gauges[index].base.x = val / 100.0;
      gauges[index].top.x = val / 100.0;
      renderCanvas();
    });

    // Base Y Sync
    bySlider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      byNum.value = val;
      gauges[index].base.y = val / 100.0;
      renderCanvas();
    });
    byNum.addEventListener("input", (e) => {
      const val = Math.max(0, Math.min(100, parseFloat(e.target.value) || 0));
      bySlider.value = val;
      gauges[index].base.y = val / 100.0;
      renderCanvas();
    });

    // Top Y Sync
    tySlider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      tyNum.value = val;
      gauges[index].top.y = val / 100.0;
      renderCanvas();
    });
    tyNum.addEventListener("input", (e) => {
      const val = Math.max(0, Math.min(100, parseFloat(e.target.value) || 0));
      tySlider.value = val;
      gauges[index].top.y = val / 100.0;
      renderCanvas();
    });

    // Max height cm
    maxCm.addEventListener("input", (e) => {
      gauges[index].max_height_cm = parseFloat(e.target.value) || 80.0;
    });
  });
}

function syncSlidersFromGauges() {
  const prefixes = ["g1", "g2", "g3"];
  prefixes.forEach((p, idx) => {
    const g = gauges[idx];
    const xPct = Math.round(g.base.x * 100);
    const byPct = Math.round(g.base.y * 100);
    const tyPct = Math.round(g.top.y * 100);

    document.getElementById(`${p}_x_slider`).value = xPct;
    document.getElementById(`${p}_x_num`).value = xPct;
    document.getElementById(`${p}_by_slider`).value = byPct;
    document.getElementById(`${p}_by_num`).value = byPct;
    document.getElementById(`${p}_ty_slider`).value = tyPct;
    document.getElementById(`${p}_ty_num`).value = tyPct;
    document.getElementById(`${p}_max_cm`).value = g.max_height_cm;
  });
}

async function fetchSampleImages() {
  try {
    const res = await fetch("/api/sample_images");
    const data = await res.json();
    const select = document.getElementById("sampleSelect");
    select.innerHTML = "";

    data.samples.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.rel_path;
      opt.textContent = `[${s.category}] ${s.name}`;
      select.appendChild(opt);
    });

    if (data.samples.length > 0) {
      const defaultImg = data.samples.find(s => s.name.includes("ngapvua")) || data.samples[0];
      select.value = defaultImg.rel_path;
      loadImage(defaultImg.rel_path);
    }
  } catch (err) {
    console.error("Lỗi nạp ảnh mẫu:", err);
  }
}

function loadImage(relPath) {
  currentImagePath = relPath;
  currentImage = new Image();
  currentImage.src = `/api/image_raw?path=${encodeURIComponent(relPath)}`;
  currentImage.onload = () => {
    canvas.width = currentImage.naturalWidth || 800;
    canvas.height = currentImage.naturalHeight || 600;
    syncSlidersFromGauges();
    renderCanvas();
  };
}

async function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload_image", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    loadImage(data.image_path);
  } catch (err) {
    alert("Tải ảnh thất bại: " + err.message);
  }
}

function renderCanvas() {
  if (!ctx || !currentImage.complete) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

  // Draw 3 Gauges
  gauges.forEach((g) => {
    const bx = g.base.x * canvas.width;
    const by = g.base.y * canvas.height;
    const tx = g.top.x * canvas.width;
    const ty = g.top.y * canvas.height;

    // Draw main gauge line with shadow glow
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(tx, ty);
    ctx.strokeStyle = g.color;
    ctx.lineWidth = 5;
    ctx.shadowColor = g.color;
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw ruler tick marks
    const ticks = 6;
    for (let t = 0; t <= ticks; t++) {
      const px = bx + (tx - bx) * (t / ticks);
      const py = by + (ty - by) * (t / ticks);
      ctx.beginPath();
      ctx.moveTo(px - 8, py);
      ctx.lineTo(px + 8, py);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Draw Base handle (Green circle)
    drawHandle(bx, by, "#10b981", "0 cm");

    // Draw Top handle (Gauge color circle)
    drawHandle(tx, ty, g.color, `${g.id} (Đỉnh)`);

    // Draw label
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 13px Inter, sans-serif";
    ctx.shadowColor = "#000000";
    ctx.shadowBlur = 6;
    ctx.fillText(`${g.name} (${g.max_height_cm}cm)`, bx - 40, ty - 16);
    ctx.shadowBlur = 0;
  });
}

function drawHandle(x, y, color, text) {
  ctx.beginPath();
  ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#ffffff";
  ctx.stroke();

  if (text) {
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 11px Inter, sans-serif";
    ctx.shadowColor = "#000000";
    ctx.shadowBlur = 4;
    ctx.fillText(text, x + 14, y + 4);
    ctx.shadowBlur = 0;
  }
}

function distToSegment(px, py, x1, y1, x2, y2) {
  const l2 = (x2 - x1)**2 + (y2 - y1)**2;
  if (l2 === 0) return Math.hypot(px - x1, py - y1);
  let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (x1 + t * (x2 - x1)), py - (y1 + t * (y2 - y1)));
}

function initCanvasEvents() {
  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  canvas.addEventListener("mousedown", (e) => {
    const { x, y } = getCanvasCoords(e);
    const normX = Math.max(0, Math.min(1, x / canvas.width));
    const normY = Math.max(0, Math.min(1, y / canvas.height));

    // Handle Click-to-Place Mode
    if (placingState) {
      const g = gauges[placingState.gaugeIndex];
      if (placingState.step === 1) {
        // Set Base Point
        g.base.x = normX;
        g.base.y = normY;
        g.top.x = normX; // Keep vertical initially
        placingState.step = 2;
        document.getElementById("canvasHint").innerHTML = `
          🎯 <strong>Đang đặt mốc ${g.id}:</strong> Bước 2 - Hãy click vào điểm <strong>ĐỈNH THƯỚC (Chiều cao tối đa)</strong> trên ảnh.
        `;
      } else {
        // Set Top Point
        g.top.x = normX;
        g.top.y = normY;
        cancelPlacingMode();
      }
      syncSlidersFromGauges();
      renderCanvas();
      return;
    }

    // Normal Dragging Check
    // 1. Check Base and Top handles first
    for (let i = 0; i < gauges.length; i++) {
      const g = gauges[i];
      const bx = g.base.x * canvas.width;
      const by = g.base.y * canvas.height;
      const tx = g.top.x * canvas.width;
      const ty = g.top.y * canvas.height;

      if (Math.hypot(x - bx, y - by) <= HANDLE_RADIUS * 2) {
        draggedPoint = { gaugeIndex: i, type: "base" };
        return;
      }
      if (Math.hypot(x - tx, y - ty) <= HANDLE_RADIUS * 2) {
        draggedPoint = { gaugeIndex: i, type: "top" };
        return;
      }
    }

    // 2. Check if clicked on the body line to move the entire gauge
    for (let i = 0; i < gauges.length; i++) {
      const g = gauges[i];
      const bx = g.base.x * canvas.width;
      const by = g.base.y * canvas.height;
      const tx = g.top.x * canvas.width;
      const ty = g.top.y * canvas.height;

      if (distToSegment(x, y, bx, by, tx, ty) <= 12) {
        draggedPoint = {
          gaugeIndex: i,
          type: "body",
          startX: normX,
          startY: normY,
          origBase: { ...g.base },
          origTop: { ...g.top }
        };
        return;
      }
    }
  });

  window.addEventListener("mousemove", (e) => {
    if (!draggedPoint) return;
    const { x, y } = getCanvasCoords(e);
    const normX = Math.max(0, Math.min(1, x / canvas.width));
    const normY = Math.max(0, Math.min(1, y / canvas.height));

    const g = gauges[draggedPoint.gaugeIndex];

    if (draggedPoint.type === "base") {
      g.base.x = normX;
      g.base.y = normY;
    } else if (draggedPoint.type === "top") {
      g.top.x = normX;
      g.top.y = normY;
    } else if (draggedPoint.type === "body") {
      const dx = normX - draggedPoint.startX;
      const dy = normY - draggedPoint.startY;
      g.base.x = Math.max(0, Math.min(1, draggedPoint.origBase.x + dx));
      g.base.y = Math.max(0, Math.min(1, draggedPoint.origBase.y + dy));
      g.top.x  = Math.max(0, Math.min(1, draggedPoint.origTop.x + dx));
      g.top.y  = Math.max(0, Math.min(1, draggedPoint.origTop.y + dy));
    }

    syncSlidersFromGauges();
    renderCanvas();
  });

  window.addEventListener("mouseup", () => {
    draggedPoint = null;
  });
}

async function runAnalysis() {
  if (!currentImagePath) {
    alert("Vui lòng chọn ảnh trước!");
    return;
  }

  const btn = document.getElementById("btnAnalyze");
  btn.disabled = true;
  btn.innerHTML = `<span>⏳ ĐANG XỬ LÝ SEGMENTATION & THẨM ĐỊNH MỰC NƯỚC...</span>`;

  // Read latest max heights
  gauges[0].max_height_cm = parseFloat(document.getElementById("g1_max_cm").value) || 80.0;
  gauges[1].max_height_cm = parseFloat(document.getElementById("g2_max_cm").value) || 100.0;
  gauges[2].max_height_cm = parseFloat(document.getElementById("g3_max_cm").value) || 80.0;

  const payload = {
    image_path: currentImagePath,
    gauges: gauges.map((g) => ({
      id: g.id,
      name: g.name,
      base_x: g.base.x,
      base_y: g.base.y,
      top_x: g.top.x,
      top_y: g.top.y,
      max_height_cm: g.max_height_cm,
    })),
    threshold: parseFloat(document.getElementById("thresholdSlider")?.value) || 0.30,
  };

  const startTime = performance.now();

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const result = await res.json();
    const elapsed = Math.round(performance.now() - startTime);

    document.getElementById("processTime").textContent = `Thời gian: ${elapsed} ms`;
    renderResults(result);
  } catch (err) {
    alert("Lỗi phân tích: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>⚡ XỬ LÝ SEGMENTATION & ƯỚC LƯỢNG ĐỘ SÂU (CLIPSEG + 3 MỐC)</span>`;
  }
}

function renderResults(res) {
  const level = res.level;

  // Level banner
  const banner = document.getElementById("levelBanner");
  banner.style.borderColor = level.badge_color;
  banner.style.background = `linear-gradient(135deg, ${level.badge_color}33, rgba(15, 23, 42, 0.9))`;

  const badge = document.getElementById("levelBadge");
  badge.textContent = level.level_name.toUpperCase();
  badge.style.color = level.badge_color;

  document.getElementById("fusedDepthText").textContent = `${res.fused_depth_cm.toFixed(1)} cm`;

  // Metrics
  document.getElementById("metricLevelCode").textContent = level.level_code;
  document.getElementById("metricLevelCode").style.color = level.badge_color;
  document.getElementById("metricFused").textContent = `${res.fused_depth_cm.toFixed(1)} cm`;
  document.getElementById("metricArea").textContent = `${res.flood_area_pct.toFixed(1)} %`;

  // Gauges table
  const tbody = document.getElementById("gaugeTableBody");
  tbody.innerHTML = "";
  res.gauges.forEach((g) => {
    const tr = document.createElement("tr");

    let statusBadge = "";
    if (g.status === "VALID") {
      statusBadge = `<span style="color: #10b981; font-weight: 700;">✅ VALID</span>`;
    } else if (g.status === "OUTLIER") {
      statusBadge = `<span style="color: #ef4444; font-weight: 700;">❌ OUTLIER</span>`;
    } else if (g.status === "NO_WATER_AT_BASE") {
      statusBadge = `<span style="color: #94a3b8; font-weight: 600;">⚪ KHÔ RÁO</span>`;
    } else {
      statusBadge = `<span style="color: #f59e0b; font-weight: 600;">⚠️ ${g.status}</span>`;
    }

    const raysTxt = `${g.valid_rays}/${g.total_rays}`;
    const calibTxt = g.calibration_mode === "multi_point" ? "Multi-Point" : "Linear";

    tr.innerHTML = `
      <td><strong>${g.id}</strong> (${g.name})</td>
      <td>${statusBadge}</td>
      <td><strong style="color: #38bdf8; font-size: 1rem;">${g.depth_cm.toFixed(1)} cm</strong></td>
      <td><code style="color: var(--text-secondary);">${raysTxt}</code></td>
      <td><span style="font-size: 0.8rem; color: var(--text-secondary);">${calibTxt}</span></td>
    `;
    tbody.appendChild(tr);
  });

  // Fusion Summary Bar
  const fusionBox = document.getElementById("fusionSummaryBox");
  if (fusionBox && res.fusion) {
    fusionBox.style.display = "block";
    const usedStr = res.fusion.used_gauges && res.fusion.used_gauges.length > 0
      ? res.fusion.used_gauges.join(", ")
      : "Không có";
    document.getElementById("fusionUsedText").textContent = `Dùng: ${usedStr} (Chất lượng: ${res.fusion.fusion_quality})`;

    const rejs = res.fusion.rejected_gauges || [];
    if (rejs.length > 0) {
      document.getElementById("fusionRejectedText").textContent = `⚠️ Loại bỏ ngoại lai: ${rejs.join(", ")}`;
    } else {
      document.getElementById("fusionRejectedText").textContent = "";
    }
  }

  // Dashboard preview
  const previewDiv = document.getElementById("dashboardPreview");
  previewDiv.innerHTML = `
    <img src="${res.dashboard_url}" alt="Dashboard Preview" title="Nhấp để phóng to ảnh Dashboard">
  `;
  previewDiv.onclick = () => {
    const modal = document.getElementById("imageModal");
    document.getElementById("modalImg").src = res.dashboard_url;
    modal.style.display = "flex";
  };
}
