const MIN_WAVELENGTH = 380;
const MAX_WAVELENGTH = 780;
const DEFAULTS = {
  mode: "single",
  singleLambda: 550,
  polyMin: 420,
  polyMax: 680,
  polySamples: 9,
  resolution: 256,
  preset: "round",
};

const ui = {
  mode: document.getElementById("mode"),
  singleLambda: document.getElementById("single-lambda"),
  polyMin: document.getElementById("poly-min"),
  polyMax: document.getElementById("poly-max"),
  polySamples: document.getElementById("poly-samples"),
  resolution: document.getElementById("resolution"),
  preset: document.getElementById("preset"),
  loadPreset: document.getElementById("load-preset"),
  clearMask: document.getElementById("clear-mask"),
  resetApp: document.getElementById("reset-app"),
  status: document.getElementById("status"),
  maskCanvas: document.getElementById("mask-canvas"),
  patternCanvas: document.getElementById("pattern-canvas"),
};

const maskCtx = ui.maskCanvas.getContext("2d", { willReadFrequently: true });
const patternCtx = ui.patternCanvas.getContext("2d");

const runtime = {
  state: createApertureState(DEFAULTS.resolution),
  isDrawing: false,
  drawValue: 1,
  computePending: false,
  lastComputeTime: 0,
  minComputeIntervalMs: 120,
  resolutionChangeInProgress: false,
};

applyPreset(runtime.state, DEFAULTS.preset);
setupUi();
renderMask();
scheduleCompute();

function setupUi() {
  ui.mode.value = DEFAULTS.mode;
  ui.singleLambda.value = String(DEFAULTS.singleLambda);
  ui.polyMin.value = String(DEFAULTS.polyMin);
  ui.polyMax.value = String(DEFAULTS.polyMax);
  ui.polySamples.value = String(DEFAULTS.polySamples);
  ui.resolution.value = String(DEFAULTS.resolution);
  ui.preset.value = DEFAULTS.preset;
  updateModeVisibility();

  ui.mode.addEventListener("change", () => {
    updateModeVisibility();
    scheduleCompute();
  });
  ui.singleLambda.addEventListener("input", scheduleCompute);
  ui.polyMin.addEventListener("input", scheduleCompute);
  ui.polyMax.addEventListener("input", scheduleCompute);
  ui.polySamples.addEventListener("input", scheduleCompute);
  ui.resolution.addEventListener("change", handleResolutionChange);
  ui.loadPreset.addEventListener("click", () => {
    applyPreset(runtime.state, ui.preset.value);
    renderMask();
    scheduleCompute();
  });
  ui.clearMask.addEventListener("click", () => {
    runtime.state.mask.fill(0);
    renderMask();
    scheduleCompute();
  });
  ui.resetApp.addEventListener("click", resetApp);

  ui.maskCanvas.addEventListener("pointerdown", (event) => {
    runtime.isDrawing = true;
    runtime.drawValue = event.button === 2 || event.shiftKey ? 0 : 1;
    drawFromPointer(event);
  });
  ui.maskCanvas.addEventListener("pointermove", (event) => {
    if (runtime.isDrawing) {
      drawFromPointer(event);
    }
  });
  const stopDrawing = () => {
    runtime.isDrawing = false;
  };
  ui.maskCanvas.addEventListener("pointerup", stopDrawing);
  ui.maskCanvas.addEventListener("pointerleave", stopDrawing);
  ui.maskCanvas.addEventListener("pointercancel", stopDrawing);
  ui.maskCanvas.addEventListener("contextmenu", (event) => event.preventDefault());
}

function updateModeVisibility() {
  const singleVisible = ui.mode.value === "single";
  document.querySelectorAll(".mode-single").forEach((el) => {
    el.style.display = singleVisible ? "" : "none";
  });
  document.querySelectorAll(".mode-poly").forEach((el) => {
    el.style.display = singleVisible ? "none" : "";
  });
}

function resetApp() {
  ui.mode.value = DEFAULTS.mode;
  ui.singleLambda.value = String(DEFAULTS.singleLambda);
  ui.polyMin.value = String(DEFAULTS.polyMin);
  ui.polyMax.value = String(DEFAULTS.polyMax);
  ui.polySamples.value = String(DEFAULTS.polySamples);
  ui.resolution.value = String(DEFAULTS.resolution);
  ui.preset.value = DEFAULTS.preset;
  runtime.state = createApertureState(DEFAULTS.resolution);
  applyPreset(runtime.state, DEFAULTS.preset);
  updateModeVisibility();
  renderMask();
  scheduleCompute();
}

function drawFromPointer(event) {
  const rect = ui.maskCanvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * runtime.state.resolution;
  const y = ((event.clientY - rect.top) / rect.height) * runtime.state.resolution;
  paintDisc(runtime.state, x, y, Math.max(1, Math.floor(runtime.state.resolution / 64)), runtime.drawValue);
  renderMask();
  scheduleCompute();
}

function paintDisc(state, x, y, radius, value) {
  const n = state.resolution;
  const mask = state.mask;
  const cx = Math.floor(x);
  const cy = Math.floor(y);
  const r2 = radius * radius;
  for (let py = Math.max(0, cy - radius); py <= Math.min(n - 1, cy + radius); py += 1) {
    for (let px = Math.max(0, cx - radius); px <= Math.min(n - 1, cx + radius); px += 1) {
      const dx = px - x;
      const dy = py - y;
      if (dx * dx + dy * dy <= r2) {
        mask[py * n + px] = value;
      }
    }
  }
}

function handleResolutionChange() {
  if (runtime.resolutionChangeInProgress) {
    return;
  }
  runtime.resolutionChangeInProgress = true;
  const nextResolution = clampInt(Number(ui.resolution.value), 128, 512);
  const resized = resizeApertureState(runtime.state, nextResolution);
  runtime.state = resized;
  renderMask();
  scheduleCompute();
  runtime.resolutionChangeInProgress = false;
}

function renderMask() {
  const n = runtime.state.resolution;
  const image = maskCtx.createImageData(n, n);
  for (let i = 0; i < runtime.state.mask.length; i += 1) {
    const v = runtime.state.mask[i] > 0.5 ? 255 : 0;
    const idx = i * 4;
    image.data[idx] = v;
    image.data[idx + 1] = v;
    image.data[idx + 2] = v;
    image.data[idx + 3] = 255;
  }
  const offscreen = createScratchCanvas(n);
  const ctx = offscreen.getContext("2d");
  ctx.putImageData(image, 0, 0);
  maskCtx.clearRect(0, 0, ui.maskCanvas.width, ui.maskCanvas.height);
  maskCtx.imageSmoothingEnabled = false;
  maskCtx.drawImage(offscreen, 0, 0, ui.maskCanvas.width, ui.maskCanvas.height);
}

function scheduleCompute() {
  if (runtime.computePending) {
    return;
  }
  runtime.computePending = true;
  const elapsed = performance.now() - runtime.lastComputeTime;
  const delay = Math.max(0, runtime.minComputeIntervalMs - elapsed);
  window.setTimeout(() => {
    window.requestAnimationFrame(recomputeDiffraction);
  }, delay);
}

function recomputeDiffraction() {
  runtime.computePending = false;
  runtime.lastComputeTime = performance.now();
  const validated = validateControls();
  if (!validated.ok) {
    setStatus(validated.message);
    return;
  }
  const t0 = performance.now();
  const pattern = computePattern(runtime.state, validated.settings);
  renderPattern(pattern, runtime.state.resolution);
  const dt = Math.round(performance.now() - t0);
  setStatus(`Updated in ${dt} ms at ${runtime.state.resolution}×${runtime.state.resolution}`);
}

function validateControls() {
  const mode = ui.mode.value === "poly" ? "poly" : "single";
  const singleLambda = clampInt(Number(ui.singleLambda.value), MIN_WAVELENGTH, MAX_WAVELENGTH);
  ui.singleLambda.value = String(singleLambda);
  let polyMin = clampInt(Number(ui.polyMin.value), MIN_WAVELENGTH, MAX_WAVELENGTH);
  let polyMax = clampInt(Number(ui.polyMax.value), MIN_WAVELENGTH, MAX_WAVELENGTH);
  if (polyMin > polyMax) {
    const tmp = polyMin;
    polyMin = polyMax;
    polyMax = tmp;
  }
  ui.polyMin.value = String(polyMin);
  ui.polyMax.value = String(polyMax);
  const polySamples = clampOdd(Number(ui.polySamples.value), 3, 21);
  ui.polySamples.value = String(polySamples);
  const resolution = clampInt(Number(ui.resolution.value), 128, 512);
  if (resolution !== runtime.state.resolution) {
    ui.resolution.value = String(runtime.state.resolution);
  }

  return {
    ok: true,
    settings: { mode, singleLambda, polyMin, polyMax, polySamples },
    message: "",
  };
}

function clampInt(value, min, max) {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, Math.round(value)));
}

function clampOdd(value, min, max) {
  let clamped = clampInt(value, min, max);
  if (clamped % 2 === 0) {
    clamped = Math.min(max, clamped + 1);
  }
  return clamped;
}

function setStatus(message) {
  ui.status.textContent = message;
}

function computePattern(state, settings) {
  const n = state.resolution;
  const output = new Float32Array(n * n * 3);
  if (settings.mode === "single") {
    const intensity = computeSingleWaveIntensity(state, settings.singleLambda);
    const [r, g, b] = wavelengthToRgb(settings.singleLambda);
    for (let i = 0; i < intensity.length; i += 1) {
      const base = i * 3;
      output[base] = intensity[i] * r;
      output[base + 1] = intensity[i] * g;
      output[base + 2] = intensity[i] * b;
    }
  } else {
    const lambdas = linspace(settings.polyMin, settings.polyMax, settings.polySamples);
    for (const lambda of lambdas) {
      const intensity = computeSingleWaveIntensity(state, lambda);
      const [r, g, b] = wavelengthToRgb(lambda);
      for (let i = 0; i < intensity.length; i += 1) {
        const base = i * 3;
        output[base] += intensity[i] * r;
        output[base + 1] += intensity[i] * g;
        output[base + 2] += intensity[i] * b;
      }
    }
  }
  normalizeRgbBuffer(output);
  return output;
}

function computeSingleWaveIntensity(state, lambdaNm) {
  const n = state.resolution;
  const sampled = sampleMaskForWavelength(state.mask, n, lambdaNm, DEFAULTS.singleLambda);
  const { re, im } = fft2d(sampled, n);
  const threshold = 1e-5;
  const shifted = new Float32Array(n * n);
  let maxIntensity = 0;
  for (let i = 0; i < re.length; i += 1) {
    const v = re[i] * re[i] + im[i] * im[i];
    if (v > maxIntensity) {
      maxIntensity = v;
    }
  }
  const half = n / 2;
  let maxLog = 0;
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const srcX = (x + half) % n;
      const srcY = (y + half) % n;
      const src = srcY * n + srcX;
      const raw = re[src] * re[src] + im[src] * im[src];
      const val = raw > threshold * maxIntensity ? Math.log1p(raw) : 0;
      shifted[y * n + x] = val;
      if (val > maxLog) {
        maxLog = val;
      }
    }
  }
  if (maxLog > 0) {
    for (let i = 0; i < shifted.length; i += 1) {
      shifted[i] /= maxLog;
    }
  }
  return shifted;
}

function sampleMaskForWavelength(mask, n, lambdaNm, referenceNm) {
  const sampled = new Float32Array(n * n);
  const center = (n - 1) / 2;
  const scale = Math.max(0.25, Math.min(3, lambdaNm / referenceNm));
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const srcX = Math.round(center + (x - center) * scale);
      const srcY = Math.round(center + (y - center) * scale);
      if (srcX >= 0 && srcX < n && srcY >= 0 && srcY < n) {
        sampled[y * n + x] = mask[srcY * n + srcX];
      }
    }
  }
  return sampled;
}

function fft2d(input, n) {
  const re = new Float64Array(input.length);
  const im = new Float64Array(input.length);
  re.set(input);
  const rowRe = new Float64Array(n);
  const rowIm = new Float64Array(n);
  for (let y = 0; y < n; y += 1) {
    const offset = y * n;
    for (let x = 0; x < n; x += 1) {
      rowRe[x] = re[offset + x];
      rowIm[x] = 0;
    }
    fft1d(rowRe, rowIm);
    for (let x = 0; x < n; x += 1) {
      re[offset + x] = rowRe[x];
      im[offset + x] = rowIm[x];
    }
  }

  const colRe = new Float64Array(n);
  const colIm = new Float64Array(n);
  for (let x = 0; x < n; x += 1) {
    for (let y = 0; y < n; y += 1) {
      const idx = y * n + x;
      colRe[y] = re[idx];
      colIm[y] = im[idx];
    }
    fft1d(colRe, colIm);
    for (let y = 0; y < n; y += 1) {
      const idx = y * n + x;
      re[idx] = colRe[y];
      im[idx] = colIm[y];
    }
  }
  return { re, im };
}

function fft1d(re, im) {
  const n = re.length;
  if ((n & (n - 1)) !== 0) {
    throw new Error("FFT size must be a power of 2.");
  }
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    const wLenRe = Math.cos(angle);
    const wLenIm = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let j = 0; j < len / 2; j += 1) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + len / 2] * wRe - im[i + j + len / 2] * wIm;
        const vIm = re[i + j + len / 2] * wIm + im[i + j + len / 2] * wRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe;
        im[i + j + len / 2] = uIm - vIm;
        const nextWRe = wRe * wLenRe - wIm * wLenIm;
        const nextWIm = wRe * wLenIm + wIm * wLenRe;
        wRe = nextWRe;
        wIm = nextWIm;
      }
    }
  }
}

function normalizeRgbBuffer(buffer) {
  let max = 0;
  for (let i = 0; i < buffer.length; i += 1) {
    if (buffer[i] > max) {
      max = buffer[i];
    }
  }
  if (max === 0) {
    return;
  }
  for (let i = 0; i < buffer.length; i += 1) {
    buffer[i] = Math.pow(buffer[i] / max, 0.85);
  }
}

function renderPattern(rgbBuffer, n) {
  const image = patternCtx.createImageData(n, n);
  for (let i = 0; i < n * n; i += 1) {
    const base = i * 3;
    const out = i * 4;
    image.data[out] = Math.round(Math.max(0, Math.min(1, rgbBuffer[base])) * 255);
    image.data[out + 1] = Math.round(Math.max(0, Math.min(1, rgbBuffer[base + 1])) * 255);
    image.data[out + 2] = Math.round(Math.max(0, Math.min(1, rgbBuffer[base + 2])) * 255);
    image.data[out + 3] = 255;
  }
  const offscreen = createScratchCanvas(n);
  const ctx = offscreen.getContext("2d");
  ctx.putImageData(image, 0, 0);
  patternCtx.clearRect(0, 0, ui.patternCanvas.width, ui.patternCanvas.height);
  patternCtx.imageSmoothingEnabled = false;
  patternCtx.drawImage(offscreen, 0, 0, ui.patternCanvas.width, ui.patternCanvas.height);
}

function wavelengthToRgb(wave) {
  const gamma = 0.8;
  let red = 0;
  let green = 0;
  let blue = 0;

  if (wave >= 380 && wave < 440) {
    red = -(wave - 440) / (440 - 380);
    blue = 1;
  } else if (wave < 490) {
    green = (wave - 440) / (490 - 440);
    blue = 1;
  } else if (wave < 510) {
    green = 1;
    blue = -(wave - 510) / (510 - 490);
  } else if (wave < 580) {
    red = (wave - 510) / (580 - 510);
    green = 1;
  } else if (wave < 645) {
    red = 1;
    green = -(wave - 645) / (645 - 580);
  } else if (wave <= 780) {
    red = 1;
  }

  let factor = 0;
  if (wave >= 380 && wave < 420) {
    factor = 0.3 + (0.7 * (wave - 380)) / (420 - 380);
  } else if (wave < 700) {
    factor = 1;
  } else if (wave <= 780) {
    factor = 0.3 + (0.7 * (780 - wave)) / (780 - 700);
  }

  const f = (channel) => (channel === 0 ? 0 : Math.pow(channel * factor, gamma));
  return [f(red), f(green), f(blue)];
}

function linspace(start, end, count) {
  if (count <= 1) {
    return [start];
  }

  function createScratchCanvas(size) {
    if (typeof OffscreenCanvas !== "undefined") {
      return new OffscreenCanvas(size, size);
    }
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    return canvas;
  }
  const out = [];
  const step = (end - start) / (count - 1);
  for (let i = 0; i < count; i += 1) {
    out.push(start + i * step);
  }
  return out;
}

function createApertureState(resolution) {
  return {
    version: 1,
    resolution,
    mask: new Float32Array(resolution * resolution),
  };
}

function resizeApertureState(state, nextResolution) {
  const next = createApertureState(nextResolution);
  const ratio = state.resolution / nextResolution;
  for (let y = 0; y < nextResolution; y += 1) {
    for (let x = 0; x < nextResolution; x += 1) {
      const oldX = Math.min(state.resolution - 1, Math.floor((x + 0.5) * ratio));
      const oldY = Math.min(state.resolution - 1, Math.floor((y + 0.5) * ratio));
      next.mask[y * nextResolution + x] = state.mask[oldY * state.resolution + oldX];
    }
  }
  return next;
}

function applyPreset(state, presetName) {
  state.mask.fill(0);
  if (presetName === "slits") {
    fillSlitsPreset(state);
  } else if (presetName === "jwst") {
    fillJwstPreset(state);
  } else {
    fillRoundPreset(state);
  }
}

function fillRoundPreset(state) {
  const n = state.resolution;
  const center = (n - 1) / 2;
  const radius = n * 0.32;
  const r2 = radius * radius;
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const dx = x - center;
      const dy = y - center;
      if (dx * dx + dy * dy <= r2) {
        state.mask[y * n + x] = 1;
      }
    }
  }
}

function fillSlitsPreset(state) {
  const n = state.resolution;
  const center = (n - 1) / 2;
  const slitWidth = Math.max(2, Math.floor(n * 0.018));
  const slitHeight = Math.floor(n * 0.64);
  const separation = Math.floor(n * 0.09);
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const dy = Math.abs(y - center);
      if (dy > slitHeight / 2) {
        continue;
      }
      const dx = x - center;
      const inLeft = Math.abs(dx + separation) <= slitWidth;
      const inRight = Math.abs(dx - separation) <= slitWidth;
      if (inLeft || inRight) {
        state.mask[y * n + x] = 1;
      }
    }
  }
}

function fillJwstPreset(state) {
  const n = state.resolution;
  const center = (n - 1) / 2;
  const segmentRadius = n * 0.07;
  const spacing = segmentRadius * 1.88;

  for (let q = -2; q <= 2; q += 1) {
    for (let r = -2; r <= 2; r += 1) {
      const s = -q - r;
      const ring = Math.max(Math.abs(q), Math.abs(r), Math.abs(s));
      if (ring > 2 || (q === 0 && r === 0)) {
        continue;
      }
      const px = center + spacing * Math.sqrt(3) * (q + r / 2);
      const py = center + spacing * 1.5 * r;
      paintHex(state, px, py, segmentRadius, 1);
    }
  }

  const obstruction = n * 0.095;
  const o2 = obstruction * obstruction;
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const dx = x - center;
      const dy = y - center;
      if (dx * dx + dy * dy <= o2) {
        state.mask[y * n + x] = 0;
      }
    }
  }

  carveStrut(state, center, center, 0, n * 0.011);
  carveStrut(state, center, center, Math.PI / 3, n * 0.011);
  carveStrut(state, center, center, -Math.PI / 3, n * 0.011);
}

function paintHex(state, cx, cy, radius, value) {
  const n = state.resolution;
  const minX = Math.max(0, Math.floor(cx - radius - 1));
  const maxX = Math.min(n - 1, Math.ceil(cx + radius + 1));
  const minY = Math.max(0, Math.floor(cy - radius - 1));
  const maxY = Math.min(n - 1, Math.ceil(cy + radius + 1));
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      if (pointInFlatHex(x + 0.5, y + 0.5, cx, cy, radius)) {
        state.mask[y * n + x] = value;
      }
    }
  }
}

function pointInFlatHex(x, y, cx, cy, radius) {
  const px = Math.abs(x - cx) / radius;
  const py = Math.abs(y - cy) / radius;
  return py <= Math.sqrt(3) / 2 && Math.sqrt(3) * px + py <= Math.sqrt(3);
}

function carveStrut(state, cx, cy, angle, halfWidth) {
  const n = state.resolution;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const dx = x - cx;
      const dy = y - cy;
      const perpendicularDistance = Math.abs(-sin * dx + cos * dy);
      const forward = cos * dx + sin * dy;
      if (forward > 0 && perpendicularDistance <= halfWidth) {
        state.mask[y * n + x] = 0;
      }
    }
  }
}

function getApertureState() {
  return {
    version: runtime.state.version,
    resolution: runtime.state.resolution,
    mask: Array.from(runtime.state.mask),
  };
}

function setApertureState(serializedState) {
  if (!serializedState || serializedState.version !== 1) {
    throw new Error("Unsupported aperture state.");
  }
  const resolution = clampInt(serializedState.resolution, 128, 512);
  const next = createApertureState(resolution);
  if (!Array.isArray(serializedState.mask) || serializedState.mask.length !== resolution * resolution) {
    throw new Error("Invalid aperture state payload.");
  }
  for (let i = 0; i < serializedState.mask.length; i += 1) {
    next.mask[i] = serializedState.mask[i] > 0.5 ? 1 : 0;
  }
  runtime.state = next;
  ui.resolution.value = String(resolution);
  renderMask();
  scheduleCompute();
}

window.diffractionApp = {
  getApertureState,
  setApertureState,
};
