const MIN_WAVELENGTH = 380;
const MAX_WAVELENGTH = 780;
const SUPPORTED_RESOLUTIONS = [128, 256, 512];
const DEFAULTS = {
  mode: "single",
  singleLambda: 550,
  polyMin: 420,
  polyMax: 680,
  polySamples: 9,
  resolution: 256,
  preset: "round",
};
const COMPUTE_INTERVAL_MS = 80;

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

const maskCtx = ui.maskCanvas.getContext("2d", { alpha: false });
const patternCtx = ui.patternCanvas.getContext("2d", { alpha: false });
const runtime = {
  state: createApertureState(DEFAULTS.resolution),
  worker: null,
  isDrawing: false,
  drawValue: 1,
  timer: null,
  lastComputeStart: 0,
  revision: 0,
  inFlight: false,
  queued: false,
};

applyPreset(runtime.state, DEFAULTS.preset);
setupUi();
renderMask();
initializeWorker();
installParentFrameResize();
scheduleCompute();

function installParentFrameResize() {
  const reportHeight = () => {
    if (window.parent !== window) {
      window.parent.postMessage(
        { type: "diffraction-frame-height", height: document.documentElement.scrollHeight },
        window.location.origin,
      );
    }
  };
  window.addEventListener("resize", reportHeight);
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(reportHeight).observe(document.body);
  }
  window.addEventListener("load", reportHeight, { once: true });
  reportHeight();
}

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
  [ui.singleLambda, ui.polyMin, ui.polyMax, ui.polySamples].forEach((input) => {
    input.addEventListener("input", scheduleCompute);
    input.addEventListener("change", scheduleCompute);
  });
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
    event.preventDefault();
    runtime.isDrawing = true;
    runtime.drawValue = event.button === 2 || event.shiftKey ? 0 : 1;
    if (ui.maskCanvas.setPointerCapture) {
      ui.maskCanvas.setPointerCapture(event.pointerId);
    }
    drawFromPointer(event);
  });
  ui.maskCanvas.addEventListener("pointermove", (event) => {
    if (runtime.isDrawing) {
      drawFromPointer(event);
    }
  });
  const stopDrawing = (event) => {
    runtime.isDrawing = false;
    if (
      event?.pointerId !== undefined &&
      ui.maskCanvas.hasPointerCapture?.(event.pointerId)
    ) {
      ui.maskCanvas.releasePointerCapture(event.pointerId);
    }
  };
  ui.maskCanvas.addEventListener("pointerup", stopDrawing);
  ui.maskCanvas.addEventListener("pointercancel", stopDrawing);
  ui.maskCanvas.addEventListener("lostpointercapture", stopDrawing);
  ui.maskCanvas.addEventListener("contextmenu", (event) => event.preventDefault());
}

function initializeWorker() {
  if (typeof Worker === "undefined") {
    setStatus("This browser does not support background simulation workers.");
    return;
  }

  try {
    runtime.worker = new Worker(new URL("./diffraction-worker.js", import.meta.url), {
      type: "module",
    });
  } catch (error) {
    setStatus("Could not start the diffraction worker: " + error.message);
    return;
  }

  runtime.worker.addEventListener("message", handleWorkerMessage);
  runtime.worker.addEventListener("error", (event) => {
    runtime.inFlight = false;
    runtime.queued = false;
    setStatus("Diffraction worker failed: " + event.message);
  });
  runtime.worker.addEventListener("messageerror", () => {
    runtime.inFlight = false;
    runtime.queued = false;
    setStatus("The diffraction worker returned an unreadable result.");
  });
}

function updateModeVisibility() {
  const singleVisible = ui.mode.value === "single";
  document.querySelectorAll(".mode-single").forEach((element) => {
    element.hidden = !singleVisible;
  });
  document.querySelectorAll(".mode-poly").forEach((element) => {
    element.hidden = singleVisible;
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
  const n = runtime.state.resolution;
  const x = ((event.clientX - rect.left) / rect.width) * n;
  const y = ((event.clientY - rect.top) / rect.height) * n;
  paintDisc(runtime.state, x, y, Math.max(1, Math.floor(n / 64)), runtime.drawValue);
  renderMask();
  scheduleCompute();
}

function paintDisc(state, x, y, radius, value) {
  const n = state.resolution;
  const cx = Math.floor(x);
  const cy = Math.floor(y);
  const radiusSquared = radius * radius;
  for (let py = Math.max(0, cy - radius); py <= Math.min(n - 1, cy + radius); py += 1) {
    for (let px = Math.max(0, cx - radius); px <= Math.min(n - 1, cx + radius); px += 1) {
      const dx = px - x;
      const dy = py - y;
      if (dx * dx + dy * dy <= radiusSquared) {
        state.mask[py * n + px] = value;
      }
    }
  }
}

function handleResolutionChange() {
  const nextResolution = Number(ui.resolution.value);
  if (!SUPPORTED_RESOLUTIONS.includes(nextResolution)) {
    ui.resolution.value = String(runtime.state.resolution);
    return;
  }
  runtime.state = resizeApertureState(runtime.state, nextResolution);
  renderMask();
  scheduleCompute();
}

function renderMask() {
  const n = runtime.state.resolution;
  const image = maskCtx.createImageData(n, n);
  for (let i = 0; i < runtime.state.mask.length; i += 1) {
    const value = runtime.state.mask[i] ? 255 : 0;
    const offset = i * 4;
    image.data[offset] = value;
    image.data[offset + 1] = value;
    image.data[offset + 2] = value;
    image.data[offset + 3] = 255;
  }
  const scratch = createScratchCanvas(n);
  scratch.getContext("2d", { alpha: false }).putImageData(image, 0, 0);
  maskCtx.clearRect(0, 0, ui.maskCanvas.width, ui.maskCanvas.height);
  maskCtx.imageSmoothingEnabled = false;
  maskCtx.drawImage(scratch, 0, 0, ui.maskCanvas.width, ui.maskCanvas.height);
}

function scheduleCompute() {
  runtime.revision += 1;
  setStatus("Updating diffraction pattern…");
  if (runtime.timer !== null) {
    window.clearTimeout(runtime.timer);
  }
  const elapsed = performance.now() - runtime.lastComputeStart;
  const delay = Math.max(0, COMPUTE_INTERVAL_MS - elapsed);
  runtime.timer = window.setTimeout(() => {
    runtime.timer = null;
    dispatchCompute();
  }, delay);
}

function dispatchCompute() {
  if (!runtime.worker) {
    return;
  }
  if (runtime.inFlight) {
    runtime.queued = true;
    return;
  }

  const validated = validateControls();
  if (!validated.ok) {
    setStatus(validated.message);
    return;
  }

  runtime.inFlight = true;
  runtime.queued = false;
  runtime.lastComputeStart = performance.now();
  const requestId = runtime.revision;
  setStatus("Computing diffraction pattern…");

  const maskCopy = runtime.state.mask.slice();
  runtime.worker.postMessage(
    {
      type: "compute",
      requestId,
      resolution: runtime.state.resolution,
      mask: maskCopy.buffer,
      settings: validated.settings,
    },
    [maskCopy.buffer],
  );
}

function handleWorkerMessage(event) {
  const result = event.data;
  if (!result || (result.type !== "result" && result.type !== "error")) {
    return;
  }
  runtime.inFlight = false;

  if (result.type === "error") {
    if (result.requestId === runtime.revision) {
      setStatus("Could not calculate the diffraction pattern: " + result.message);
    }
  } else if (result.requestId === runtime.revision) {
    renderPattern(new Uint8ClampedArray(result.rgba), runtime.state.resolution);
    setStatus(
      "Updated in " +
        Math.round(result.elapsedMs) +
        " ms at " +
        runtime.state.resolution +
        " × " +
        runtime.state.resolution,
    );
  }

  if (runtime.queued || result.requestId !== runtime.revision) {
    runtime.queued = false;
    dispatchCompute();
  }
}

function validateControls() {
  const mode = ui.mode.value === "poly" ? "poly" : "single";
  const singleLambda = clampInt(Number(ui.singleLambda.value), MIN_WAVELENGTH, MAX_WAVELENGTH);
  ui.singleLambda.value = String(singleLambda);

  let polyMin = clampInt(Number(ui.polyMin.value), MIN_WAVELENGTH, MAX_WAVELENGTH);
  let polyMax = clampInt(Number(ui.polyMax.value), MIN_WAVELENGTH, MAX_WAVELENGTH);
  if (polyMin > polyMax) {
    const swap = polyMin;
    polyMin = polyMax;
    polyMax = swap;
  }
  ui.polyMin.value = String(polyMin);
  ui.polyMax.value = String(polyMax);

  const polySamples = clampOdd(Number(ui.polySamples.value), 3, 21);
  ui.polySamples.value = String(polySamples);

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

function createScratchCanvas(size) {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(size, size);
  }
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  return canvas;
}

function renderPattern(rgba, n) {
  const image = patternCtx.createImageData(n, n);
  image.data.set(rgba);
  const scratch = createScratchCanvas(n);
  scratch.getContext("2d", { alpha: false }).putImageData(image, 0, 0);
  patternCtx.clearRect(0, 0, ui.patternCanvas.width, ui.patternCanvas.height);
  patternCtx.imageSmoothingEnabled = false;
  patternCtx.drawImage(scratch, 0, 0, ui.patternCanvas.width, ui.patternCanvas.height);
}

function createApertureState(resolution) {
  return {
    version: 1,
    resolution,
    mask: new Uint8Array(resolution * resolution),
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
  const radiusSquared = radius * radius;
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const dx = x - center;
      const dy = y - center;
      if (dx * dx + dy * dy <= radiusSquared) {
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
      if (Math.abs(dx + separation) <= slitWidth || Math.abs(dx - separation) <= slitWidth) {
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
  const obstructionSquared = obstruction * obstruction;
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const dx = x - center;
      const dy = y - center;
      if (dx * dx + dy * dy <= obstructionSquared) {
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
  const resolution = Number(serializedState.resolution);
  if (!SUPPORTED_RESOLUTIONS.includes(resolution)) {
    throw new Error("Unsupported aperture resolution.");
  }
  if (
    !Array.isArray(serializedState.mask) ||
    serializedState.mask.length !== resolution * resolution
  ) {
    throw new Error("Invalid aperture state payload.");
  }

  const next = createApertureState(resolution);
  for (let i = 0; i < serializedState.mask.length; i += 1) {
    next.mask[i] = Number(serializedState.mask[i]) > 0.5 ? 1 : 0;
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