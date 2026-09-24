self.addEventListener("message", (event) => {
  const request = event.data;
  if (!request || request.type !== "compute") {
    return;
  }

  const startedAt = performance.now();
  try {
    const resolution = request.resolution;
    const aperture = new Uint8Array(request.mask);
    const settings = request.settings;
    const intensity = computeReferenceIntensity(aperture, resolution);
    const rgba = composeSpectrum(intensity, resolution, settings);
    self.postMessage(
      {
        type: "result",
        requestId: request.requestId,
        elapsedMs: performance.now() - startedAt,
        rgba: rgba.buffer,
      },
      [rgba.buffer],
    );
  } catch (error) {
    self.postMessage({
      type: "error",
      requestId: request.requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

function composeSpectrum(referenceIntensity, n, settings) {
  const output = new Float32Array(n * n * 3);
  const wavelengths =
    settings.mode === "single"
      ? [settings.singleLambda]
      : linspace(settings.polyMin, settings.polyMax, settings.polySamples);

  for (const wavelength of wavelengths) {
    const color = wavelengthToRgb(wavelength);
    const scale = 550 / wavelength;
    for (let y = 0; y < n; y += 1) {
      const sourceY = n / 2 + (y - n / 2) * scale;
      for (let x = 0; x < n; x += 1) {
        const value = sampleBilinear(
          referenceIntensity,
          n,
          n / 2 + (x - n / 2) * scale,
          sourceY,
        );
        const offset = (y * n + x) * 3;
        output[offset] += value * color[0];
        output[offset + 1] += value * color[1];
        output[offset + 2] += value * color[2];
      }
    }
  }

  normalizeRgbBuffer(output);
  const rgba = new Uint8ClampedArray(n * n * 4);
  for (let i = 0; i < n * n; i += 1) {
    const source = i * 3;
    const target = i * 4;
    rgba[target] = Math.round(255 * Math.pow(Math.max(0, output[source]), 0.85));
    rgba[target + 1] = Math.round(255 * Math.pow(Math.max(0, output[source + 1]), 0.85));
    rgba[target + 2] = Math.round(255 * Math.pow(Math.max(0, output[source + 2]), 0.85));
    rgba[target + 3] = 255;
  }
  return rgba;
}

function computeReferenceIntensity(aperture, n) {
  const re = new Float64Array(n * n);
  const im = new Float64Array(n * n);
  for (let i = 0; i < aperture.length; i += 1) {
    re[i] = aperture[i];
  }

  const rowRe = new Float64Array(n);
  const rowIm = new Float64Array(n);
  for (let y = 0; y < n; y += 1) {
    const rowOffset = y * n;
    for (let x = 0; x < n; x += 1) {
      rowRe[x] = re[rowOffset + x];
      rowIm[x] = im[rowOffset + x];
    }
    fft1d(rowRe, rowIm);
    for (let x = 0; x < n; x += 1) {
      re[rowOffset + x] = rowRe[x];
      im[rowOffset + x] = rowIm[x];
    }
  }

  const columnRe = new Float64Array(n);
  const columnIm = new Float64Array(n);
  for (let x = 0; x < n; x += 1) {
    for (let y = 0; y < n; y += 1) {
      const index = y * n + x;
      columnRe[y] = re[index];
      columnIm[y] = im[index];
    }
    fft1d(columnRe, columnIm);
    for (let y = 0; y < n; y += 1) {
      const index = y * n + x;
      re[index] = columnRe[y];
      im[index] = columnIm[y];
    }
  }

  const half = n / 2;
  const raw = new Float64Array(n * n);
  let maxIntensity = 0;
  for (let i = 0; i < raw.length; i += 1) {
    raw[i] = re[i] * re[i] + im[i] * im[i];
    if (raw[i] > maxIntensity) {
      maxIntensity = raw[i];
    }
  }

  const shifted = new Float32Array(n * n);
  if (maxIntensity === 0) {
    return shifted;
  }
  let maxLog = 0;
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const srcX = (x + half) % n;
      const srcY = (y + half) % n;
      const value = raw[srcY * n + srcX];
      const logged = value > maxIntensity * 1e-5 ? Math.log1p(value) : 0;
      shifted[y * n + x] = logged;
      if (logged > maxLog) {
        maxLog = logged;
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

function sampleBilinear(image, n, x, y) {
  if (x < 0 || y < 0 || x > n - 1 || y > n - 1) {
    return 0;
  }
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(n - 1, x0 + 1);
  const y1 = Math.min(n - 1, y0 + 1);
  const tx = x - x0;
  const ty = y - y0;
  const top = image[y0 * n + x0] * (1 - tx) + image[y0 * n + x1] * tx;
  const bottom = image[y1 * n + x0] * (1 - tx) + image[y1 * n + x1] * tx;
  return top * (1 - ty) + bottom * ty;
}

function fft1d(re, im) {
  const n = re.length;
  if ((n & (n - 1)) !== 0) {
    throw new Error("FFT size must be a power of two.");
  }

  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    while ((j & bit) !== 0) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const real = re[i];
      re[i] = re[j];
      re[j] = real;
      const imaginary = im[i];
      im[i] = im[j];
      im[j] = imaginary;
    }
  }

  for (let length = 2; length <= n; length <<= 1) {
    const angle = (-2 * Math.PI) / length;
    const stepReal = Math.cos(angle);
    const stepImaginary = Math.sin(angle);
    for (let offset = 0; offset < n; offset += length) {
      let weightReal = 1;
      let weightImaginary = 0;
      const halfLength = length >> 1;
      for (let j = 0; j < halfLength; j += 1) {
        const left = offset + j;
        const right = left + halfLength;
        const rightReal = re[right] * weightReal - im[right] * weightImaginary;
        const rightImaginary = re[right] * weightImaginary + im[right] * weightReal;
        const leftReal = re[left];
        const leftImaginary = im[left];
        re[left] = leftReal + rightReal;
        im[left] = leftImaginary + rightImaginary;
        re[right] = leftReal - rightReal;
        im[right] = leftImaginary - rightImaginary;

        const nextReal = weightReal * stepReal - weightImaginary * stepImaginary;
        const nextImaginary = weightReal * stepImaginary + weightImaginary * stepReal;
        weightReal = nextReal;
        weightImaginary = nextImaginary;
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
    buffer[i] /= max;
  }
}

function wavelengthToRgb(wave) {
  const gamma = 0.8;
  let red = 0;
  let green = 0;
  let blue = 0;

  if (wave >= 380 && wave < 440) {
    red = -(wave - 440) / 60;
    blue = 1;
  } else if (wave < 490) {
    green = (wave - 440) / 50;
    blue = 1;
  } else if (wave < 510) {
    green = 1;
    blue = -(wave - 510) / 20;
  } else if (wave < 580) {
    red = (wave - 510) / 70;
    green = 1;
  } else if (wave < 645) {
    red = 1;
    green = -(wave - 645) / 65;
  } else if (wave <= 780) {
    red = 1;
  }

  let factor = 0;
  if (wave >= 380 && wave < 420) {
    factor = 0.3 + (0.7 * (wave - 380)) / 40;
  } else if (wave < 700) {
    factor = 1;
  } else if (wave <= 780) {
    factor = 0.3 + (0.7 * (780 - wave)) / 80;
  }

  const channel = (value) => (value === 0 ? 0 : Math.pow(value * factor, gamma));
  return [channel(red), channel(green), channel(blue)];
}

function linspace(start, end, count) {
  if (count <= 1) {
    return [start];
  }
  const values = [];
  const step = (end - start) / (count - 1);
  for (let i = 0; i < count; i += 1) {
    values.push(start + i * step);
  }
  return values;
}