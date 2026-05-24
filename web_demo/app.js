const state = {
  config: null,
  selectedFile: null,
  previewUrl: null,
  running: false,
};

const dom = {
  form: document.getElementById("inference-form"),
  imageInput: document.getElementById("image-input"),
  uploadZone: document.getElementById("upload-zone"),
  previewShell: document.getElementById("preview-shell"),
  inputPreview: document.getElementById("input-preview"),
  clearImageBtn: document.getElementById("clear-image-btn"),
  promptInput: document.getElementById("prompt-input"),
  alphasInput: document.getElementById("alphas-input"),
  seedInput: document.getElementById("seed-input"),
  stepsInput: document.getElementById("steps-input"),
  guidanceInput: document.getElementById("guidance-input"),
  maxSeqInput: document.getElementById("max-seq-input"),
  gifDurationInput: document.getElementById("gif-duration-input"),
  runButton: document.getElementById("run-button"),
  statusPill: document.getElementById("status-pill"),
  runtimeMeta: document.getElementById("runtime-meta"),
  emptyState: document.getElementById("empty-state"),
  resultsShell: document.getElementById("results-shell"),
  resultTitle: document.getElementById("result-title"),
  gifLink: document.getElementById("gif-link"),
  gridLink: document.getElementById("grid-link"),
  metaLink: document.getElementById("meta-link"),
  outputDir: document.getElementById("output-dir"),
  promptValue: document.getElementById("prompt-value"),
  resultGifImage: document.getElementById("result-gif-image"),
  resultCompare: document.getElementById("result-compare"),
  resultCompareBefore: document.getElementById("result-compare-before"),
  resultCompareAfter: document.getElementById("result-compare-after"),
  resultCompareLabel: document.getElementById("result-compare-label"),
  compareAlphaPill: document.getElementById("compare-alpha-pill"),
  strengthSlider: document.getElementById("strength-slider"),
  strengthControlValue: document.getElementById("strength-control-value"),
  strengthGrid: document.getElementById("strength-grid"),
  gifLinkInline: document.getElementById("gif-link-inline"),
  gridLinkInline: document.getElementById("grid-link-inline"),
  metaLinkInline: document.getElementById("meta-link-inline"),
};

function setStatus(text, variant = "default") {
  dom.statusPill.textContent = text;
  dom.statusPill.classList.remove("is-running", "is-success", "is-error");
  if (variant === "running") dom.statusPill.classList.add("is-running");
  if (variant === "success") dom.statusPill.classList.add("is-success");
  if (variant === "error") dom.statusPill.classList.add("is-error");
}

function revokePreviewUrl() {
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
  }
}

function applyPreview(file) {
  revokePreviewUrl();
  state.selectedFile = file;
  if (!file) {
    dom.previewShell.hidden = true;
    dom.inputPreview.removeAttribute("src");
    setStatus("等待上传");
    return;
  }
  state.previewUrl = URL.createObjectURL(file);
  dom.inputPreview.src = state.previewUrl;
  dom.previewShell.hidden = false;
  setStatus(`已选择：${file.name}`);
}

function buildMetaItem(label, value) {
  const wrap = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value;
  wrap.appendChild(dt);
  wrap.appendChild(dd);
  return wrap;
}

function renderRuntimeMeta(config) {
  dom.runtimeMeta.innerHTML = "";
  const items = [
    ["Device", config.device],
    ["Torch dtype", config.torch_dtype],
    ["Default alphas", config.alphas.join(", ")],
    ["Default steps", String(config.num_inference_steps)],
    ["Default seed", String(config.seed)],
    ["Model", config.model_path],
    ["LoRA", config.lora_path],
  ];
  items.forEach(([label, value]) => dom.runtimeMeta.appendChild(buildMetaItem(label, value)));
}

function setComparePosition(compare, value) {
  const clamped = Math.max(0, Math.min(100, value));
  const afterWrap = compare.querySelector("[data-after-wrap]");
  const divider = compare.querySelector("[data-divider]");
  afterWrap.style.clipPath = `inset(0 ${100 - clamped}% 0 0)`;
  divider.style.left = `${clamped}%`;
  compare.dataset.position = String(clamped);
}

function bindCompare(compare) {
  const initial = Number(compare.dataset.position || 50);
  setComparePosition(compare, initial);

  let dragging = false;

  const move = (clientX) => {
    const bounds = compare.getBoundingClientRect();
    const next = ((clientX - bounds.left) / bounds.width) * 100;
    setComparePosition(compare, next);
  };

  compare.addEventListener("pointerdown", (event) => {
    dragging = true;
    compare.setPointerCapture(event.pointerId);
    move(event.clientX);
  });

  compare.addEventListener("pointermove", (event) => {
    if (dragging) move(event.clientX);
  });

  const stopDragging = (event) => {
    if (!dragging) return;
    dragging = false;
    if (event.pointerId !== undefined && compare.hasPointerCapture(event.pointerId)) {
      compare.releasePointerCapture(event.pointerId);
    }
  };

  compare.addEventListener("pointerup", stopDragging);
  compare.addEventListener("pointercancel", stopDragging);
  compare.addEventListener("pointerleave", stopDragging);
}

async function loadConfig() {
  const response = await fetch("/api/config", { cache: "no-store" });
  const payload = await response.json();
  state.config = payload;
  if (payload.default_prompt) dom.promptInput.value = payload.default_prompt;
  dom.alphasInput.value = payload.alphas.join(",");
  dom.seedInput.value = payload.seed;
  dom.stepsInput.value = payload.num_inference_steps;
  dom.guidanceInput.value = payload.guidance_scale;
  dom.maxSeqInput.value = payload.max_sequence_length;
  dom.gifDurationInput.value = payload.gif_duration_ms;
  renderRuntimeMeta(payload);
}

function renderOutputs(result) {
  dom.emptyState.hidden = true;
  dom.resultsShell.hidden = false;

  dom.resultTitle.textContent = `Run ${result.run_id}`;
  dom.gifLink.href = result.gif_url;
  dom.gridLink.href = result.grid_url;
  dom.metaLink.href = result.metadata_url;
  dom.gifLinkInline.href = result.gif_url;
  dom.gridLinkInline.href = result.grid_url;
  dom.metaLinkInline.href = result.metadata_url;
  dom.outputDir.textContent = result.output_dir;
  dom.promptValue.textContent = result.prompt || "(empty prompt)";
  dom.resultGifImage.src = `${result.gif_url}?t=${Date.now()}`;
  dom.resultCompareBefore.src = `${result.input_url}?t=${Date.now()}`;

  dom.strengthGrid.innerHTML = "";
  const outputs = result.outputs || [];
  const defaultIndex = Math.max(0, outputs.length - 1);
  dom.strengthSlider.max = String(Math.max(0, outputs.length - 1));
  dom.strengthSlider.value = String(defaultIndex);

  const syncStrength = () => {
    const index = Number(dom.strengthSlider.value);
    const item = outputs[index];
    if (!item) return;
    const alphaText = typeof item.alpha === "number" ? item.alpha.toFixed(2) : item.label.replace("Alpha ", "");
    dom.resultCompareAfter.src = `${item.image_url}?t=${Date.now()}`;
    dom.resultCompareLabel.textContent = `Alpha ${alphaText}`;
    dom.compareAlphaPill.textContent = `Alpha ${alphaText}`;
    dom.strengthControlValue.textContent = alphaText;
  };

  dom.strengthSlider.oninput = syncStrength;
  syncStrength();
}

function validateBeforeSubmit() {
  if (!state.selectedFile) {
    throw new Error("请先上传一张图片。");
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  if (state.running) return;

  try {
    validateBeforeSubmit();
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }

  const formData = new FormData();
  formData.append("image", state.selectedFile);
  formData.append("prompt", dom.promptInput.value.trim());
  formData.append("alphas", dom.alphasInput.value.trim());
  formData.append("seed", dom.seedInput.value.trim());
  formData.append("num_inference_steps", dom.stepsInput.value.trim());
  formData.append("guidance_scale", dom.guidanceInput.value.trim());
  formData.append("max_sequence_length", dom.maxSeqInput.value.trim());
  formData.append("gif_duration_ms", dom.gifDurationInput.value.trim());

  state.running = true;
  dom.runButton.disabled = true;
  setStatus("推理中：首次加载模型会更慢，请耐心等一下…", "running");

  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || `请求失败（${response.status}）`);
    }
    renderOutputs(payload);
    setStatus("推理完成", "success");
    document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setStatus(error.message || "推理失败", "error");
  } finally {
    state.running = false;
    dom.runButton.disabled = false;
  }
}

function onFileSelected(file) {
  if (!file) return;
  applyPreview(file);
}

function wireUploadInteractions() {
  dom.imageInput.addEventListener("change", () => {
    const file = dom.imageInput.files?.[0];
    onFileSelected(file);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dom.uploadZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dom.uploadZone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dom.uploadZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dom.uploadZone.classList.remove("is-dragover");
    });
  });

  dom.uploadZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      dom.imageInput.files = event.dataTransfer.files;
      onFileSelected(file);
    }
  });

  dom.clearImageBtn.addEventListener("click", () => {
    dom.imageInput.value = "";
    applyPreview(null);
  });
}

async function bootstrap() {
  setStatus("加载配置中…", "running");
  try {
    await loadConfig();
    bindCompare(dom.resultCompare);
    wireUploadInteractions();
    dom.form.addEventListener("submit", handleSubmit);
    setStatus("等待上传");
  } catch (error) {
    setStatus(`初始化失败：${error.message || error}`, "error");
  }
}

bootstrap();
