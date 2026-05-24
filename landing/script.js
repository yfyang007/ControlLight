const heroSteps = [
  { label: "0.25", title: "Alpha 0.25", footnote: "Gentle lift", image: "./assets/lowlight_0003/enhance_default/alpha_0.25.png" },
  { label: "0.50", title: "Alpha 0.50", footnote: "Balanced recovery", image: "./assets/lowlight_0003/enhance_default/alpha_0.50.png" },
  { label: "0.75", title: "Alpha 0.75", footnote: "Strong relight", image: "./assets/lowlight_0003/enhance_default/alpha_0.75.png" },
  { label: "1.00", title: "Alpha 1.00", footnote: "Maximum intensity", image: "./assets/lowlight_0003/enhance_default/alpha_1.00.png" },
];

const showcaseItems = [
  {
    title: "Portrait by candlelight",
    kicker: "Portrait",
    description: "同一推理路径下观察四个强度点，面部结构、手部和烛光高光会随着 alpha 逐步抬起。",
    gif: "./assets/lowlight_0003/comparison.gif",
    before: "./assets/lowlight_0003/input.png",
    steps: [
      { label: "0.25", title: "Alpha 0.25", footnote: "Low intensity", image: "./assets/lowlight_0003/enhance_default/alpha_0.25.png" },
      { label: "0.50", title: "Alpha 0.50", footnote: "Mid intensity", image: "./assets/lowlight_0003/enhance_default/alpha_0.50.png" },
      { label: "0.75", title: "Alpha 0.75", footnote: "High intensity", image: "./assets/lowlight_0003/enhance_default/alpha_0.75.png" },
      { label: "1.00", title: "Alpha 1.00", footnote: "Peak intensity", image: "./assets/lowlight_0003/enhance_default/alpha_1.00.png" },
    ],
  },
  {
    title: "Stone tunnel exposure",
    kicker: "Architecture",
    description: "观察隧道暗部纹理如何逐级打开，同时出口高光保持可控，不是一次性过曝。",
    gif: "./assets/lowlight_0012/comparison.gif",
    before: "./assets/lowlight_0012/input.png",
    steps: [
      { label: "0.25", title: "Alpha 0.25", footnote: "Low intensity", image: "./assets/lowlight_0012/enhance_default/alpha_0.25.png" },
      { label: "0.50", title: "Alpha 0.50", footnote: "Mid intensity", image: "./assets/lowlight_0012/enhance_default/alpha_0.50.png" },
      { label: "0.75", title: "Alpha 0.75", footnote: "High intensity", image: "./assets/lowlight_0012/enhance_default/alpha_0.75.png" },
      { label: "1.00", title: "Alpha 1.00", footnote: "Peak intensity", image: "./assets/lowlight_0012/enhance_default/alpha_1.00.png" },
    ],
  },
  {
    title: "Interior window recovery",
    kicker: "Room scene",
    description: "室内场景很适合展示强度轴，窗边亮部和椅子阴影在四个点上都有连续变化。",
    gif: "./assets/lowlight_0026/comparison.gif",
    before: "./assets/lowlight_0026/input.png",
    steps: [
      { label: "0.25", title: "Alpha 0.25", footnote: "Low intensity", image: "./assets/lowlight_0026/enhance_default/alpha_0.25.png" },
      { label: "0.50", title: "Alpha 0.50", footnote: "Mid intensity", image: "./assets/lowlight_0026/enhance_default/alpha_0.50.png" },
      { label: "0.75", title: "Alpha 0.75", footnote: "High intensity", image: "./assets/lowlight_0026/enhance_default/alpha_0.75.png" },
      { label: "1.00", title: "Alpha 1.00", footnote: "Peak intensity", image: "./assets/lowlight_0026/enhance_default/alpha_1.00.png" },
    ],
  },
  {
    title: "Urban facade relighting",
    kicker: "Exterior",
    description: "建筑外立面适合看整体曝光推进，低强度保留夜感，高强度接近白天信息量。",
    gif: "./assets/lowlight_0033/comparison.gif",
    before: "./assets/lowlight_0033/input.png",
    steps: [
      { label: "0.25", title: "Alpha 0.25", footnote: "Low intensity", image: "./assets/lowlight_0033/enhance_default/alpha_0.25.png" },
      { label: "0.50", title: "Alpha 0.50", footnote: "Mid intensity", image: "./assets/lowlight_0033/enhance_default/alpha_0.50.png" },
      { label: "0.75", title: "Alpha 0.75", footnote: "High intensity", image: "./assets/lowlight_0033/enhance_default/alpha_0.75.png" },
      { label: "1.00", title: "Alpha 1.00", footnote: "Peak intensity", image: "./assets/lowlight_0033/enhance_default/alpha_1.00.png" },
    ],
  },
];

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
    if (dragging) {
      move(event.clientX);
    }
  });

  const stopDragging = (event) => {
    if (!dragging) {
      return;
    }
    dragging = false;
    if (event.pointerId !== undefined && compare.hasPointerCapture(event.pointerId)) {
      compare.releasePointerCapture(event.pointerId);
    }
  };

  compare.addEventListener("pointerup", stopDragging);
  compare.addEventListener("pointercancel", stopDragging);
  compare.addEventListener("pointerleave", stopDragging);
}

function createProgressionStrip(container, before, steps) {
  container.innerHTML = "";
  const frames = [{ label: "Input", image: before }, ...steps.map((step) => ({ label: step.label, image: step.image }))];

  frames.forEach((frame) => {
    const item = document.createElement("div");
    item.className = "progression-item";

    const media = document.createElement("div");
    media.className = "progression-media";

    const image = document.createElement("img");
    image.src = frame.image;
    image.alt = `${frame.label} frame`;
    media.appendChild(image);

    const label = document.createElement("span");
    label.className = "progression-label";
    label.textContent = frame.label;

    item.appendChild(media);
    item.appendChild(label);
    container.appendChild(item);
  });
}

function setupHero() {
  const heroCompare = document.querySelector(".compare-large");
  const afterImage = heroCompare.querySelector("[data-after-image]");
  const heroCompareLabel = document.getElementById("hero-compare-label");
  const heroLabel = document.getElementById("hero-label");
  const heroFootnote = document.getElementById("hero-footnote");
  const slider = document.getElementById("hero-strength-slider");
  const strip = document.getElementById("hero-progression-strip");

  createProgressionStrip(strip, "./assets/lowlight_0003/input.png", heroSteps);

  const sync = () => {
    const step = heroSteps[Number(slider.value)];
    afterImage.src = step.image;
    heroCompareLabel.textContent = step.title;
    heroLabel.textContent = step.title;
    heroFootnote.textContent = step.footnote;
  };

  slider.addEventListener("input", sync);
  sync();
}

function renderShowcase() {
  const rail = document.getElementById("showcase-rail");
  const template = document.getElementById("showcase-card-template");

  showcaseItems.forEach((item) => {
    const fragment = template.content.cloneNode(true);
    const kicker = fragment.querySelector(".showcase-card-kicker");
    const title = fragment.querySelector(".showcase-card-title");
    const active = fragment.querySelector(".showcase-card-active");
    const note = fragment.querySelector(".showcase-card-note");
    const description = fragment.querySelector(".showcase-description");
    const footnote = fragment.querySelector(".showcase-footnote");
    const gifLink = fragment.querySelector(".showcase-gif-link");
    const compare = fragment.querySelector("[data-compare]");
    const beforeImage = compare.querySelector("[data-before-image]");
    const afterImage = compare.querySelector("[data-after-image]");
    const rightLabel = fragment.querySelector(".showcase-card-label");
    const slider = fragment.querySelector(".showcase-strength-slider");
    const strip = fragment.querySelector(".showcase-progression-strip");

    kicker.textContent = item.kicker;
    title.textContent = item.title;
    description.textContent = item.description;
    gifLink.href = item.gif;
    beforeImage.src = item.before;
    createProgressionStrip(strip, item.before, item.steps);

    const sync = () => {
      const step = item.steps[Number(slider.value)];
      afterImage.src = step.image;
      rightLabel.textContent = step.title;
      active.textContent = step.title;
      note.textContent = step.footnote;
      footnote.textContent = `${step.footnote} · same image progression`;
    };

    slider.addEventListener("input", sync);
    sync();

    rail.appendChild(fragment);
    bindCompare(rail.lastElementChild.querySelector("[data-compare]"));
  });
}

function startHeroAutoplay() {
  const heroCompare = document.querySelector(".compare-large");
  let direction = 1;
  let position = Number(heroCompare.dataset.position || 58);

  window.setInterval(() => {
    if (heroCompare.matches(":hover")) {
      return;
    }
    position += direction * 0.5;
    if (position >= 72) {
      direction = -1;
    }
    if (position <= 34) {
      direction = 1;
    }
    setComparePosition(heroCompare, position);
  }, 32);
}

document.querySelectorAll("[data-compare]").forEach(bindCompare);
setupHero();
renderShowcase();
startHeroAutoplay();
