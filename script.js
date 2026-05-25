const heroData = window.PAPER_HERO_DATA || null;
const galleryData = window.PAPER_GALLERY_DATA || [];
const FRAMES = [
  { label: 'Input', key: '0' },
  { label: '0.25', key: '25' },
  { label: '0.50', key: '50' },
  { label: '0.75', key: '75' },
  { label: '1.00', key: '100' },
];

const imageCache = new Map();
const preloadQueue = [];
let preloadPumpActive = false;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sliderPercent(slider) {
  const min = Number(slider.min || 0);
  const max = Number(slider.max || 100);
  const value = Number(slider.value || 0);
  if (max === min) return 0;
  return clamp(((value - min) / (max - min)) * 100, 0, 100);
}

function updateSliderFill(slider) {
  slider.style.setProperty('--pct', `${sliderPercent(slider)}%`);
}

function nearestFrame(value) {
  const percent = clamp(Number(value || 0), 0, 100);
  const index = clamp(Math.round((percent / 100) * (FRAMES.length - 1)), 0, FRAMES.length - 1);
  return FRAMES[index];
}

function preloadImage(src) {
  if (!src || imageCache.has(src)) return imageCache.get(src);
  const image = new Image();
  image.decoding = 'async';
  image.src = src;
  const promise = image.decode ? image.decode().catch(() => undefined) : Promise.resolve();
  imageCache.set(src, promise);
  return promise;
}

function enqueuePreload(src) {
  if (!src || imageCache.has(src)) return;
  preloadQueue.push(src);
}

function pumpPreloadQueue() {
  if (preloadPumpActive) return;
  preloadPumpActive = true;

  const pump = () => {
    let budget = 2;
    while (budget > 0 && preloadQueue.length) {
      preloadImage(preloadQueue.shift());
      budget -= 1;
    }

    if (preloadQueue.length) {
      window.setTimeout(pump, 90);
    } else {
      preloadPumpActive = false;
    }
  };

  pump();
}

function enqueueSample(sample, allFrames = false) {
  if (!sample?.paths) return;
  const frames = allFrames ? FRAMES : [FRAMES[0]];
  frames.forEach((frame) => enqueuePreload(sample.paths[frame.key]));
}

function primeInitialImages() {
  if (heroData) FRAMES.forEach((frame) => preloadImage(heroData.paths[frame.key]));

  // Keep first paint light: only queue input frames for gallery first.
  galleryData.forEach((sample) => enqueueSample(sample, false));
  pumpPreloadQueue();

  const loadRest = () => {
    galleryData.forEach((sample) => enqueueSample(sample, true));
    pumpPreloadQueue();
  };

  if ('requestIdleCallback' in window) {
    window.requestIdleCallback(loadRest, { timeout: 2200 });
  } else {
    window.setTimeout(loadRest, 900);
  }
}

function bindNearestViewer({ slider, image, sample, preloadAll = false }) {
  if (!slider || !image || !sample) return;

  let activeKey = '';
  let framePending = false;
  let interacted = false;

  function ensureAllFrames() {
    if (interacted) return;
    interacted = true;
    FRAMES.forEach((frame) => preloadImage(sample.paths[frame.key]));
  }

  function applyFrame() {
    framePending = false;
    updateSliderFill(slider);

    const frame = nearestFrame(slider.value);
    const nextSrc = sample.paths[frame.key];
    if (activeKey !== frame.key) {
      activeKey = frame.key;
      if (image.getAttribute('src') !== nextSrc) image.src = nextSrc;
    }
  }

  function scheduleFrame() {
    ensureAllFrames();
    if (framePending) return;
    framePending = true;
    requestAnimationFrame(applyFrame);
  }

  preloadImage(sample.paths[FRAMES[0].key]);
  if (preloadAll) FRAMES.forEach((frame) => preloadImage(sample.paths[frame.key]));

  slider.addEventListener('pointerdown', ensureAllFrames, { passive: true });
  slider.addEventListener('touchstart', ensureAllFrames, { passive: true });
  slider.addEventListener('focus', ensureAllFrames, { passive: true });
  slider.addEventListener('input', scheduleFrame, { passive: true });
  requestAnimationFrame(applyFrame);
}

function setupHero() {
  const sample = heroData || galleryData[0];
  if (!sample) return;

  bindNearestViewer({
    slider: document.getElementById('hero-slider'),
    image: document.getElementById('hero-image'),
    sample,
    preloadAll: true,
  });
}

function renderGalleryBatch(startIndex, batchSize) {
  const grid = document.getElementById('gallery-grid');
  const template = document.getElementById('interactive-card-template');
  if (!grid || !template) return;

  const fragmentList = document.createDocumentFragment();
  const slice = galleryData.slice(startIndex, startIndex + batchSize);
  slice.forEach((sample, offset) => {
    const index = startIndex + offset;
    const fragment = template.content.cloneNode(true);
    const image = fragment.querySelector('.target-img');
    const slider = fragment.querySelector('.result-slider');

    image.alt = `ControlLight example ${index + 1}`;
    image.loading = index < 6 ? 'eager' : 'lazy';
    image.decoding = 'async';

    bindNearestViewer({ slider, image, sample, preloadAll: index < 4 });
    fragmentList.appendChild(fragment);
  });
  grid.appendChild(fragmentList);
}

function renderGallery() {
  const batchSize = 6;
  let index = 0;

  renderGalleryBatch(index, batchSize);
  index += batchSize;

  const renderNext = () => {
    if (index >= galleryData.length) return;
    renderGalleryBatch(index, batchSize);
    index += batchSize;
    window.setTimeout(renderNext, 80);
  };

  window.setTimeout(renderNext, 80);
}

document.addEventListener('DOMContentLoaded', () => {
  setupHero();
  renderGallery();
  primeInitialImages();
});
