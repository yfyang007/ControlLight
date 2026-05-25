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

function preloadSample(sample) {
  if (!sample?.paths) return;
  FRAMES.forEach((frame) => preloadImage(sample.paths[frame.key]));
}

function primeInitialImages() {
  if (heroData) preloadSample(heroData);
  galleryData.slice(0, 8).forEach(preloadSample);
  if ('requestIdleCallback' in window) {
    window.requestIdleCallback(() => galleryData.slice(8).forEach(preloadSample), { timeout: 1800 });
  } else {
    window.setTimeout(() => galleryData.slice(8).forEach(preloadSample), 400);
  }
}

function bindNearestViewer({ slider, image, sample }) {
  if (!slider || !image || !sample) return;

  let activeKey = '';
  let framePending = false;

  function applyFrame() {
    framePending = false;
    updateSliderFill(slider);

    const frame = nearestFrame(slider.value);
    const nextSrc = sample.paths[frame.key];
    if (activeKey !== frame.key) {
      activeKey = frame.key;
      if (image.src !== nextSrc) image.src = nextSrc;
    }
  }

  function scheduleFrame() {
    if (framePending) return;
    framePending = true;
    requestAnimationFrame(applyFrame);
  }

  preloadSample(sample);
  slider.addEventListener('input', scheduleFrame, { passive: true });
  scheduleFrame();
}

function setupHero() {
  const sample = heroData || galleryData[0];
  if (!sample) return;

  bindNearestViewer({
    slider: document.getElementById('hero-slider'),
    image: document.getElementById('hero-image'),
    sample,
  });
}

function renderGallery() {
  const grid = document.getElementById('gallery-grid');
  const template = document.getElementById('interactive-card-template');
  if (!grid || !template) return;

  const fragmentList = document.createDocumentFragment();
  galleryData.forEach((sample, index) => {
    const fragment = template.content.cloneNode(true);
    const image = fragment.querySelector('.target-img');
    const slider = fragment.querySelector('.result-slider');

    image.alt = `ControlLight example ${index + 1}`;
    image.loading = index < 6 ? 'eager' : 'lazy';
    image.decoding = 'async';

    bindNearestViewer({ slider, image, sample });
    fragmentList.appendChild(fragment);
  });
  grid.appendChild(fragmentList);
}

document.addEventListener('DOMContentLoaded', () => {
  primeInitialImages();
  setupHero();
  renderGallery();
});
