const galleryData = window.PAPER_GALLERY_DATA || [];
const FRAMES = [
  { label: 'Input', key: '0' },
  { label: '0.25', key: '25' },
  { label: '0.50', key: '50' },
  { label: '0.75', key: '75' },
  { label: '1.00', key: '100' },
];

function updateSliderFill(slider) {
  slider.style.setProperty('--pct', `${Number(slider.value || 0)}%`);
}

function labelForValue(value) {
  if (value <= 1) return 'Input';
  return `s = ${(value / 100).toFixed(2)}`;
}

function bindBlendViewer({ slider, baseImage, overlayImage, indicator, sample }) {
  function sync() {
    updateSliderFill(slider);
    const value = Number(slider.value || 0);
    const position = (value / 100) * (FRAMES.length - 1);
    const lower = Math.max(0, Math.min(FRAMES.length - 1, Math.floor(position)));
    const upper = Math.max(0, Math.min(FRAMES.length - 1, Math.ceil(position)));
    const opacity = upper === lower ? 0 : position - lower;

    baseImage.src = sample.paths[FRAMES[lower].key];
    overlayImage.src = sample.paths[FRAMES[upper].key];
    overlayImage.style.opacity = String(opacity);
    indicator.textContent = labelForValue(value);
  }

  slider.addEventListener('input', sync);
  sync();
}

function setupHero() {
  const heroSample = galleryData[3] || galleryData[0];
  if (!heroSample) return;
  document.getElementById('hero-title').textContent = heroSample.title;
  document.getElementById('hero-subtitle').textContent = heroSample.subtitle;
  bindBlendViewer({
    slider: document.getElementById('hero-slider'),
    baseImage: document.getElementById('hero-image-base'),
    overlayImage: document.getElementById('hero-image-overlay'),
    indicator: document.getElementById('hero-state'),
    sample: heroSample,
  });
}

function renderGallery() {
  const grid = document.getElementById('gallery-grid');
  const template = document.getElementById('interactive-card-template');

  galleryData.forEach((sample) => {
    const fragment = template.content.cloneNode(true);
    const title = fragment.querySelector('.card-title');
    const subtitle = fragment.querySelector('.card-subtitle');
    const baseImage = fragment.querySelector('.blend-base');
    const overlayImage = fragment.querySelector('.blend-overlay');
    const slider = fragment.querySelector('.blend-slider');
    const indicator = fragment.querySelector('.val-indicator');

    title.textContent = sample.title;
    subtitle.textContent = `${sample.subtitle} · ${sample.group}`;
    baseImage.alt = `${sample.title} sample`;
    overlayImage.alt = `${sample.title} blended enhancement overlay`;

    bindBlendViewer({ slider, baseImage, overlayImage, indicator, sample });
    grid.appendChild(fragment);
  });
}

setupHero();
renderGallery();
