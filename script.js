const galleryData = window.PAPER_GALLERY_DATA || [];
const FRAMES = [
  { label: 'Input', key: '0' },
  { label: '0.25', key: '25' },
  { label: '0.50', key: '50' },
  { label: '0.75', key: '75' },
  { label: '1.00', key: '100' },
];

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

function bindNearestViewer({ slider, image, indicator, sample }) {
  if (!slider || !image || !indicator || !sample) return;

  function sync() {
    updateSliderFill(slider);
    const frame = nearestFrame(slider.value);
    image.src = sample.paths[frame.key];
    indicator.textContent = frame.label;
  }

  slider.addEventListener('input', sync);
  sync();
}

function setupHero() {
  const heroSample = galleryData[3] || galleryData[0];
  if (!heroSample) return;

  const title = document.getElementById('hero-title');
  const subtitle = document.getElementById('hero-subtitle');
  if (title) title.textContent = heroSample.title;
  if (subtitle) subtitle.textContent = heroSample.subtitle;

  bindNearestViewer({
    slider: document.getElementById('hero-slider'),
    image: document.getElementById('hero-image'),
    indicator: document.getElementById('hero-state'),
    sample: heroSample,
  });
}

function renderGallery() {
  const grid = document.getElementById('gallery-grid');
  const template = document.getElementById('interactive-card-template');
  if (!grid || !template) return;

  galleryData.forEach((sample) => {
    const fragment = template.content.cloneNode(true);
    const title = fragment.querySelector('.card-title');
    const subtitle = fragment.querySelector('.card-subtitle');
    const image = fragment.querySelector('.target-img');
    const slider = fragment.querySelector('.result-slider');
    const indicator = fragment.querySelector('.val-indicator');

    title.textContent = sample.title;
    subtitle.textContent = `${sample.subtitle} · ${sample.group}`;
    image.alt = `${sample.title} sample`;

    bindNearestViewer({ slider, image, indicator, sample });
    grid.appendChild(fragment);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  setupHero();
  renderGallery();
});
