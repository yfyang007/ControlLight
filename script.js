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

function frameIndexForValue(value) {
  return Math.max(0, Math.min(FRAMES.length - 1, Math.round((Number(value) / 100) * (FRAMES.length - 1))));
}

function bindNearestViewer({ slider, image, indicator, sample }) {
  function sync() {
    updateSliderFill(slider);
    const frame = FRAMES[frameIndexForValue(slider.value)];
    image.src = sample.paths[frame.key];
    indicator.textContent = frame.label;
  }

  slider.addEventListener('input', sync);
  sync();
}

function setupHero() {
  const heroSample = galleryData[3] || galleryData[0];
  if (!heroSample) return;
  document.getElementById('hero-title').textContent = heroSample.title;
  document.getElementById('hero-subtitle').textContent = heroSample.subtitle;
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

setupHero();
renderGallery();
