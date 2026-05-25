const galleryData = window.PAPER_GALLERY_DATA || [];
const STEPS = [
  { label: 'Input', key: '0' },
  { label: '0.25', key: '25' },
  { label: '0.50', key: '50' },
  { label: '0.75', key: '75' },
  { label: '1.00', key: '100' },
];

function updateSliderFill(slider) {
  const min = Number(slider.min || 0);
  const max = Number(slider.max || 4);
  const value = Number(slider.value || 0);
  const pct = ((value - min) / (max - min)) * 100;
  slider.style.setProperty('--pct', `${pct}%`);
}

function bindStepViewer({ slider, image, indicator, sample }) {
  function sync() {
    updateSliderFill(slider);
    const index = Math.max(0, Math.min(STEPS.length - 1, Number(slider.value)));
    const step = STEPS[index];
    image.src = sample.paths[step.key];
    indicator.textContent = step.label;
  }

  slider.addEventListener('input', sync);
  sync();
}

function setupHero() {
  const heroSample = galleryData[3] || galleryData[0];
  if (!heroSample) return;
  document.getElementById('hero-title').textContent = heroSample.title;
  document.getElementById('hero-subtitle').textContent = heroSample.subtitle;
  bindStepViewer({
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
    const slider = fragment.querySelector('.discrete-slider');
    const indicator = fragment.querySelector('.val-indicator');

    title.textContent = sample.title;
    subtitle.textContent = `${sample.subtitle} · ${sample.group}`;
    image.alt = `${sample.title} sample`;

    bindStepViewer({ slider, image, indicator, sample });
    grid.appendChild(fragment);
  });
}

setupHero();
renderGallery();
