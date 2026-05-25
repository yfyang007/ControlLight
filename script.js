const galleryData = window.PAPER_GALLERY_DATA || [];
const VALID_STEPS = {
  0: { label: 'Input', key: '0' },
  25: { label: 'Alpha 0.25', key: '25' },
  50: { label: 'Alpha 0.50', key: '50' },
  75: { label: 'Alpha 0.75', key: '75' },
  100: { label: 'Alpha 1.00', key: '100' },
};

function updateSliderFill(slider) {
  slider.style.setProperty('--pct', `${slider.value}%`);
}

function bindDiscreteViewer({ slider, image, blank, indicator, sample }) {
  function sync() {
    updateSliderFill(slider);
    const value = Number(slider.value);
    const step = VALID_STEPS[value];
    if (!step) {
      image.style.visibility = 'hidden';
      blank.classList.remove('hidden');
      indicator.textContent = 'Blank';
      return;
    }
    image.src = sample.paths[step.key];
    image.style.visibility = 'visible';
    blank.classList.add('hidden');
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
  bindDiscreteViewer({
    slider: document.getElementById('hero-slider'),
    image: document.getElementById('hero-image'),
    blank: document.getElementById('hero-blank'),
    indicator: document.getElementById('hero-state'),
    sample: heroSample,
  });
}

function renderGallery() {
  const grid = document.getElementById('gallery-grid');
  const template = document.getElementById('interactive-card-template');

  galleryData.forEach((sample) => {
    const fragment = template.content.cloneNode(true);
    const card = fragment.querySelector('.interactive-card');
    const title = fragment.querySelector('.card-title');
    const subtitle = fragment.querySelector('.card-subtitle');
    const image = fragment.querySelector('.target-img');
    const blank = fragment.querySelector('.card-blank');
    const slider = fragment.querySelector('.discrete-slider');
    const indicator = fragment.querySelector('.val-indicator');

    title.textContent = sample.title;
    subtitle.textContent = `${sample.subtitle} · ${sample.group}`;
    image.alt = `${sample.title} sample`;

    bindDiscreteViewer({ slider, image, blank, indicator, sample });
    grid.appendChild(fragment);
  });
}

setupHero();
renderGallery();
