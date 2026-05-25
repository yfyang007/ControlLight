const galleryData = window.PAPER_GALLERY_DATA || [];

function applyInteractiveFilter(card, sample) {
  const slider = card.querySelector(".apple-slider");
  const image = card.querySelector(".target-img");
  const indicator = card.querySelector(".val-indicator");

  const { baseBrightness, maxBrightness, contrastBoost, saturateBoost } = sample.strength;

  function updateState() {
    const percent = Number(slider.value);
    const s = percent / 100;
    slider.style.setProperty("--pct", `${percent}%`);

    const brightness = baseBrightness + s * (maxBrightness - baseBrightness);
    const contrast = 0.74 + s * contrastBoost;
    const saturation = 0.26 + s * saturateBoost;
    const sepia = 0.16 * (1 - s);

    image.style.filter = `brightness(${brightness}) contrast(${contrast}) saturate(${saturation}) sepia(${sepia})`;
    indicator.textContent = s === 0 ? "s = 0.00 (Input)" : `s = ${s.toFixed(2)}`;
  }

  slider.addEventListener("input", updateState);
  updateState();
}

function renderGallery() {
  const grid = document.getElementById("gallery-grid");
  const template = document.getElementById("interactive-card-template");

  galleryData.forEach((sample) => {
    const fragment = template.content.cloneNode(true);
    const card = fragment.querySelector(".interactive-card");
    const title = fragment.querySelector(".card-title");
    const subtitle = fragment.querySelector(".card-subtitle");
    const image = fragment.querySelector(".target-img");

    title.textContent = sample.title;
    subtitle.textContent = `${sample.subtitle} · ${sample.split}`;
    image.src = sample.image;
    image.alt = `${sample.title} sample`;

    applyInteractiveFilter(card, sample);
    grid.appendChild(fragment);
  });
}

renderGallery();
