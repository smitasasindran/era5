const PALETTE = [
  "#ffd6a5", "#caffbf", "#9bf6ff", "#bdb2ff", "#ffc6ff",
  "#fdffb6", "#a0c4ff", "#ffadad", "#c8f4de", "#e4c1f9",
  "#f1c0e8", "#b9fbc0",
];

const EXAMPLES = {
  en: "India, officially the Republic of India, is a country in South Asia. It is the seventh-largest country by area.",
  hi: "भारत एक विशाल देश है। यह दक्षिण एशिया में स्थित है और अनेक भाषाओं तथा संस्कृतियों का घर है।",
  te: "భారతదేశం దక్షిణాసియాలో ఒక పెద్ద దేశం. ఇది అనేక భాషలు మరియు సంస్కృతులకు నిలయం.",
  mr: "भारत हा दक्षिण आशियातील एक मोठा देश आहे. हा अनेक भाषा आणि संस्कृतींचे घर आहे.",
  mixed: "India / भारत / భారతదేశం / भारत — one country, many scripts. 2024 census: 1.4 billion+ people.",
};

let tokenizer = null;
let models = [];

const el = {
  input: document.getElementById("input-text"),
  tokenCount: document.getElementById("token-count"),
  charCount: document.getElementById("char-count"),
  wordCount: document.getElementById("word-count"),
  fertility: document.getElementById("fertility"),
  tokenView: document.getElementById("token-view"),
  vocabBadge: document.getElementById("vocab-badge"),
  themeToggle: document.getElementById("theme-toggle"),
  evalContent: document.getElementById("eval-content"),
  modelSelect: document.getElementById("model-select"),
  downloadLink: document.getElementById("download-link"),
};

// --- Theme toggle (persisted; explicit choice always overrides system pref) ---
const THEME_KEY = "tokenizer-playground-theme";

function effectiveTheme() {
  const explicit = document.documentElement.getAttribute("data-theme");
  if (explicit) return explicit;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  if (theme === "light" || theme === "dark") {
    document.documentElement.setAttribute("data-theme", theme);
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  el.themeToggle.textContent = effectiveTheme() === "dark" ? "☀️" : "\u{1F319}";
}

function toggleTheme() {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
}

el.themeToggle.addEventListener("click", toggleTheme);
applyTheme(localStorage.getItem(THEME_KEY));

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function currentViewMode() {
  return document.querySelector('input[name="view-mode"]:checked').value;
}

function render() {
  const text = el.input.value;
  el.charCount.textContent = text.length.toLocaleString();

  if (!tokenizer) return;

  if (text.length === 0) {
    el.tokenView.innerHTML = `<span class="placeholder">Tokens will appear here&hellip;</span>`;
    el.tokenCount.textContent = "0";
    el.wordCount.textContent = "0";
    el.fertility.textContent = "—";
    return;
  }

  const { atoms, gaps } = tokenizer.tokenize(text);
  const mode = currentViewMode();

  let totalTokens = 0;
  let colorIdx = 0;
  let html = "";

  // Real tokenizer behavior (this model has unk_token=None, verified against
  // Python): a piece with no vocab entry is silently dropped from the actual
  // token/id output entirely -- it does not become a placeholder token. We
  // still render it distinctly (dashed, uncolored) so characters this
  // tokenizer can't represent don't just silently vanish from the view.
  html += escapeHtml(gaps[0]);
  atoms.forEach((atom, i) => {
    for (const { piece, id } of atom.pieces) {
      const isUnk = id === null;

      if (isUnk) {
        html += `<span class="tok unk">${escapeHtml(piece)}</span>`;
        continue;
      }

      totalTokens += 1;
      const color = PALETTE[colorIdx % PALETTE.length];
      colorIdx += 1;

      if (mode === "hide") {
        html += escapeHtml(piece);
      } else if (mode === "ids") {
        html += `<span class="tok-id" style="background:${color}">${id}</span>`;
      } else {
        html += `<span class="tok" style="background:${color}">${escapeHtml(piece)}</span>`;
      }
    }
    html += escapeHtml(gaps[i + 1]);
  });

  el.tokenView.innerHTML = html;
  el.tokenCount.textContent = totalTokens.toLocaleString();
  el.wordCount.textContent = atoms.length.toLocaleString();
  el.fertility.textContent = atoms.length > 0 ? (totalTokens / atoms.length).toFixed(3) : "—";
}

let debounceTimer = null;
function scheduleRender() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(render, 80);
}

el.input.addEventListener("input", scheduleRender);
document.querySelectorAll('input[name="view-mode"]').forEach((r) => r.addEventListener("change", render));
document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    el.input.value = EXAMPLES[btn.dataset.lang];
    render();
  });
});

function renderEvalResults(summary) {
  const langs = ["en", "hi", "te", "mr"]; // fixed display order, independent of sort
  const headerCells = langs.map((l) => `<th>${summary.languages[l].name}</th>`).join("");
  const valueCells = langs
    .map((l) => `<td class="ratio">${summary.languages[l].ratio.toFixed(4)}</td>`)
    .join("");

  el.evalContent.innerHTML = `
    <table class="eval-table eval-table-cols">
      <thead><tr>${headerCells}</tr></thead>
      <tbody><tr>${valueCells}</tr></tbody>
    </table>
    <div class="eval-summary">
      <div class="stat">
        <div class="stat-label">Spread (max &minus; min)</div>
        <div class="stat-value">${summary.spread.toFixed(4)}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Score (1000 / spread)</div>
        <div class="stat-value">${summary.score.toFixed(1)}</div>
      </div>
    </div>
  `;
}

// --- Multi-version tokenizer support ---
// Available versions are listed in models/index.json, each pointing at its
// own folder (models/<id>/tokenizer.json + eval_results.json). Adding a new
// version is just: add a folder + one entry there, no code changes needed.
const SELECTED_MODEL_KEY = "tokenizer-playground-model";

async function loadModelById(id) {
  const model = models.find((m) => m.id === id);
  if (!model) {
    console.error(`Unknown model id "${id}"`);
    return;
  }

  el.vocabBadge.textContent = "loading…";
  el.downloadLink.setAttribute("aria-disabled", "true");

  try {
    tokenizer = await loadTokenizer(`${model.dir}/tokenizer.json`);
    el.vocabBadge.textContent = `vocab size: ${tokenizer.vocabSize.toLocaleString()}`;
    el.downloadLink.href = `${model.dir}/tokenizer.json`;
    el.downloadLink.download = `tokenizer_${model.id}.json`;
    el.downloadLink.removeAttribute("aria-disabled");
    if (!el.input.value) el.input.value = EXAMPLES.mixed;
    render();
  } catch (err) {
    tokenizer = null;
    el.vocabBadge.textContent = "failed to load tokenizer";
    el.tokenView.innerHTML = `<span class="placeholder">Error loading ${model.dir}/tokenizer.json: ${escapeHtml(
      String(err)
    )}</span>`;
    console.error(err);
  }

  try {
    const res = await fetch(`${model.dir}/eval_results.json`);
    if (!res.ok) throw new Error(`${res.status}`);
    renderEvalResults(await res.json());
  } catch (err) {
    el.evalContent.innerHTML = `<span class="placeholder">Could not load ${model.dir}/eval_results.json (${escapeHtml(
      String(err)
    )})</span>`;
    console.error(err);
  }

  localStorage.setItem(SELECTED_MODEL_KEY, id);
}

el.modelSelect.addEventListener("change", () => loadModelById(el.modelSelect.value));

async function init() {
  try {
    const res = await fetch("models/index.json");
    if (!res.ok) throw new Error(`${res.status}`);
    models = await res.json();
  } catch (err) {
    el.modelSelect.innerHTML = `<option>failed to load models/index.json</option>`;
    console.error(err);
    return;
  }

  el.modelSelect.innerHTML = models
    .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)}</option>`)
    .join("");

  const saved = localStorage.getItem(SELECTED_MODEL_KEY);
  const initialId = models.some((m) => m.id === saved) ? saved : models[0]?.id;
  if (!initialId) return;

  el.modelSelect.value = initialId;
  await loadModelById(initialId);
}

init();
