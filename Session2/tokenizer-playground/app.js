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

const el = {
  input: document.getElementById("input-text"),
  tokenCount: document.getElementById("token-count"),
  charCount: document.getElementById("char-count"),
  fertility: document.getElementById("fertility"),
  tokenView: document.getElementById("token-view"),
  vocabBadge: document.getElementById("vocab-badge"),
};

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

async function init() {
  try {
    tokenizer = await loadTokenizer("tokenizer.json");
    el.vocabBadge.textContent = `vocab size: ${tokenizer.vocabSize.toLocaleString()}`;
    el.input.value = EXAMPLES.mixed;
    render();
  } catch (err) {
    el.vocabBadge.textContent = "failed to load tokenizer.json";
    el.tokenView.innerHTML = `<span class="placeholder">Error: ${escapeHtml(String(err))}</span>`;
    console.error(err);
  }
}

init();
