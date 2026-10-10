// NextWord AI Client Application Logic

document.addEventListener("DOMContentLoaded", () => {
  const promptBox = document.getElementById("prompt");
  const language = document.getElementById("language");
  const seqLength = document.getElementById("seqLength");
  const seqVal = document.getElementById("seqVal");
  const predictBtn = document.getElementById("predict");
  const clearBtn = document.getElementById("clear");
  const statusBox = document.getElementById("status");
  const suggestionsBox = document.getElementById("suggestions");
  const sequenceBox = document.getElementById("sequenceBox");
  const sequenceText = document.getElementById("sequenceText");
  const samplePills = document.querySelectorAll(".pill");

  // Modal elements
  const addSentenceBtn = document.getElementById("addSentenceBtn");
  const addModal = document.getElementById("addModal");
  const closeModal = document.getElementById("closeModal");
  const sentenceForm = document.getElementById("sentenceForm");
  const modalLang = document.getElementById("modalLang");
  const newSentenceInput = document.getElementById("newSentenceInput");
  const retrainBtn = document.getElementById("retrainBtn");

  let debounceTimer = null;

  // Update slider display value
  if (seqLength && seqVal) {
    seqLength.addEventListener("input", () => {
      seqVal.textContent = seqLength.value;
      triggerPrediction();
    });
  }

  // Quick sample pill clicks
  samplePills.forEach(pill => {
    pill.addEventListener("click", () => {
      promptBox.value = pill.dataset.text || pill.textContent;
      if (pill.dataset.lang) {
        language.value = pill.dataset.lang;
      }
      triggerPrediction();
    });
  });

  // Debounced auto-prediction on typing
  promptBox?.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      triggerPrediction();
    }, 250);
  });

  language?.addEventListener("change", triggerPrediction);
  predictBtn?.addEventListener("click", triggerPrediction);

  clearBtn?.addEventListener("click", () => {
    if (promptBox) promptBox.value = "";
    if (suggestionsBox) suggestionsBox.innerHTML = "";
    if (sequenceBox) sequenceBox.style.display = "none";
    if (statusBox) statusBox.innerHTML = `<span>Ready</span>`;
    promptBox?.focus();
  });

  async function triggerPrediction() {
    const text = promptBox.value.trim();
    if (!text) {
      suggestionsBox.innerHTML = `<p style="color:var(--text-subtle);font-size:0.9rem;">Start typing above to see neural predictions...</p>`;
      if (sequenceBox) sequenceBox.style.display = "none";
      if (statusBox) statusBox.innerHTML = `<span>Ready</span>`;
      return;
    }

    if (predictBtn) predictBtn.disabled = true;
    if (statusBox) statusBox.innerHTML = `<span>Computing predictions…</span>`;

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: promptBox.value, // send full text preserving trailing spaces
          language: language.value,
          sequence_length: seqLength ? parseInt(seqLength.value) : 1,
          count: 5
        })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Prediction request failed");

      // Update status
      if (statusBox) {
        statusBox.innerHTML = `
          <span>Engine: <strong style="color:#fff;">${data.mode}</strong></span>
          <span class="mode-badge">${language.value.toUpperCase()}</span>
        `;
      }

      // Render probability bars
      renderSuggestions(data.suggestions || []);

      // Render multi-word sequence output
      if (data.sequence && sequenceBox && sequenceText) {
        sequenceText.textContent = data.sequence;
        sequenceBox.style.display = "block";
      } else if (sequenceBox) {
        sequenceBox.style.display = "none";
      }

    } catch (err) {
      if (statusBox) statusBox.innerHTML = `<span style="color:#f87171;">${err.message}</span>`;
    } finally {
      if (predictBtn) predictBtn.disabled = false;
    }
  }

  function renderSuggestions(suggestions) {
    if (!suggestionsBox) return;
    suggestionsBox.innerHTML = "";

    if (suggestions.length === 0) {
      suggestionsBox.innerHTML = `<p style="color:var(--text-subtle);font-size:0.9rem;">No matching words found. Try another phrase or train on more corpus text.</p>`;
      return;
    }

    suggestions.forEach((item, index) => {
      const div = document.createElement("div");
      div.className = "prob-item";

      const fill = document.createElement("div");
      fill.className = "prob-fill";
      fill.style.width = `${Math.min(item.confidence, 100)}%`;

      const content = document.createElement("div");
      content.className = "prob-content";

      const wordSpan = document.createElement("span");
      wordSpan.className = "prob-word";

      if (index < 3) {
        const keyBadge = document.createElement("span");
        keyBadge.className = "prob-key";
        keyBadge.textContent = `${index + 1}`;
        wordSpan.appendChild(keyBadge);
      }

      const textNode = document.createTextNode(item.word);
      wordSpan.appendChild(textNode);

      const valSpan = document.createElement("span");
      valSpan.className = "prob-val";
      valSpan.textContent = `${item.confidence}%`;

      content.appendChild(wordSpan);
      content.appendChild(valSpan);

      div.appendChild(fill);
      div.appendChild(content);

      div.addEventListener("click", () => {
        appendWord(item.word);
      });

      suggestionsBox.appendChild(div);
    });
  }

  function appendWord(word) {
    if (!promptBox) return;
    const rawText = promptBox.value;
    const endsWithSpace = rawText.endsWith(" ");
    const trimmed = rawText.trimEnd();
    const words = trimmed.split(/\s+/);

    if (words.length > 0 && !endsWithSpace) {
      const lastToken = words[words.length - 1].toLowerCase();
      // If the suggested word completes the typed prefix (e.g. 'going' completes 'goi')
      if (word.toLowerCase().startsWith(lastToken)) {
        words[words.length - 1] = word;
        promptBox.value = words.join(" ") + " ";
        promptBox.focus();
        triggerPrediction();
        return;
      }
    }

    promptBox.value = `${trimmed} ${word} `;
    promptBox.focus();
    triggerPrediction();
  }

  // Keyboard Shortcuts: Press 1, 2, 3 to pick suggestions
  document.addEventListener("keydown", (e) => {
    if (document.activeElement === newSentenceInput) return;
    if (["1", "2", "3"].includes(e.key) && !e.ctrlKey && !e.metaKey) {
      const items = suggestionsBox?.querySelectorAll(".prob-item");
      const idx = parseInt(e.key) - 1;
      if (items && items[idx]) {
        e.preventDefault();
        items[idx].click();
      }
    }
  });

  // Modal Handlers
  if (addSentenceBtn && addModal) {
    addSentenceBtn.addEventListener("click", () => addModal.classList.add("active"));
  }
  if (closeModal && addModal) {
    closeModal.addEventListener("click", () => addModal.classList.remove("active"));
  }

  sentenceForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const sentence = newSentenceInput.value.trim();
    const lang = modalLang.value;
    if (!sentence) return;

    try {
      const res = await fetch("/api/add-sentence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence, language: lang })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      alert(data.message);
      newSentenceInput.value = "";
      addModal.classList.remove("active");
    } catch (err) {
      alert(err.message);
    }
  });

  retrainBtn?.addEventListener("click", async () => {
    retrainBtn.disabled = true;
    retrainBtn.textContent = "Training in progress...";
    try {
      const res = await fetch("/api/train", { method: "POST" });
      const data = await res.json();
      alert(data.message);
    } catch (err) {
      alert("Failed to start training.");
    } finally {
      setTimeout(() => {
        retrainBtn.disabled = false;
        retrainBtn.textContent = "⚡ Retrain Neural Models";
      }, 5000);
    }
  });
});
