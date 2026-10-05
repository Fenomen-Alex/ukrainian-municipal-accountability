const step1 = document.getElementById("step1");
const step2 = document.getElementById("step2");
const step3 = document.getElementById("step3");
const step4 = document.getElementById("step4");

const complaintTextEl = document.getElementById("complaintText");
const locationEl = document.getElementById("location");
const contactEl = document.getElementById("contact");
const analyzeBtn = document.getElementById("analyzeBtn");
const analyzeSpinner = document.getElementById("analyzeSpinner");
const error1 = document.getElementById("error1");

const topicsContainer = document.getElementById("topicsContainer");
const addTopicBtn = document.getElementById("addTopicBtn");
const toReviewBtn = document.getElementById("toReviewBtn");
const backToInputBtn = document.getElementById("backToInputBtn");
const error2 = document.getElementById("error2");

const reviewContainer = document.getElementById("reviewContainer");
const generateAppealBtn = document.getElementById("generateAppealBtn");
const backToStructBtn = document.getElementById("backToStructBtn");

const appealTextEl = document.getElementById("appealText");
const copyBtn = document.getElementById("copyBtn");
const openChannelBtn = document.getElementById("openChannelBtn");
const backToReviewBtn = document.getElementById("backToReviewBtn");
const copyConfirm = document.getElementById("copyConfirm");
const statusMsg = document.getElementById("statusMsg");
const channelInfo = document.getElementById("channelInfo");

const DEFAULT_CHANNEL = "https://www.kr-rada.gov.ua/";

let currentTopics = [];
let currentDraftId = null;
let channelUrl = DEFAULT_CHANNEL;

const storedChannel = localStorage.getItem("uma_channel_url");
if (storedChannel) {
  channelUrl = storedChannel;
}

function showError(el, msg) {
  el.textContent = msg;
  el.style.display = "block";
}

function hideError(el) {
  el.style.display = "none";
}

function setStep(n) {
  step1.style.display = n === 1 ? "block" : "none";
  step2.style.display = n === 2 ? "block" : "none";
  step3.style.display = n === 3 ? "block" : "none";
  step4.style.display = n === 4 ? "block" : "none";
}

function newId() {
  return "topic_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 6);
}

function emptyTopic() {
  return {
    id: newId(),
    domain: "other",
    issue: "",
    object: "",
    requested_action: "",
    attributes: {},
  };
}

function renderTopics() {
  topicsContainer.innerHTML = "";
  currentTopics.forEach((topic, idx) => {
    const div = document.createElement("div");
    div.className = "topic";
    div.innerHTML = `
      <h3>Тема ${idx + 1}</h3>
      <div class="field">
        <label for="domain-${topic.id}">Сфера (domain)</label>
        <select id="domain-${topic.id}" data-field="domain" data-id="${topic.id}" aria-label="Сфера">
          <option value="roads">roads</option>
          <option value="water">water</option>
          <option value="heating">heating</option>
          <option value="housing">housing</option>
          <option value="transport">transport</option>
          <option value="sanitation">sanitation</option>
          <option value="electricity">electricity</option>
          <option value="construction">construction</option>
          <option value="benefits">benefits</option>
          <option value="government">government</option>
          <option value="commerce">commerce</option>
          <option value="payments">payments</option>
          <option value="other">other</option>
        </select>
      </div>
      <div class="field">
        <label for="issue-${topic.id}">Проблема (issue)</label>
        <textarea id="issue-${topic.id}" data-field="issue" data-id="${topic.id}" placeholder="Конкретна проблема" aria-label="Проблема"></textarea>
      </div>
      <div class="field">
        <label for="object-${topic.id}">Об'єкт (object)</label>
        <input id="object-${topic.id}" data-field="object" data-id="${topic.id}" type="text" placeholder="Адреса, об'єкт чи порожньо" aria-label="Об'єкт" />
      </div>
      <div class="field">
        <label for="requested-${topic.id}">Прошу (requested_action)</label>
        <textarea id="requested-${topic.id}" data-field="requested_action" data-id="${topic.id}" placeholder="Що просите зробити" aria-label="Прошу"></textarea>
      </div>
      <div class="field">
        <label for="attrs-${topic.id}">Додаткові відомості (attributes)</label>
        <input id="attrs-${topic.id}" data-field="attrs" data-id="${topic.id}" type="text" placeholder="Наприклад: вулиця=Шевченка, будинок=24" aria-label="Додаткові відомості" />
      </div>
      <div class="actions">
        <button type="button" class="btn btn-danger btn-sm" data-action="remove" data-id="${topic.id}">Видалити тему</button>
      </div>
    `;
    topicsContainer.appendChild(div);
    const select = div.querySelector('[data-field="domain"]');
    if (select) select.value = topic.domain;
    div.querySelector('[data-field="issue"]').value = topic.issue;
    div.querySelector('[data-field="object"]').value = topic.object;
    div.querySelector('[data-field="requested_action"]').value = topic.requested_action;
    div.querySelector('[data-field="attrs"]').value = Object.entries(topic.attributes)
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");
  });

  topicsContainer.querySelectorAll("[data-action='remove']").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.id;
      currentTopics = currentTopics.filter((t) => t.id !== id);
      renderTopics();
    });
  });

  topicsContainer.querySelectorAll("[data-field]").forEach((el) => {
    el.addEventListener("input", (e) => {
      const target = e.target;
      const id = target.dataset.id;
      const field = target.dataset.field;
      const t = currentTopics.find((x) => x.id === id);
      if (!t) return;
      if (field === "domain") t.domain = target.value;
      if (field === "issue") t.issue = target.value;
      if (field === "object") t.object = target.value;
      if (field === "requested_action") t.requested_action = target.value;
      if (field === "attrs") {
        const attrs = {};
        target.value.split(",").forEach((pair) => {
          const p = pair.trim();
          if (!p) return;
          const eq = p.indexOf("=");
          if (eq > 0) {
            attrs[p.slice(0, eq).trim()] = p.slice(eq + 1).trim();
          }
        });
        t.attributes = attrs;
      }
    });
  });
}

function sanitizeTopicsForSchema() {
  return currentTopics.map((t) => ({
    domain: t.domain || "other",
    issue: t.issue || "",
    object: t.object || "",
    requested_action: t.requested_action || "",
    attributes: t.attributes || {},
  }));
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderReview() {
  const topics = sanitizeTopicsForSchema();
  let html = "";
  topics.forEach((topic, idx) => {
    html += `
      <div class="topic">
        <h3>Тема ${idx + 1}</h3>
        <p><strong>Сфера:</strong> ${escapeHtml(topic.domain)}</p>
        <p><strong>Проблема:</strong> ${escapeHtml(topic.issue) || "—"}</p>
        <p><strong>Об'єкт:</strong> ${escapeHtml(topic.object) || "—"}</p>
        <p><strong>Прошу:</strong> ${escapeHtml(topic.requested_action) || "—"}</p>
        <p><strong>Додаткові відомості:</strong></p>
        <ul>
          ${Object.entries(topic.attributes)
            .map(([k, v]) => `<li>${escapeHtml(k)}: ${escapeHtml(v)}</li>`)
            .join("")}
          ${Object.keys(topic.attributes).length === 0 ? "<li>—</li>" : ""}
        </ul>
      </div>
    `;
  });
  if (topics.length === 0) {
    html = '<div class="info">Жодної теми не визначено</div>';
  }
  html += `
    <div class="field">
      <label>Оригінальний опис</label>
      <textarea readonly aria-label="Оригінальний опис">${escapeHtml(complaintTextEl.value)}</textarea>
    </div>
    ${locationEl.value ? `<p><strong>Місце:</strong> ${escapeHtml(locationEl.value)}</p>` : ""}
    ${contactEl.value ? `<p><strong>Контакт:</strong> ${escapeHtml(contactEl.value)}</p>` : ""}
  `;
  reviewContainer.innerHTML = html;
}

analyzeBtn.addEventListener("click", async () => {
  hideError(error1);
  const text = complaintTextEl.value.trim();
  if (!text) {
    showError(error1, "Будь ласка, опишіть проблему");
    return;
  }
  if (text.length > 10000) {
    showError(error1, "Текст занадто довгий (максимум 10000 символів)");
    return;
  }
  analyzeBtn.disabled = true;
  analyzeSpinner.style.display = "inline-block";
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        location: locationEl.value || undefined,
        contact: contactEl.value || undefined,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Помилка аналізу");
    }
    const topics = data.structured.topics || [];
    currentTopics = topics.map((t) => ({ id: newId(), ...t, attributes: t.attributes || {} }));
    if (currentTopics.length === 0) {
      currentTopics = [emptyTopic()];
    }
    renderTopics();
    setStep(2);
  } catch (err) {
    showError(error1, err.message || "Помилка аналізу");
  } finally {
    analyzeBtn.disabled = false;
    analyzeSpinner.style.display = "none";
  }
});

addTopicBtn.addEventListener("click", () => {
  currentTopics.push(emptyTopic());
  renderTopics();
});

backToInputBtn.addEventListener("click", () => setStep(1));
toReviewBtn.addEventListener("click", () => {
  hideError(error2);
  renderReview();
  setStep(3);
});
backToStructBtn.addEventListener("click", () => setStep(2));

generateAppealBtn.addEventListener("click", () => {
  const topics = sanitizeTopicsForSchema();
  const context = {
    location: locationEl.value || undefined,
    contact: contactEl.value || undefined,
    channelUrl,
  };
  const lines = [];
  lines.push("Звернення громадянина до органу місцевого самоврядування");
  lines.push("");
  if (channelUrl) {
    lines.push(`Канал надсилання (офіційний): ${channelUrl}`);
    lines.push("");
  }
  if (topics.length === 0) {
    lines.push("Проблема не визначена.");
    lines.push("");
  } else {
    topics.forEach((topic, idx) => {
      lines.push(`${idx + 1}. Проблема: ${topic.issue || "—"}`);
      if (topic.object) lines.push(`   Об'єкт: ${topic.object}`);
      if (topic.domain) lines.push(`   Сфера: ${topic.domain}`);
      if (topic.requested_action) lines.push(`   Прошу: ${topic.requested_action}`);
      if (Object.keys(topic.attributes).length > 0) {
        lines.push("   Додаткові відомості:");
        for (const [k, v] of Object.entries(topic.attributes)) {
          if (v) lines.push(`     - ${k}: ${v}`);
        }
      }
      lines.push("");
    });
  }
  if (context.location) {
    lines.push(`Місце порушення: ${context.location}`);
    lines.push("");
  }
  if (context.contact) {
    lines.push(`Контактні дані для відповіді: ${context.contact}`);
    lines.push("");
  }
  lines.push(
    "Звернення підготовлене за допомогою асистента. Кожне твердження перевірено користувачем перед надсиланням.",
  );
  const appeal = lines.join("\n");
  appealTextEl.value = appeal;

  currentDraftId = currentDraftId || ("draft_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8));
  const draft = {
    id: currentDraftId,
    createdAt: new Date().toISOString(),
    originalText: complaintTextEl.value,
    location: locationEl.value || undefined,
    contact: contactEl.value || undefined,
    structured: { topics },
    finalAppeal: appeal,
    channelUrl,
    status: "ready",
  };
  try {
    const drafts = JSON.parse(localStorage.getItem("uma_drafts_v1") || "[]");
    const idx = drafts.findIndex((d) => d.id === draft.id);
    if (idx >= 0) drafts[idx] = draft;
    else drafts.unshift(draft);
    localStorage.setItem("uma_drafts_v1", JSON.stringify(drafts));
  } catch (e) {}

  statusMsg.textContent = "Чернетка збережена локально (готове до надсилання)";
  statusMsg.style.display = "block";
  channelInfo.textContent = `Офіційний канал: ${channelUrl}`;
  channelInfo.style.display = "block";
  setStep(4);
});

copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(appealTextEl.value);
    copyConfirm.style.display = "block";
    setTimeout(() => (copyConfirm.style.display = "none"), 2000);
  } catch (e) {
    alert("Не вдалося скопіювати текст");
  }
});

openChannelBtn.addEventListener("click", () => {
  try {
    const drafts = JSON.parse(localStorage.getItem("uma_drafts_v1") || "[]");
    const d = drafts.find((x) => x.id === currentDraftId);
    if (d) {
      d.status = "submitted_to_user";
      localStorage.setItem("uma_drafts_v1", JSON.stringify(drafts));
    }
  } catch (e) {}
  window.open(channelUrl, "_blank", "noopener,noreferrer");
});

backToReviewBtn.addEventListener("click", () => setStep(3));
