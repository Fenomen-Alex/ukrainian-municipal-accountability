const DOMAIN_LABELS = {
  roads: "Дороги",
  water: "Водопостачання",
  heating: "Опалення",
  housing: "Житло",
  transport: "Транспорт",
  sanitation: "Відходи та санітарія",
  electricity: "Електропостачання",
  construction: "Будівництво",
  benefits: "Соціальні виплати",
  government: "Державні послуги",
  commerce: "Торгівля",
  payments: "Платежі",
  other: "Інше",
};

const ATTRIBUTE_LABELS = {
  street: "Вулиця",
};

const DOMAIN_ORDER = [
  "roads",
  "water",
  "heating",
  "housing",
  "transport",
  "sanitation",
  "electricity",
  "construction",
  "benefits",
  "government",
  "commerce",
  "payments",
  "other",
];

const MAX_LEN = 10000;
const CLIENT_TIMEOUT_MS = 80000;
const DEFAULT_CHANNEL = "https://www.kr-rada.gov.ua/";

const step1 = document.getElementById("step1");
const step2 = document.getElementById("step2");
const step3 = document.getElementById("step3");
const step4 = document.getElementById("step4");

const complaintTextEl = document.getElementById("complaintText");
const locationEl = document.getElementById("location");
const contactEl = document.getElementById("contact");
const counterEl = document.getElementById("counter");
const analyzeBtn = document.getElementById("analyzeBtn");
const analyzeSpinner = document.getElementById("analyzeSpinner");
const analyzeStatus = document.getElementById("analyzeStatus");
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

const stepperItems = Array.from(document.querySelectorAll(".stepper-item"));

let currentTopics = [];
let currentDraftId = null;
let channelUrl = DEFAULT_CHANNEL;

try {
  const storedChannel = localStorage.getItem("uma_channel_url");
  if (storedChannel) channelUrl = storedChannel;
} catch (e) {
  // localStorage unavailable — keep default
}

function show(el, on) {
  el.classList.toggle("hidden", !on);
}

function showMessage(el, msg) {
  el.textContent = msg;
  show(el, true);
}

function hideMessage(el) {
  show(el, false);
  el.textContent = "";
}

function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function setStep(n, focusHeading = true) {
  [step1, step2, step3, step4].forEach((el, i) => show(el, i === n - 1));
  stepperItems.forEach((el, i) => {
    el.classList.toggle("is-active", i === n - 1);
    el.classList.toggle("is-done", i < n - 1);
    if (i === n - 1) el.setAttribute("aria-current", "step");
    else el.removeAttribute("aria-current");
  });
  const heading = document.querySelector("#step" + n + " .step-heading");
  if (heading && focusHeading) heading.focus({ preventScroll: true });
  window.scrollTo({
    top: 0,
    behavior: prefersReducedMotion() ? "auto" : "smooth",
  });
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

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
  }
  for (const child of children) {
    if (child == null) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function fieldBlock(labelText, control, hint) {
  const id = control.id;
  const wrap = el(
    "div",
    { class: "field" },
    el("label", { for: id, text: labelText }),
    control,
    hint ? el("div", { class: "hint", text: hint }) : null,
  );
  return wrap;
}

function parseAttributesInput(value) {
  const attrs = {};
  value.split(",").forEach((pair) => {
    const p = pair.trim();
    if (!p) return;
    const eq = p.indexOf("=");
    if (eq > 0) attrs[p.slice(0, eq).trim()] = p.slice(eq + 1).trim();
  });
  return attrs;
}

function attributesToText(attrs) {
  return Object.entries(attrs)
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
}

function renderTopics() {
  topicsContainer.textContent = "";

  currentTopics.forEach((topic, idx) => {
    const domainSelect = el("select", {
      id: "domain-" + topic.id,
      "data-field": "domain",
      "data-id": topic.id,
      "aria-label": "Сфера",
    });
    for (const d of DOMAIN_ORDER) {
      const opt = el("option", { value: d, text: DOMAIN_LABELS[d] });
      domainSelect.append(opt);
    }
    domainSelect.value = topic.domain;

    const issueTa = el("textarea", {
      id: "issue-" + topic.id,
      "data-field": "issue",
      "data-id": topic.id,
      placeholder: "Конкретна проблема",
      "aria-label": "Проблема",
    });
    issueTa.value = topic.issue;

    const objectInput = el("input", {
      id: "object-" + topic.id,
      "data-field": "object",
      "data-id": topic.id,
      type: "text",
      placeholder: "Адреса, об’єкт чи порожньо",
      "aria-label": "Об’єкт",
    });
    objectInput.value = topic.object;

    const requestedTa = el("textarea", {
      id: "requested-" + topic.id,
      "data-field": "requested_action",
      "data-id": topic.id,
      placeholder: "Що просите зробити",
      "aria-label": "Прошу",
    });
    requestedTa.value = topic.requested_action;

    const attrsInput = el("input", {
      id: "attrs-" + topic.id,
      "data-field": "attrs",
      "data-id": topic.id,
      type: "text",
      placeholder: "Наприклад: вулиця=Шевченка, будинок=24",
      "aria-label": "Додаткові відомості",
    });
    attrsInput.value = attributesToText(topic.attributes);

    const removeBtn = el("button", {
      type: "button",
      class: "btn btn-danger btn-sm",
      "data-action": "remove",
      "data-id": topic.id,
      text: "Видалити тему",
    });

    const card = el(
      "div",
      { class: "topic" },
      el("h3", { text: "Тема " + (idx + 1) }),
      fieldBlock("Сфера", domainSelect),
      fieldBlock("Проблема", issueTa),
      fieldBlock("Об’єкт", objectInput),
      fieldBlock("Прошу", requestedTa),
      fieldBlock("Додаткові відомості", attrsInput),
      el("div", { class: "actions" }, removeBtn),
    );
    topicsContainer.append(card);

    removeBtn.addEventListener("click", () => {
      currentTopics = currentTopics.filter((t) => t.id !== topic.id);
      renderTopics();
    });

    card.querySelectorAll("[data-field]").forEach((input) => {
      input.addEventListener("input", () => {
        const t = currentTopics.find((x) => x.id === topic.id);
        if (!t) return;
        const field = input.dataset.field;
        if (field === "domain") t.domain = input.value;
        else if (field === "issue") t.issue = input.value;
        else if (field === "object") t.object = input.value;
        else if (field === "requested_action") t.requested_action = input.value;
        else if (field === "attrs") t.attributes = parseAttributesInput(input.value);
      });
      if (input.tagName === "SELECT") {
        input.addEventListener("change", () => {
          const t = currentTopics.find((x) => x.id === topic.id);
          if (t) t.domain = input.value;
        });
      }
    });
  });
}

function sanitizeTopicsForSchema() {
  return currentTopics.map((t) => ({
    domain: DOMAIN_ORDER.includes(t.domain) ? t.domain : "other",
    issue: t.issue || "",
    object: t.object || "",
    requested_action: t.requested_action || "",
    attributes: t.attributes || {},
  }));
}

function renderReview() {
  const topics = sanitizeTopicsForSchema();
  reviewContainer.textContent = "";

  if (topics.length === 0) {
    reviewContainer.append(el("div", { class: "info", text: "Жодної теми не визначено" }));
  }

  topics.forEach((topic, idx) => {
    const attrList = el("ul", { class: "attrs-list" });
    const entries = Object.entries(topic.attributes);
    if (entries.length === 0) {
      attrList.append(el("li", { text: "—" }));
    } else {
      for (const [k, v] of entries) attrList.append(el("li", { text: k + ": " + v }));
    }

    reviewContainer.append(
      el(
        "div",
        { class: "topic" },
        el("h3", { text: "Тема " + (idx + 1) }),
        el("p", {}, el("strong", { text: "Сфера: " }), DOMAIN_LABELS[topic.domain] || topic.domain),
        el("p", {}, el("strong", { text: "Проблема: " }), topic.issue || "—"),
        el("p", {}, el("strong", { text: "Об’єкт: " }), topic.object || "—"),
        el("p", {}, el("strong", { text: "Прошу: " }), topic.requested_action || "—"),
        el("p", {}, el("strong", { text: "Додаткові відомості:" })),
        attrList,
      ),
    );
  });

  const original = el("textarea", { id: "reviewOriginal", readonly: "", "aria-label": "Оригінальний опис" });
  original.value = complaintTextEl.value;
  reviewContainer.append(
    el(
      "div",
      { class: "field" },
      el("label", { for: "reviewOriginal", text: "Оригінальний опис" }),
      original,
    ),
  );

  if (locationEl.value.trim()) {
    reviewContainer.append(
      el("p", {}, el("strong", { text: "Місце: " }), locationEl.value.trim()),
    );
  }
  if (contactEl.value.trim()) {
    reviewContainer.append(
      el("p", {}, el("strong", { text: "Контакт: " }), contactEl.value.trim()),
    );
  }
}

function updateCounter() {
  const len = complaintTextEl.value.length;
  counterEl.textContent = `${len} / ${MAX_LEN}`;
  counterEl.classList.toggle("counter--over", len > MAX_LEN);
}

complaintTextEl.addEventListener("input", updateCounter);
updateCounter();

const ANALYZE_ERRORS = {
  validation: "Перевірте введені дані та спробуйте ще раз.",
  rateLimit: "Забагато запитів. Зачекайте хвилину та спробуйте ще раз.",
  timeout: "Сервіс не відповів вчасно. Спробуйте ще раз.",
  network: "Немає з’єднання з сервером. Перевірте інтернет та спробуйте ще раз.",
  server: "Не вдалося підготувати структуру. Спробуйте ще раз трохи пізніше.",
};

async function analyzeComplaint() {
  hideMessage(error1);
  const text = complaintTextEl.value.trim();
  if (!text) {
    showMessage(error1, "Будь ласка, опишіть проблему");
    return;
  }
  if (text.length > MAX_LEN) {
    showMessage(error1, `Текст занадто довгий (максимум ${MAX_LEN} символів)`);
    return;
  }

  analyzeBtn.disabled = true;
  show(analyzeSpinner, true);
  show(analyzeStatus, true);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CLIENT_TIMEOUT_MS);
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        location: locationEl.value.trim() || undefined,
        contact: contactEl.value.trim() || undefined,
      }),
      signal: controller.signal,
    });

    let data = null;
    try {
      data = await res.json();
    } catch (e) {
      data = null;
    }

    if (res.status === 429) {
      showMessage(error1, (data && data.error) || ANALYZE_ERRORS.rateLimit);
      return;
    }
    if (res.status === 400) {
      const detail = data && data.details && data.details[0] && data.details[0].message;
      showMessage(error1, detail || (data && data.error) || ANALYZE_ERRORS.validation);
      return;
    }
    if (!res.ok || !data || !data.structured) {
      showMessage(error1, ANALYZE_ERRORS.server);
      return;
    }

    const topics = data.structured.topics || [];
    currentTopics = topics.map((t) => ({
      id: newId(),
      domain: DOMAIN_ORDER.includes(t.domain) ? t.domain : "other",
      issue: t.issue || "",
      object: t.object || "",
      requested_action: t.requested_action || "",
      attributes: t.attributes || {},
    }));
    if (currentTopics.length === 0) currentTopics = [emptyTopic()];
    renderTopics();
    setStep(2);
  } catch (err) {
    if (err && err.name === "AbortError") {
      showMessage(error1, ANALYZE_ERRORS.timeout);
    } else {
      showMessage(error1, ANALYZE_ERRORS.network);
    }
  } finally {
    clearTimeout(timer);
    analyzeBtn.disabled = false;
    show(analyzeSpinner, false);
    show(analyzeStatus, false);
  }
}

analyzeBtn.addEventListener("click", analyzeComplaint);

addTopicBtn.addEventListener("click", () => {
  currentTopics.push(emptyTopic());
  renderTopics();
});

backToInputBtn.addEventListener("click", () => setStep(1));

toReviewBtn.addEventListener("click", () => {
  hideMessage(error2);
  if (currentTopics.length === 0) {
    showMessage(error2, "Додайте хоча б одну тему");
    return;
  }
  const anyContent = currentTopics.some(
    (t) => (t.issue && t.issue.trim()) || (t.requested_action && t.requested_action.trim()),
  );
  if (!anyContent) {
    showMessage(error2, "Заповніть хоча б опис проблеми в одній темі");
    return;
  }
  renderReview();
  setStep(3);
});

backToStructBtn.addEventListener("click", () => setStep(2));

function buildAppeal(topics, context) {
  const lines = [];
  lines.push("Звернення громадянина до органу місцевого самоврядування");
  lines.push("");
  lines.push("До: ______________________________________ (орган місцевого самоврядування)");
  lines.push("");
  if (topics.length === 0) {
    lines.push("Проблема не визначена.");
    lines.push("");
  } else {
    topics.forEach((topic, idx) => {
      lines.push(`${idx + 1}. Проблема: ${topic.issue || "—"}`);
      if (topic.object) lines.push(`   Об'єкт: ${topic.object}`);
      if (topic.domain) lines.push(`   Сфера: ${DOMAIN_LABELS[topic.domain] || topic.domain}`);
      if (topic.requested_action) lines.push(`   Прошу: ${topic.requested_action}`);
      if (Object.keys(topic.attributes).length > 0) {
        lines.push("   Додаткові відомості:");
        for (const [k, v] of Object.entries(topic.attributes)) {
          if (v) lines.push(`     - ${ATTRIBUTE_LABELS[k] || k}: ${v}`);
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
  lines.push("Дата: ____ . ____ . 20___ р.          Підпис: ______________");
  lines.push("");
  lines.push("———");
  if (context.channelUrl) {
    lines.push(`Канал надсилання (офіційний): ${context.channelUrl}`);
  }
  lines.push(
    "Звернення підготовлене за допомогою асистента. Кожне твердження перевірено користувачем перед надсиланням.",
  );
  return lines.join("\n");
}

generateAppealBtn.addEventListener("click", () => {
  const topics = sanitizeTopicsForSchema();
  const context = {
    location: locationEl.value.trim() || undefined,
    contact: contactEl.value.trim() || undefined,
    channelUrl,
  };
  const appeal = buildAppeal(topics, context);
  appealTextEl.value = appeal;

  currentDraftId =
    currentDraftId ||
    "draft_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
  const draft = {
    id: currentDraftId,
    createdAt: new Date().toISOString(),
    originalText: complaintTextEl.value,
    location: context.location,
    contact: context.contact,
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
  } catch (e) {
    // storage unavailable — draft simply not persisted
  }

  statusMsg.textContent = "Чернетка збережена локально (готове до надсилання)";
  show(statusMsg, true);
  channelInfo.textContent = `Офіційний канал: ${channelUrl}`;
  show(channelInfo, true);
  setStep(4);
});

copyBtn.addEventListener("click", async () => {
  const text = appealTextEl.value;
  show(copyConfirm, false);
  try {
    await navigator.clipboard.writeText(text);
    copyConfirm.textContent = "Текст скопійовано в буфер обміну";
    show(copyConfirm, true);
    setTimeout(() => show(copyConfirm, false), 3000);
  } catch (e) {
    appealTextEl.focus();
    appealTextEl.select();
    copyConfirm.textContent =
      "Не вдалося скопіювати автоматично — виділений текст можна скопіювати вручну (Ctrl/Cmd+C)";
    show(copyConfirm, true);
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
  } catch (e) {
    // ignore storage errors
  }
  window.open(channelUrl, "_blank", "noopener,noreferrer");
});

backToReviewBtn.addEventListener("click", () => setStep(3));

setStep(1, false);
