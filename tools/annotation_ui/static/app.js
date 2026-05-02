const state = {
  config: null,
  records: [],
  filteredIndices: [],
  filteredPos: 0,
  currentRecord: null,
};

const elements = {
  datasetInfo: document.getElementById("datasetInfo"),
  saveStatus: document.getElementById("saveStatus"),
  categoryFilter: document.getElementById("categoryFilter"),
  statusFilter: document.getElementById("statusFilter"),
  progressText: document.getElementById("progressText"),
  progressPct: document.getElementById("progressPct"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  saveBtn: document.getElementById("saveBtn"),
  cropImage: document.getElementById("cropImage"),
  metadataGrid: document.getElementById("metadataGrid"),
  tagsChecklist: document.getElementById("tagsChecklist"),
  customTagsInput: document.getElementById("customTagsInput"),
  shortEn: document.getElementById("shortEn"),
  snippetEn: document.getElementById("snippetEn"),
  notes: document.getElementById("notes"),
  visibilityButtons: Array.from(document.querySelectorAll(".visibility-btn")),
};

const READONLY_FIELDS = [
  "record_id",
  "image_id",
  "box_id",
  "bbox_xywh",
  "source",
  "split",
  "category_name",
  "label_version",
  "crop_path",
  "completed",
];

const TAG_TRANSLATIONS_RU = {
  intact_structure: "целостная структура",
  regular_disc_shape: "регулярная форма дисков",
  no_visible_break: "нет видимых разрушений",
  no_visible_burn_mark: "нет видимых следов подгорания",
  clean_surface: "чистая поверхность",
  uniform_appearance: "однородный внешний вид",
  dark_surface_trace: "тёмный след на поверхности",
  burn_like_mark: "след, похожий на подгорание",
  surface_stain: "пятно/загрязнение поверхности",
  localized_darkening: "локальное потемнение",
  flashover_like_trace: "след, похожий на flashover",
  surface_damage_mark: "след поверхностного повреждения",
  missing_fragment: "отсутствует фрагмент (скол)",
  edge_discontinuity: "нарушение края",
  broken_profile: "нарушенный профиль",
  structural_gap: "структурный разрыв",
  shape_irregularity: "нерегулярная форма",
  material_loss: "утрата материала",
  low_contrast: "низкий контраст",
  blurred_region: "размытая область",
  partial_view: "частичный обзор объекта",
  occluded_region: "область закрыта/перекрыта",
  unclear_boundary: "нечёткая граница",
  ambiguous_evidence: "неоднозначные визуальные признаки",
};

const TAG_GROUPS = [
  {
    id: "visibility_quality",
    title: "Видимость / качество",
    tags: [
      "blurred_region",
      "low_contrast",
      "occluded_region",
      "partial_view",
      "unclear_boundary",
      "ambiguous_evidence",
    ],
  },
  {
    id: "normal_state",
    title: "Нормальное состояние",
    tags: [
      "intact_structure",
      "clean_surface",
      "uniform_appearance",
      "regular_disc_shape",
      "no_visible_break",
      "no_visible_burn_mark",
    ],
  },
  {
    id: "damage",
    title: "Повреждения",
    tags: [
      "missing_fragment",
      "material_loss",
      "edge_discontinuity",
      "broken_profile",
      "structural_gap",
      "shape_irregularity",
      "burn_like_mark",
      "flashover_like_trace",
      "dark_surface_trace",
      "localized_darkening",
      "surface_damage_mark",
      "surface_stain",
    ],
  },
];

const TAG_PAIR_TOOLTIPS = {
  missing_fragment:
    "Подсказка: missing_fragment — когда явно отсутствует фрагмент.",
  material_loss:
    "Подсказка: material_loss — для износа/эрозии без явно отделённого фрагмента.",
  burn_like_mark:
    "Подсказка: burn_like_mark — общий след, похожий на подгорание.",
  flashover_like_trace:
    "Подсказка: flashover_like_trace — когда паттерн похож именно на flashover.",
  dark_surface_trace:
    "Tip (dark_surface_trace vs localized_darkening): dark_surface_trace for streak/trace-like pattern.",
  localized_darkening:
    "Tip (dark_surface_trace vs localized_darkening): localized_darkening for compact local dark patch, not a long trace.",
};

let activeVisibility = "";

function setStatus(message, isError = false) {
  elements.saveStatus.textContent = message;
  elements.saveStatus.classList.toggle("error", isError);
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return await res.json();
}

function formatValue(value) {
  if (Array.isArray(value)) {
    return JSON.stringify(value);
  }
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function buildMetadata(record) {
  const metadataRecord = {
    ...record,
    completed: record._completed ? "yes" : "no",
  };
  elements.metadataGrid.innerHTML = "";
  for (const key of READONLY_FIELDS) {
    const keyEl = document.createElement("div");
    keyEl.className = "metadata-key";
    keyEl.textContent = key;

    const valEl = document.createElement("div");
    valEl.className = "metadata-value";
    valEl.textContent = formatValue(metadataRecord[key]);

    elements.metadataGrid.appendChild(keyEl);
    elements.metadataGrid.appendChild(valEl);
  }
}

function createTagItem(tag, selectedTags) {
  const wrap = document.createElement("label");
  wrap.className = "tag-item";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.value = tag;
  checkbox.checked = selectedTags.has(tag);

  const textWrap = document.createElement("div");
  textWrap.className = "tag-text-wrap";
  const en = document.createElement("span");
  en.className = "tag-en";
  en.textContent = tag;
  const ru = document.createElement("span");
  ru.className = "tag-ru";
  ru.textContent = `- ${TAG_TRANSLATIONS_RU[tag] || "перевод не задан"}`;
  textWrap.appendChild(en);
  textWrap.appendChild(ru);

  const tooltipParts = [];
  if (TAG_PAIR_TOOLTIPS[tag]) {
    tooltipParts.push(TAG_PAIR_TOOLTIPS[tag]);
  }
  if (TAG_TRANSLATIONS_RU[tag]) {
    tooltipParts.push(`RU: ${TAG_TRANSLATIONS_RU[tag]}`);
  }
  if (tooltipParts.length > 0) {
    wrap.title = tooltipParts.join("\n");
  }

  wrap.appendChild(checkbox);
  wrap.appendChild(textWrap);
  return wrap;
}

function buildTagChecklist(options, selectedTags) {
  elements.tagsChecklist.innerHTML = "";
  const available = new Set(options);
  const used = new Set();

  for (const group of TAG_GROUPS) {
    const groupTags = group.tags.filter((tag) => available.has(tag));
    if (groupTags.length === 0) continue;

    const section = document.createElement("section");
    section.className = "tag-section";
    const heading = document.createElement("h4");
    heading.className = "tag-section-title";
    heading.textContent = group.title;
    section.appendChild(heading);

    const grid = document.createElement("div");
    grid.className = "tag-group-grid";
    for (const tag of groupTags) {
      used.add(tag);
      grid.appendChild(createTagItem(tag, selectedTags));
    }
    section.appendChild(grid);
    elements.tagsChecklist.appendChild(section);
  }

  const otherTags = [...available].filter((tag) => !used.has(tag)).sort();
  if (otherTags.length > 0) {
    const section = document.createElement("section");
    section.className = "tag-section";
    const heading = document.createElement("h4");
    heading.className = "tag-section-title";
    heading.textContent = "Other";
    section.appendChild(heading);

    const grid = document.createElement("div");
    grid.className = "tag-group-grid";
    for (const tag of otherTags) {
      grid.appendChild(createTagItem(tag, selectedTags));
    }
    section.appendChild(grid);
    elements.tagsChecklist.appendChild(section);
  }
}

function setVisibility(value) {
  activeVisibility = value;
  for (const btn of elements.visibilityButtons) {
    btn.classList.toggle("active", btn.dataset.value === value);
  }
}

function populateForm(record) {
  buildMetadata(record);
  const tags = Array.isArray(record.visual_evidence_tags) ? record.visual_evidence_tags : [];
  const selected = new Set(tags);
  buildTagChecklist(state.config.tag_options, selected);

  const customTags = tags.filter((tag) => !state.config.tag_options.includes(tag));
  elements.customTagsInput.value = customTags.join(", ");

  elements.shortEn.value =
    record.short_canonical_description_en ||
    record.short_canonical_description ||
    "";
  elements.snippetEn.value =
    record.report_snippet_en ||
    record.report_snippet ||
    "";
  elements.notes.value = record.annotator_notes || "";
  setVisibility(record.visibility || "");

  const imageUrl = `/api/image?crop_path=${encodeURIComponent(record.crop_path || "")}`;
  elements.cropImage.src = imageUrl;
}

function collectTags() {
  const checked = Array.from(
    elements.tagsChecklist.querySelectorAll('input[type="checkbox"]:checked'),
  ).map((input) => input.value);

  const custom = elements.customTagsInput.value
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);

  const merged = [];
  const seen = new Set();
  for (const tag of [...checked, ...custom]) {
    if (seen.has(tag)) continue;
    seen.add(tag);
    merged.push(tag);
  }
  return merged;
}

function collectPayload() {
  return {
    visual_evidence_tags: collectTags(),
    visibility: activeVisibility,
    short_canonical_description_en: elements.shortEn.value,
    report_snippet_en: elements.snippetEn.value,
    annotator_notes: elements.notes.value,
  };
}

function getFilteredCount() {
  return state.filteredIndices.length;
}

function updateProgressDisplay() {
  const total = getFilteredCount();
  if (total === 0) {
    elements.progressText.textContent = "0 / 0";
    elements.progressPct.textContent = "0%";
    return;
  }
  elements.progressText.textContent = `${state.filteredPos + 1} / ${total}`;

  const completedCount = state.records
    .filter((rec) => state.filteredIndices.includes(rec.index))
    .filter((rec) => rec.completed).length;
  const pct = Math.round((completedCount / total) * 100);
  elements.progressPct.textContent = `${pct}% completed`;
}

async function loadRecordByFilteredPos(pos) {
  if (state.filteredIndices.length === 0) {
    state.currentRecord = null;
    elements.metadataGrid.innerHTML = "";
    elements.cropImage.removeAttribute("src");
    updateProgressDisplay();
    return;
  }
  state.filteredPos = Math.max(0, Math.min(pos, state.filteredIndices.length - 1));
  const realIndex = state.filteredIndices[state.filteredPos];
  const record = await fetchJson(`/api/record/${realIndex}`);
  state.currentRecord = record;
  populateForm(record);
  updateProgressDisplay();
}

function getSummaryByIndex(index) {
  return state.records.find((rec) => rec.index === index);
}

async function saveCurrent() {
  if (!state.currentRecord) return;
  const payload = collectPayload();
  const recordId = encodeURIComponent(state.currentRecord.record_id);
  const updated = await fetchJson(`/api/update/${recordId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const summary = getSummaryByIndex(updated._index);
  if (summary) {
    summary.completed = updated._completed;
    summary.visibility = updated.visibility;
    summary.category_name = updated.category_name;
  }

  state.currentRecord = updated;
  setStatus("Autosaved");
}

function rebuildFilteredIndices() {
  const cat = elements.categoryFilter.value || "all";
  const status = elements.statusFilter.value || "all";

  state.filteredIndices = state.records
    .filter((rec) => (cat === "all" ? true : rec.category_name === cat))
    .filter((rec) => {
      if (status === "all") return true;
      if (status === "completed") return rec.completed;
      return !rec.completed;
    })
    .map((rec) => rec.index);
}

async function applyFiltersAndReload() {
  rebuildFilteredIndices();
  await loadRecordByFilteredPos(0);
}

async function onFilterChange() {
  try {
    await saveCurrent();
  } catch (err) {
    setStatus(`Save failed: ${err.message}`, true);
  }
  await applyFiltersAndReload();
}

async function onNext() {
  try {
    await saveCurrent();
    await loadRecordByFilteredPos(state.filteredPos + 1);
  } catch (err) {
    setStatus(`Save failed: ${err.message}`, true);
  }
}

async function onPrev() {
  try {
    await saveCurrent();
    await loadRecordByFilteredPos(state.filteredPos - 1);
  } catch (err) {
    setStatus(`Save failed: ${err.message}`, true);
  }
}

async function onSaveButton() {
  try {
    await saveCurrent();
    await fetchJson("/api/save", { method: "POST" });
    setStatus("Saved");
  } catch (err) {
    setStatus(`Save failed: ${err.message}`, true);
  }
}

function shouldIgnoreShortcutTarget(target) {
  if (!target) return false;
  const tag = target.tagName?.toLowerCase() || "";
  return tag === "input" || tag === "textarea" || target.isContentEditable;
}

function bindVisibilityButtons() {
  for (const btn of elements.visibilityButtons) {
    btn.addEventListener("click", () => setVisibility(btn.dataset.value));
  }
}

function bindEvents() {
  elements.prevBtn.addEventListener("click", onPrev);
  elements.nextBtn.addEventListener("click", onNext);
  elements.saveBtn.addEventListener("click", onSaveButton);
  elements.categoryFilter.addEventListener("change", onFilterChange);
  elements.statusFilter.addEventListener("change", onFilterChange);
  bindVisibilityButtons();

  document.addEventListener("keydown", async (event) => {
    if (event.ctrlKey && event.key.toLowerCase() === "s") {
      event.preventDefault();
      await onSaveButton();
      return;
    }

    const ignoreForTextEntry = shouldIgnoreShortcutTarget(event.target);
    if (ignoreForTextEntry) return;

    if (event.key === "ArrowLeft") {
      event.preventDefault();
      await onPrev();
      return;
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      await onNext();
      return;
    }
    if (event.key === "1") {
      setVisibility("clear");
      return;
    }
    if (event.key === "2") {
      setVisibility("partial");
      return;
    }
    if (event.key === "3") {
      setVisibility("ambiguous");
    }
  });
}

function fillFilterOptions(categories) {
  elements.categoryFilter.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "all";
  allOpt.textContent = "all";
  elements.categoryFilter.appendChild(allOpt);

  for (const cat of categories) {
    const opt = document.createElement("option");
    opt.value = cat;
    opt.textContent = cat;
    elements.categoryFilter.appendChild(opt);
  }
}

async function initialize() {
  try {
    state.config = await fetchJson("/api/config");
    const recordsRes = await fetchJson("/api/records");
    state.records = recordsRes.records || [];

    elements.datasetInfo.textContent =
      `Input: ${state.config.input_path} | Sidecar: ${state.config.sidecar_path}`;

    fillFilterOptions(state.config.categories || []);
    bindEvents();
    rebuildFilteredIndices();
    await loadRecordByFilteredPos(0);
    setStatus("Ready");
  } catch (err) {
    setStatus(`Initialization error: ${err.message}`, true);
  }
}

initialize();
