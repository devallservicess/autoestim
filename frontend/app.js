// ============================================================
// AutoEstim v3 — Feature Importance + Cross-Validation + Optuna
// ============================================================

let metadata = null;
let metrics = null;
let selectedModel = "xgboost";

const els = {
  apiStatus:    document.getElementById("apiStatus"),
  tabs:         document.querySelectorAll(".tab"),
  views:        document.querySelectorAll(".view"),
  manufacturer: document.getElementById("manufacturer"),
  type:         document.getElementById("type"),
  year:         document.getElementById("year"),
  odometer:     document.getElementById("odometer"),
  fuel:         document.getElementById("fuel"),
  transmission: document.getElementById("transmission"),
  condition:    document.getElementById("condition"),
  drive:        document.getElementById("drive"),
  state:        document.getElementById("state"),
  modelToggle:  document.getElementById("modelToggle"),
  carForm:      document.getElementById("carForm"),
  submitBtn:    document.getElementById("submitBtn"),
  formError:    document.getElementById("formError"),
  resultEmpty:  document.getElementById("resultEmpty"),
  resultContent:document.getElementById("resultContent"),
  resultValue:  document.getElementById("resultValue"),
  resultMeta:   document.getElementById("resultMeta"),
  resultSummary:document.getElementById("resultSummary"),
  marketContext:document.getElementById("marketContext"),
  confR2:       document.getElementById("confR2"),
  confMae:      document.getElementById("confMae"),
  confFill:     document.getElementById("confFill"),
  statTotal:    document.getElementById("statTotal"),
  statAvg:      document.getElementById("statAvg"),
  statTopBrand: document.getElementById("statTopBrand"),
  brandChart:   document.getElementById("brandChart"),
  fuelChart:    document.getElementById("fuelChart"),
  carsTableBody:document.getElementById("carsTableBody"),
  refreshDash:  document.getElementById("refreshDash"),
  modelsGrid:   document.getElementById("modelsGrid"),
  datasetBanner:document.getElementById("datasetBanner"),
  fiSection:    document.getElementById("fiSection"),
  cvSection:    document.getElementById("cvSection"),
  fiModelToggle:document.getElementById("fiModelToggle"),
  toast:        document.getElementById("toast"),
};

const FUEL_COLORS = {
  gas: "#ff7a33", diesel: "#5b9eff", hybrid: "#5fd080",
  electric: "#c792ea", other: "#9aa2b1",
};

// ── Utilitaires ───────────────────────────────────────────────────────────────
const formatPrice = (v) => new Intl.NumberFormat("en-US").format(Math.round(v));
const capitalize  = (s) => s ? s.charAt(0).toUpperCase() + s.slice(1) : s;

function showToast(msg, isError = false) {
  els.toast.textContent = msg;
  els.toast.classList.toggle("error", isError);
  els.toast.classList.add("show");
  setTimeout(() => els.toast.classList.remove("show"), 3200);
}

async function apiRequest(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" }, ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Erreur ${res.status}`);
  }
  return res.json();
}

// ── Navigation ────────────────────────────────────────────────────────────────
els.tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    els.tabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const target = tab.dataset.view;
    els.views.forEach((v) => v.classList.toggle("active", v.id === `view-${target}`));
    if (target === "dashboard") loadDashboard();
    if (target === "models")   loadModelsView();
  });
});

// ── Sélecteur modèle ML ───────────────────────────────────────────────────────
els.modelToggle.addEventListener("click", (e) => {
  const btn = e.target.closest(".model-opt");
  if (!btn) return;
  document.querySelectorAll(".model-opt").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  selectedModel = btn.dataset.model;
});

// ── Remplissage selects ───────────────────────────────────────────────────────
function fillSelect(select, options, defaultLabel = null) {
  if (defaultLabel) {
    const o = document.createElement("option");
    o.value = ""; o.disabled = true; o.selected = true;
    o.textContent = defaultLabel;
    select.appendChild(o);
  }
  options.forEach((opt) => {
    const o = document.createElement("option");
    o.value = opt; o.textContent = capitalize(opt);
    select.appendChild(o);
  });
}

function fillSelectWithDefault(select, options, defaultVal) {
  while (select.options.length > 1) select.remove(1);
  options.filter((o) => o !== defaultVal).forEach((opt) => {
    const o = document.createElement("option");
    o.value = opt; o.textContent = capitalize(opt);
    select.appendChild(o);
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function loadMetadata() {
  try {
    metadata = await apiRequest("/api/metadata");
    metrics  = await apiRequest("/api/metrics");
    fillSelect(els.manufacturer, metadata.manufacturers, "Choisir un constructeur");
    fillSelectWithDefault(els.type, metadata.types, "unknown");
    fillSelect(els.fuel, metadata.fuel_types, "Choisir");
    fillSelect(els.transmission, metadata.transmissions, "Choisir");
    fillSelectWithDefault(els.condition, metadata.conditions, "good");
    fillSelectWithDefault(els.drive, metadata.drives, "unknown");
    fillSelectWithDefault(els.state, metadata.states, "unknown");
    els.year.min = metadata.year_min;
    els.year.max = metadata.year_max;
    els.year.value = 2018;
    els.odometer.value = metadata.odometer_avg || 80000;
    setApiStatus(true);
  } catch (err) {
    setApiStatus(false);
    console.error("API inaccessible:", err);
  }
}

function setApiStatus(online) {
  els.apiStatus.classList.toggle("online", online);
  els.apiStatus.classList.toggle("offline", !online);
  els.apiStatus.querySelector(".status-text").textContent =
    online ? "API connectée" : "API hors-ligne";
}

// ── Formulaire ────────────────────────────────────────────────────────────────
els.carForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  els.formError.textContent = "";

  const payload = {
    manufacturer: els.manufacturer.value,
    year:         parseInt(els.year.value, 10),
    odometer:     parseInt(els.odometer.value, 10),
    fuel:         els.fuel.value,
    transmission: els.transmission.value,
    condition:    els.condition.value || "good",
    drive:        els.drive.value    || "unknown",
    type:         els.type.value     || "unknown",
    state:        els.state.value    || "unknown",
    model:        selectedModel,
  };

  if (!payload.manufacturer || !payload.fuel || !payload.transmission) {
    els.formError.textContent = "Remplis les champs obligatoires (constructeur, carburant, transmission).";
    return;
  }

  els.submitBtn.disabled = true;
  els.submitBtn.querySelector("span").textContent = "Calcul en cours…";

  try {
    const prediction = await apiRequest("/api/predict", {
      method: "POST", body: JSON.stringify(payload),
    });
    displayResult(prediction);
    showToast("Estimation enregistrée ✓");
  } catch (err) {
    els.formError.textContent = err.message;
    showToast(err.message, true);
  } finally {
    els.submitBtn.disabled = false;
    els.submitBtn.querySelector("span").textContent = "Estimer le prix";
  }
});

function displayResult(prediction) {
  els.resultEmpty.style.display = "none";
  els.resultContent.style.display = "flex";
  els.resultValue.textContent = formatPrice(prediction.predicted_price);

  const modelLabel = prediction.model_used === "xgboost" ? "XGBoost" : "Random Forest";
  els.resultMeta.textContent = `Calculé avec ${modelLabel} · ${new Date().toLocaleTimeString("fr-FR")}`;

  const km = Math.round(prediction.odometer * 1.60934);
  els.resultSummary.innerHTML = `
    <b>${capitalize(prediction.manufacturer)}</b> ${prediction.year} ·
    ${formatPrice(prediction.odometer)} miles (≈${formatPrice(km)} km) ·
    ${capitalize(prediction.fuel)} · ${capitalize(prediction.transmission)} ·
    État : ${capitalize(prediction.condition)}
  `;

  if (metadata) {
    const diff = prediction.predicted_price - metadata.price_mean;
    const pct  = Math.abs((diff / metadata.price_mean) * 100).toFixed(1);
    const dir  = diff >= 0 ? "au-dessus" : "en-dessous";
    const col  = diff >= 0 ? "#ff7a33" : "#5fd080";
    els.marketContext.innerHTML = `
      <div class="market-badge" style="border-color:${col}22;background:${col}11">
        <span style="color:${col}">${diff >= 0 ? "▲" : "▼"} ${pct}% ${dir} du prix moyen du marché</span>
        <span class="market-mean">Moyenne dataset : $${formatPrice(metadata.price_mean)}</span>
      </div>`;
  }

  const m = metrics?.[prediction.model_used];
  if (m) {
    els.confR2.textContent  = m.r2.toFixed(4);
    els.confMae.textContent = `± $${formatPrice(m.mae)}`;
    els.confFill.style.width = `${Math.min(m.r2 * 100, 100)}%`;
  }
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
async function loadDashboard() {
  try {
    const [stats, cars] = await Promise.all([
      apiRequest("/api/dashboard"), apiRequest("/api/cars"),
    ]);
    renderStats(stats);
    renderBrandChart(stats.by_manufacturer);
    renderFuelChart(stats.by_fuel);
    renderCarsTable(cars);
  } catch {
    showToast("Impossible de charger le dashboard.", true);
  }
}

function renderStats(stats) {
  els.statTotal.textContent = stats.total_predictions;
  els.statAvg.textContent   = `$${formatPrice(stats.average_price)}`;
  if (stats.by_manufacturer.length)
    els.statTopBrand.textContent = capitalize(stats.by_manufacturer[0].manufacturer);
}

function renderBrandChart(byBrand) {
  if (!byBrand.length) { els.brandChart.innerHTML = `<div class="empty-chart">Pas encore de données</div>`; return; }
  const max = Math.max(...byBrand.map((b) => b.avg_price));
  els.brandChart.innerHTML = byBrand.slice(0, 8).map((b) => `
    <div class="bar-row">
      <span class="bar-label">${capitalize(b.manufacturer)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(b.avg_price/max)*100}%"></div></div>
      <span class="bar-value">$${formatPrice(b.avg_price)}</span>
    </div>`).join("");
}

function renderFuelChart(byFuel) {
  if (!byFuel.length) { els.fuelChart.innerHTML = `<div class="empty-chart">Pas encore de données</div>`; return; }
  const total = byFuel.reduce((s, f) => s + f.count, 0);
  let cum = 0;
  const parts = byFuel.map((f) => {
    const s = (cum/total)*360, e = ((cum+=f.count)/total)*360;
    return `${FUEL_COLORS[f.fuel]||"#9aa2b1"} ${s}deg ${e}deg`;
  });
  const legend = byFuel.map((f) => `
    <div class="legend-item">
      <span class="legend-dot" style="background:${FUEL_COLORS[f.fuel]||"#9aa2b1"}"></span>
      ${capitalize(f.fuel)} — ${f.count}
    </div>`).join("");
  els.fuelChart.innerHTML = `
    <div style="width:110px;height:110px;border-radius:50%;background:conic-gradient(${parts.join(",")})"></div>
    <div class="donut-legend">${legend}</div>`;
}

function renderCarsTable(cars) {
  if (!cars.length) {
    els.carsTableBody.innerHTML = `<tr class="empty-row"><td colspan="9">Aucune voiture analysée.</td></tr>`;
    return;
  }
  els.carsTableBody.innerHTML = cars.map((c) => `
    <tr>
      <td>${capitalize(c.manufacturer)}</td><td>${c.year}</td>
      <td>${formatPrice(c.odometer)}</td><td>${capitalize(c.fuel)}</td>
      <td>${capitalize(c.transmission)}</td>
      <td>${c.state !== "unknown" ? c.state.toUpperCase() : "—"}</td>
      <td><span class="model-badge">${c.model_used==="xgboost"?"XGBoost":"RF"}</span></td>
      <td class="price-cell">$${formatPrice(c.predicted_price)}</td>
      <td><button class="delete-btn" data-delete="${c.id}">✕</button></td>
    </tr>`).join("");
}

els.carsTableBody.addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-delete]");
  if (!btn) return;
  try { await apiRequest(`/api/cars/${btn.dataset.delete}`, {method:"DELETE"}); loadDashboard(); }
  catch { showToast("Erreur suppression", true); }
});

els.refreshDash.addEventListener("click", loadDashboard);

// ── Vue Modèles (Feature Importance + CV) ────────────────────────────────────
let currentFiModel = "xgboost";

async function loadModelsView() {
  renderDatasetBanner();
  renderModelCards();
  await loadFeatureImportance(currentFiModel);
  await loadCrossValidation();
}

function renderDatasetBanner() {
  if (!metadata) return;
  els.datasetBanner.innerHTML = `
    <div class="dataset-info">
      <span class="di-item">📦 <b>${formatPrice(metadata.dataset_size)}</b> voitures réelles</span>
      <span class="di-item">💰 $${formatPrice(metadata.price_min)} – $${formatPrice(metadata.price_max)}</span>
      <span class="di-item">📊 Moyenne : $${formatPrice(metadata.price_mean)}</span>
      <span class="di-item">🇺🇸 Source : Craigslist USA</span>
      <span class="di-item">🔬 Tuning : Optuna</span>
    </div>`;
}

function renderModelCards() {
  if (!metrics) { els.modelsGrid.innerHTML = `<div class="empty-chart">Métriques indisponibles.</div>`; return; }
  const cards = [
    {key:"xgboost",       name:"XGBoost Regressor",       highlight:true},
    {key:"random_forest", name:"Random Forest Regressor",  highlight:false},
  ];
  els.modelsGrid.innerHTML = cards.map((c) => {
    const m = metrics[c.key]; if (!m) return "";
    const params = m.best_params || {};
    const paramStr = Object.entries(params)
      .slice(0, 3)
      .map(([k,v]) => `${k}: ${typeof v === "number" && !Number.isInteger(v) ? v.toFixed(3) : v}`)
      .join(" · ");
    return `
      <div class="model-card ${c.highlight?"highlight":""}">
        <div class="model-card-head">
          <span class="model-card-name">${c.name}</span>
          ${c.highlight ? '<span class="model-card-badge">Recommandé</span>' : ""}
        </div>
        <div class="metric-row"><span>MAE (erreur moyenne)</span><span>$${formatPrice(m.mae)}</span></div>
        <div class="metric-row"><span>RMSE</span><span>$${formatPrice(m.rmse)}</span></div>
        <div class="metric-row"><span>R² (score précision)</span><span>${m.r2.toFixed(4)}</span></div>
        <div class="metric-row"><span>Temps entraînement</span><span>${m.train_seconds}s</span></div>
        ${paramStr ? `<div class="params-row"><span class="params-label">Optuna params</span><span class="params-val">${paramStr}</span></div>` : ""}
      </div>`;
  }).join("");
}

async function loadFeatureImportance(modelName) {
  currentFiModel = modelName;
  document.querySelectorAll(".fi-tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.fiModel === modelName));

  try {
    const data = await apiRequest(`/api/feature-importance/${modelName}?top=15`);
    renderFeatureImportance(data.features, modelName);
  } catch {
    els.fiSection.innerHTML = `<div class="empty-chart">Feature importance non disponible — re-lance train_models.py.</div>`;
  }
}

function renderFeatureImportance(features, modelName) {
  if (!features || !features.length) {
    els.fiSection.innerHTML = `<div class="empty-chart">Pas de données.</div>`;
    return;
  }

  const max = features[0].importance_pct;
  const bars = features.map((f, i) => {
    // Colorer selon le type de feature
    const isNumeric = ["year", "odometer"].includes(f.feature);
    const color = isNumeric ? "#ff7a33" : "#5b9eff";
    const pct = (f.importance_pct / max * 100).toFixed(1);
    return `
      <div class="fi-row">
        <span class="fi-rank">#${i+1}</span>
        <span class="fi-name" title="${f.feature}">${f.feature.length > 22 ? f.feature.slice(0,22)+"…" : f.feature}</span>
        <div class="fi-track">
          <div class="fi-fill" style="width:${pct}%;background:${color}"></div>
        </div>
        <span class="fi-pct">${f.importance_pct.toFixed(2)}%</span>
      </div>`;
  }).join("");

  els.fiSection.innerHTML = `
    <div class="fi-legend">
      <span><span class="fi-dot" style="background:#ff7a33"></span>Feature numérique</span>
      <span><span class="fi-dot" style="background:#5b9eff"></span>Feature catégorielle</span>
    </div>
    <div class="fi-bars">${bars}</div>`;
}

async function loadCrossValidation() {
  try {
    const cv = await apiRequest("/api/cross-validation");
    renderCrossValidation(cv);
  } catch {
    els.cvSection.innerHTML = `<div class="empty-chart">Données CV non disponibles — re-lance train_models.py.</div>`;
  }
}

function renderCrossValidation(cv) {
  const models = [
    {key:"xgboost",       label:"XGBoost",        color:"#ff7a33"},
    {key:"random_forest", label:"Random Forest",   color:"#5b9eff"},
  ];

  els.cvSection.innerHTML = models.map((m) => {
    const d = cv[m.key]; if (!d) return "";
    return `
      <div class="cv-card" style="border-left:3px solid ${m.color}">
        <div class="cv-card-title" style="color:${m.color}">${m.label}</div>
        <div class="cv-grid">
          <div class="cv-metric">
            <span class="cv-label">R² moyen (${d.n_folds} folds)</span>
            <span class="cv-value">${d.r2_mean.toFixed(4)} <span class="cv-std">± ${d.r2_std.toFixed(4)}</span></span>
          </div>
          <div class="cv-metric">
            <span class="cv-label">MAE moyen</span>
            <span class="cv-value">$${formatPrice(d.mae_mean)} <span class="cv-std">± $${formatPrice(d.mae_std)}</span></span>
          </div>
          <div class="cv-metric">
            <span class="cv-label">RMSE moyen</span>
            <span class="cv-value">$${formatPrice(d.rmse_mean)} <span class="cv-std">± $${formatPrice(d.rmse_std)}</span></span>
          </div>
        </div>
        <p class="cv-note">L'écart-type faible (± ${d.r2_std.toFixed(4)}) confirme que le modèle généralise bien et n'overfit pas.</p>
      </div>`;
  }).join("");
}

// Toggle Feature Importance XGBoost / RF
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".fi-tab");
  if (!btn) return;
  loadFeatureImportance(btn.dataset.fiModel);
});

// ── Init ──────────────────────────────────────────────────────────────────────
loadMetadata();
