/**
 * app.js — CreditAI Frontend Logic
 * Handles form validation, API calls, gauge animation, and results rendering.
 */

"use strict";

// ─── DOM References ────────────────────────────────────────────────────────────
const form            = document.getElementById("predict-form");
const predictBtn      = document.getElementById("predict-btn");
const btnSpinner      = document.getElementById("btn-spinner");
const formError       = document.getElementById("form-error");
const formErrorMsg    = document.getElementById("form-error-msg");

const resultsPlaceholder = document.getElementById("results-placeholder");
const resultsContent     = document.getElementById("results-content");

const gaugeArc        = document.getElementById("gauge-arc");
const gaugeNeedle     = document.getElementById("gauge-needle");
const gaugeScore      = document.getElementById("gauge-score");
const riskBadge       = document.getElementById("risk-badge");

const lrProb          = document.getElementById("lr-prob");
const rfProb          = document.getElementById("rf-prob");
const ensembleProb    = document.getElementById("ensemble-prob");

const explanationText    = document.getElementById("explanation-text");
const recommendationText = document.getElementById("recommendation-text");

const creditUtilInput  = document.getElementById("credit_utilization");
const utilBar          = document.getElementById("util-bar");

// ─── Gauge constants ───────────────────────────────────────────────────────────
// Arc length of the gauge semicircle (PI * r = PI * 80 ≈ 251.2)
const GAUGE_ARC_LEN = 251.2;

// ─── Risk colour map ──────────────────────────────────────────────────────────
const RISK_COLORS = {
  Low:    "#34d399",
  Medium: "#fbbf24",
  High:   "#f87171",
};

// ─── Utility: show/hide error banner ─────────────────────────────────────────
function showError(msg) {
  formErrorMsg.textContent = msg;
  formError.hidden = false;
}
function clearError() {
  formError.hidden = true;
}

// ─── Utility: set loading state ───────────────────────────────────────────────
function setLoading(active) {
  predictBtn.disabled = active;
  predictBtn.classList.toggle("loading", active);
}

// ─── Credit utilization bar ────────────────────────────────────────────────────
function updateUtilBar() {
  const val  = Math.min(100, Math.max(0, parseFloat(creditUtilInput.value) || 0));
  utilBar.style.width = `${val}%`;

  if (val > 75) {
    utilBar.style.background = "var(--clr-high)";
  } else if (val > 50) {
    utilBar.style.background = "var(--clr-medium)";
  } else {
    utilBar.style.background = "var(--clr-success)";
  }
}
creditUtilInput.addEventListener("input", updateUtilBar);
updateUtilBar(); // initialise on load

// ─── Gauge animation ───────────────────────────────────────────────────────────
/**
 * Animate the semicircular gauge to represent riskPercent (0–100).
 * stroke-dashoffset: 251.2 = fully hidden (0%); 0 = fully shown (100%)
 */
function animateGauge(riskPercent, riskLevel) {
  const color      = RISK_COLORS[riskLevel] || "#6366f1";
  const fraction   = riskPercent / 100;
  const dashOffset = GAUGE_ARC_LEN * (1 - fraction);
  const needleDeg  = -90 + fraction * 180;   // -90° (left) → +90° (right)

  // Apply styles
  gaugeArc.style.stroke           = color;
  gaugeArc.style.strokeDashoffset = dashOffset;
  gaugeNeedle.setAttribute("transform", `rotate(${needleDeg} 100 100)`);

  // Animate counter
  animateCounter(gaugeScore, 0, riskPercent, 800, "%");
}

/**
 * Animate a number counter from `start` to `end` over `duration` ms.
 */
function animateCounter(el, start, end, duration, suffix = "") {
  const startTime = performance.now();
  const update = (now) => {
    const elapsed  = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out
    const eased   = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + (end - start) * eased);
    el.textContent = `${current}${suffix}`;
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

// ─── Render results ───────────────────────────────────────────────────────────
function renderResults(data) {
  // Show results panel
  resultsPlaceholder.hidden = true;
  resultsContent.hidden     = false;

  // Gauge
  animateGauge(data.risk_score, data.risk_level);

  // Risk badge
  riskBadge.textContent  = `${data.risk_level} Risk`;
  riskBadge.className    = `risk-badge ${data.risk_level.toLowerCase()}`;

  // Model probabilities
  lrProb.textContent       = `${data.lr_probability.toFixed(1)}%`;
  rfProb.textContent       = `${data.rf_probability.toFixed(1)}%`;
  ensembleProb.textContent = `${data.risk_score.toFixed(1)}%`;

  // Explanation & recommendation
  explanationText.textContent    = data.explanation;
  recommendationText.textContent = data.recommendation;
}

// ─── Form validation ──────────────────────────────────────────────────────────
function validateForm(fields) {
  for (const [name, label, min, max] of fields) {
    const el  = document.getElementById(name);
    const val = parseFloat(el.value);
    el.classList.remove("is-invalid");

    if (isNaN(val)) {
      el.classList.add("is-invalid");
      return `${label} must be a number.`;
    }
    if (val < min || val > max) {
      el.classList.add("is-invalid");
      return `${label} must be between ${min} and ${max}.`;
    }
  }
  return null;
}

// ─── Form submit ──────────────────────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();

  // Field validation rules: [id, label, min, max]
  const validationRules = [
    ["age",                "Age",               18,  100],
    ["income",             "Annual Income",       0,  1e9],
    ["credit_limit",       "Credit Limit",        1,  1e9],
    ["credit_utilization", "Credit Utilization",  0,  100],
    ["late_payments",      "Late Payments",       0,    6],
    ["monthly_balance",    "Monthly Payment",     0,  1e9],
  ];

  const errorMsg = validateForm(validationRules);
  if (errorMsg) {
    showError(errorMsg);
    return;
  }

  // Collect values
  const payload = {
    age:               parseInt(document.getElementById("age").value, 10),
    income:            parseFloat(document.getElementById("income").value),
    credit_limit:      parseFloat(document.getElementById("credit_limit").value),
    credit_utilization:parseFloat(document.getElementById("credit_utilization").value),
    late_payments:     parseInt(document.getElementById("late_payments").value, 10),
    monthly_balance:   parseFloat(document.getElementById("monthly_balance").value),
  };

  // Call API
  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${response.status}`);
    }

    const data = await response.json();
    renderResults(data);

    // Smooth scroll to results on mobile
    if (window.innerWidth < 860) {
      document.getElementById("results-card").scrollIntoView({ behavior: "smooth", block: "start" });
    }

  } catch (err) {
    showError(`Prediction failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

// ─── Keyboard accessibility: submit on Enter in inputs ─────────────────────
form.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.classList.contains("field-input")) {
    predictBtn.click();
  }
});
