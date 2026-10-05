"use strict";

const $ = (selector) => document.querySelector(selector);
const form = $("#passenger-form");
const predictButton = $("#predict-button");
const resultPanel = $("#result-panel");
const scene = $("#scene");
const numberFormat = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
let modelReady = false;
let revision = 0;
let controller = null;
let hasPrediction = false;

function formValue(name) {
  return new FormData(form).get(name);
}

function passengerData() {
  const numberOrNull = (name) => {
    const value = formValue(name);
    return value === "" ? null : Number(value);
  };
  return {
    sex: formValue("sex"), pclass: Number(formValue("pclass")), age: numberOrNull("age"),
    sibsp: Number(formValue("sibsp")), parch: Number(formValue("parch")),
    fare: numberOrNull("fare"), deck: formValue("deck"), embarked: formValue("embarked"),
  };
}

function updateTicket() {
  const age = $("#age").value;
  $("#age-range").disabled = age === "";
  $("#age-range").setAttribute("aria-valuetext", age === "" ? "Возраст неизвестен" : `${age} лет`);
  $("#ticket-name").textContent = $("#passenger-name").value.trim().toLocaleUpperCase("ru") || "ВАША ИСТОРИЯ";
  $("#ticket-class").textContent = { 1: "I", 2: "II", 3: "III" }[formValue("pclass")];
  const deck = formValue("deck");
  $("#ticket-deck").textContent = `ПАЛУБА ${deck === "UNKNOWN" ? "—" : deck}`;
  $("#ticket-port").textContent = { S: "САУТГЕМПТОН", C: "ШЕРБУР", Q: "КВИНСТАУН", UNKNOWN: "ПОРТ НЕИЗВЕСТЕН" }[formValue("embarked")];
  for (const button of document.querySelectorAll("[data-step]")) {
    const input = document.getElementById(button.dataset.target);
    button.disabled = Number(button.dataset.step) < 0 ? Number(input.value) <= 0 : Number(input.value) >= 20;
  }
}

function setBusy(busy) {
  scene.classList.toggle("is-scanning", busy);
  predictButton.disabled = busy || !modelReady;
  predictButton.setAttribute("aria-busy", String(busy));
  $("#predict-label").textContent = busy ? "Изучаем ваш билет…" : "Узнать свою историю";
}

function invalidatePrediction() {
  revision += 1;
  controller?.abort();
  controller = null;
  setBusy(false);
  $("#form-error").hidden = true;
  if (hasPrediction) {
    resultPanel.classList.remove("has-result", "low-chance");
    $("#probability-track").hidden = true;
    $("#result-missing").hidden = true;
    $("#result-kicker").textContent = "ВАШ БИЛЕТ ИЗМЕНИЛСЯ";
    $("#result-title").textContent = "Новый билет — новая история";
    $("#result-copy").textContent = "Нажмите «Узнать свою историю», чтобы пересчитать прогноз для новых данных.";
    hasPrediction = false;
  }
}

form.addEventListener("input", (event) => {
  const target = event.target;
  if (target.id === "age-range") $("#age").value = target.value;
  if (target.id === "age" && target.value !== "" && target.validity.valid) $("#age-range").value = target.value;
  if (target.id !== "passenger-name") invalidatePrediction();
  updateTicket();
});
form.addEventListener("change", (event) => {
  if (event.target.id !== "passenger-name") invalidatePrediction();
  updateTicket();
});

form.addEventListener("invalid", (event) => {
  const details = event.target.closest("details");
  if (details) details.open = true;
}, true);

for (const button of document.querySelectorAll("[data-step]")) {
  button.addEventListener("click", () => {
    const input = document.getElementById(button.dataset.target);
    input.value = String(Math.max(0, Math.min(20, (Number(input.value) || 0) + Number(button.dataset.step))));
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

const presets = [
  { name: "Элеонора", sex: "female", pclass: 1, age: 32, sibsp: 1, parch: 0, fare: 80, deck: "B", embarked: "C" },
  { name: "Томас", sex: "male", pclass: 3, age: 24, sibsp: 0, parch: 0, fare: 8, deck: "UNKNOWN", embarked: "S" },
  { name: "Эмили", sex: "female", pclass: 2, age: 19, sibsp: 0, parch: 0, fare: 13, deck: "UNKNOWN", embarked: "S" },
  { name: "Артур", sex: "male", pclass: 2, age: 6, sibsp: 1, parch: 1, fare: 26, deck: "F", embarked: "S" },
  { name: "Генри", sex: "male", pclass: 1, age: 48, sibsp: 1, parch: 0, fare: 76, deck: "C", embarked: "C" },
];
let previousPreset = -1;
$("#randomize").addEventListener("click", () => {
  const choices = presets.map((_, index) => index).filter((index) => index !== previousPreset);
  previousPreset = choices[Math.floor(Math.random() * choices.length)];
  const preset = presets[previousPreset];
  for (const key of ["sex", "pclass"]) form.querySelector(`input[name="${key}"][value="${preset[key]}"]`).checked = true;
  for (const key of ["age", "sibsp", "parch", "fare", "deck", "embarked"]) document.getElementById(key).value = preset[key];
  $("#passenger-name").value = preset.name;
  $("#age-range").value = preset.age;
  invalidatePrediction();
  updateTicket();
});

function showResult(result) {
  const probability = result.survival_probability;
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) throw new Error("Модель вернула некорректный результат.");
  const likelySurvived = result.predicted_survived === 1;
  resultPanel.classList.add("has-result");
  resultPanel.classList.toggle("low-chance", !likelySurvived);
  $("#probability-value").textContent = `${numberFormat.format(probability * 100)}%`;
  $("#probability-caption").textContent = "ВЫЖИВАНИЕ";
  $("#result-kicker").textContent = "ВАША ИСТОРИЯ · ПРОГНОЗ НЕЙРОСЕТИ";
  $("#result-title").textContent = likelySurvived ? "Шанс на новую главу" : "Путешествие с высоким риском";
  $("#result-copy").textContent = likelySurvived
    ? "Для этого билета модель считает выживание более вероятным исходом."
    : "Для этого билета модель считает гибель более вероятным исходом.";
  $("#probability-track").hidden = false;
  $("#probability-progress").value = probability * 100;
  const fieldNames = { age: "возраст", fare: "цена билета", deck: "палуба", embarked: "порт посадки" };
  const missing = result.missing_inputs || [];
  $("#result-missing").hidden = missing.length === 0;
  $("#result-missing").textContent = `Учтены неизвестные данные: ${missing.map((name) => fieldNames[name]).join(", ")}.`;
  hasPrediction = true;
  if (window.matchMedia("(max-width: 650px)").matches) resultPanel.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "center" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!modelReady || !form.reportValidity()) return;
  controller?.abort();
  controller = new AbortController();
  const currentController = controller;
  const requestRevision = ++revision;
  $("#form-error").hidden = true;
  setBusy(true);
  const timeout = setTimeout(() => currentController.abort(), 12000);
  try {
    const [response] = await Promise.all([
      fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(passengerData()), signal: currentController.signal }),
      new Promise((resolve) => setTimeout(resolve, window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 850)),
    ]);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Не удалось получить прогноз.");
    if (revision !== requestRevision) return;
    showResult(result);
  } catch (error) {
    if (revision !== requestRevision) return;
    $("#form-error").textContent = error.name === "AbortError"
      ? "Сервер не ответил вовремя. Проверьте, что приложение запущено, и повторите попытку."
      : error instanceof TypeError ? "Нет связи с моделью. Проверьте, что локальное приложение запущено."
      : error.message;
    $("#form-error").hidden = false;
  } finally {
    clearTimeout(timeout);
    if (revision === requestRevision) { controller = null; setBusy(false); }
  }
});

const dialog = $("#about-dialog");
for (const id of ["about-open", "method-open"]) document.getElementById(id).addEventListener("click", () => dialog.showModal());
$("#about-close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  const rect = dialog.getBoundingClientRect();
  if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
});

async function connectModel() {
  try {
    const response = await fetch("/api/model", { signal: AbortSignal.timeout(8000) });
    if (!response.ok) throw new Error("Model unavailable");
    const model = await response.json();
    if (model.ready !== true) throw new Error("Model not ready");
    modelReady = true;
    $("#connection").classList.add("ready");
    $("#connection-text").textContent = "НЕЙРОСЕТЬ ГОТОВА";
    for (const id of ["accuracy-stat", "about-accuracy"]) document.getElementById(id).textContent = `${numberFormat.format(model.accuracy * 100)}%`;
    for (const id of ["dataset-stat", "about-dataset"]) document.getElementById(id).textContent = new Intl.NumberFormat("ru-RU").format(model.dataset_count);
    $("#about-test").textContent = model.test_count;
    predictButton.disabled = false;
  } catch {
    $("#connection").classList.add("offline");
    $("#connection-text").textContent = "НЕТ СВЯЗИ";
    $("#form-error").textContent = "Не удалось подключиться к модели. Запустите приложение и обновите страницу.";
    $("#form-error").hidden = false;
  }
}

updateTicket();
connectModel();
