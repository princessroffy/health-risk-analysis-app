const form = document.getElementById("riskForm");
const resetBtn = document.getElementById("resetBtn");
const resultState = document.getElementById("resultState");
const resultBox = document.getElementById("resultBox");
const errorBox = document.getElementById("errorBox");
const riskLevel = document.getElementById("riskLevel");
const riskScore = document.getElementById("riskScore");
const summaryText = document.getElementById("summaryText");
const notesList = document.getElementById("notesList");
const resultCard = document.querySelector(".result-card");
const predictBtn = document.getElementById("predictBtn");

function clearVisualState() {
  resultCard.classList.remove("low", "moderate", "high");
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  resultBox.classList.add("hidden");
  resultState.classList.add("hidden");
  clearVisualState();
}

function hideError() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function setRiskStyle(level) {
  clearVisualState();
  if (level === "Low") resultCard.classList.add("low");
  if (level === "Moderate") resultCard.classList.add("moderate");
  if (level === "High") resultCard.classList.add("high");
}

function getFormData() {
  return {
    age: Number(document.getElementById("age").value),
    bmi: Number(document.getElementById("bmi").value),
    systolic_bp: Number(document.getElementById("systolic_bp").value),
    diastolic_bp: Number(document.getElementById("diastolic_bp").value),
    blood_sugar: Number(document.getElementById("blood_sugar").value),
    cholesterol: Number(document.getElementById("cholesterol").value),
    activity_level: Number(document.getElementById("activity_level").value),
    smoking: Number(document.getElementById("smoking").value),
    family_history: Number(document.getElementById("family_history").value),
  };
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError();
  predictBtn.disabled = true;
  predictBtn.textContent = "Analyzing...";

  try {
    const payload = getFormData();

    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    resultState.classList.add("hidden");
    resultBox.classList.remove("hidden");

    riskLevel.textContent = data.risk_level;
    riskScore.textContent = `${data.risk_score}%`;
    summaryText.textContent = data.summary;
    notesList.innerHTML = "";

    data.notes.forEach(note => {
      const li = document.createElement("li");
      li.textContent = note;
      notesList.appendChild(li);
    });

    setRiskStyle(data.risk_level);
  } catch (error) {
    showError(error.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Analyze Risk";
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  resultBox.classList.add("hidden");
  errorBox.classList.add("hidden");
  resultState.classList.remove("hidden");
  resultState.textContent = "Submit the form to see the health risk result.";
  clearVisualState();
});