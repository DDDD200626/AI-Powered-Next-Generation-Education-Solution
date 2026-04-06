(function () {
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

  function showTab(id) {
    $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === id));
    $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + id));
  }

  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => showTab(btn.dataset.tab));
  });

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || r.statusText || "요청 실패");
    return data;
  }

  function renderTable(container, columns, rows) {
    if (!rows.length) {
      container.innerHTML = "<p class='note'>데이터가 없습니다.</p>";
      return;
    }
    const th = columns.map((c) => `<th>${escapeHtml(String(c))}</th>`).join("");
    const tr = rows
      .map(
        (row) =>
          "<tr>" +
          columns.map((c) => `<td>${escapeHtml(String(row[c] ?? ""))}</td>`).join("") +
          "</tr>"
      )
      .join("");
    container.innerHTML = `<table class="data"><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table>`;
  }

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderHits(container, hits) {
    container.innerHTML = "";
    if (!hits || !hits.length) {
      container.innerHTML = "<p class='note'>근거가 없습니다.</p>";
      return;
    }
    hits.forEach((h) => {
      const det = document.createElement("details");
      det.className = "hit";
      det.open = false;
      det.innerHTML =
        "<summary>[" +
        h.chunk_id +
        "] " +
        escapeHtml(h.source) +
        " · 유사도 " +
        (h.score != null ? Number(h.score).toFixed(3) : "") +
        "</summary><pre>" +
        escapeHtml(h.text) +
        "</pre>";
      container.appendChild(det);
    });
  }

  function setText(el, text) {
    el.textContent = text || "";
  }

  function setProse(el, text) {
    el.textContent = text || "";
  }

  $("#btn-learn").addEventListener("click", async () => {
    const q = $("#learn-q").value.trim();
    const k = Math.min(8, Math.max(2, parseInt($("#learn-k").value, 10) || 4));
    $("#learn-hits").innerHTML = "";
    setProse($("#learn-ans"), "");
    if (!q) {
      alert("질문을 입력하세요.");
      return;
    }
    try {
      const data = await postJSON("/api/learn", { query: q, top_k: k });
      renderHits($("#learn-hits"), data.hits);
      if (data.answer_error) setProse($("#learn-ans"), data.answer_error);
      else setProse($("#learn-ans"), data.answer || "");
    } catch (e) {
      alert(e.message);
    }
  });

  $("#btn-comp").addEventListener("click", async () => {
    const question = $("#comp-q").value.trim();
    const answer = $("#comp-a").value.trim();
    const top_k = Math.min(8, Math.max(2, parseInt($("#comp-k").value, 10) || 4));
    $("#comp-hits").innerHTML = "";
    setProse($("#comp-report"), "");
    if (!question || !answer) {
      alert("질문과 답변을 입력하세요.");
      return;
    }
    try {
      const data = await postJSON("/api/comprehension", { question, answer, top_k });
      renderHits($("#comp-hits"), data.hits);
      if (data.report_error) setProse($("#comp-report"), data.report_error);
      else setProse($("#comp-report"), data.report || "");
    } catch (e) {
      alert(e.message);
    }
  });

  $("#btn-integ").addEventListener("click", async () => {
    const submission = $("#integ-sub").value.trim();
    const assignment_prompt = $("#integ-prompt").value;
    setText($("#integ-heur"), "");
    setText($("#integ-sim"), "");
    setProse($("#integ-narr"), "");
    if (!submission) {
      alert("제출 본문을 입력하세요.");
      return;
    }
    try {
      const data = await postJSON("/api/integrity", { assignment_prompt, submission });
      setText($("#integ-heur"), JSON.stringify(data.heuristic, null, 2));
      const simLine =
        "자료 대비 최대 유사도(참고): " +
        data.max_similarity +
        (data.max_similarity_source ? " · 출처: " + data.max_similarity_source : "");
      setText($("#integ-sim"), simLine);
      if (data.narrative_error) setProse($("#integ-narr"), data.narrative_error);
      else setProse($("#integ-narr"), data.narrative || "");
    } catch (e) {
      alert(e.message);
    }
  });

  $("#btn-fb").addEventListener("click", async () => {
    const rubric = $("#rub").value;
    const submission = $("#sub").value.trim();
    setProse($("#fb-out"), "");
    if (!submission) {
      alert("학생 제출을 입력하세요.");
      return;
    }
    try {
      const data = await postJSON("/api/teacher-feedback", { rubric, submission });
      setProse($("#fb-out"), data.draft || "");
    } catch (e) {
      alert(e.message);
    }
  });

  async function dashPreview() {
    const csv_text = $("#dash-csv").value;
    const data = await postJSON("/api/dashboard/preview", { csv_text });
    setText($("#dash-note"), data.note || "");
    renderTable($("#dash-table-wrap"), data.columns, data.rows);
    setText($("#dash-stats"), JSON.stringify(data.stats, null, 2));
  }

  $("#btn-dash-preview").addEventListener("click", async () => {
    try {
      await dashPreview();
    } catch (e) {
      alert(e.message);
    }
  });

  $("#btn-dash-sum").addEventListener("click", async () => {
    setProse($("#dash-summary"), "");
    try {
      await dashPreview();
      const data = await postJSON("/api/dashboard/summarize", { csv_text: $("#dash-csv").value });
      setProse($("#dash-summary"), data.summary || "");
    } catch (e) {
      alert(e.message);
    }
  });

  async function opsPreview() {
    const csv_text = $("#ops-csv").value;
    const data = await postJSON("/api/ops/preview", { csv_text });
    setText($("#ops-note"), data.note || "");
    renderTable($("#ops-table-wrap"), data.columns, data.rows);
    setText($("#ops-stats"), JSON.stringify(data.stats, null, 2));
  }

  $("#btn-ops-preview").addEventListener("click", async () => {
    try {
      await opsPreview();
    } catch (e) {
      alert(e.message);
    }
  });

  $("#btn-ops-sum").addEventListener("click", async () => {
    setProse($("#ops-summary"), "");
    try {
      await opsPreview();
      const data = await postJSON("/api/ops/summarize", { csv_text: $("#ops-csv").value });
      setProse($("#ops-summary"), data.summary || "");
    } catch (e) {
      alert(e.message);
    }
  });

  showTab("dash");
})();
