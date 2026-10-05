const API = location.protocol === "file:" ? "http://localhost:8000" : "";
const $ = (id) => document.getElementById(id);
const st = { ready: false, busy: false };

const SAMPLES = {
  fake: {
    title: "BREAKING: Secret Documents PROVE Elites Are Hiding Miracle Cure From You, Media Silent!",
    text: "Insiders have finally leaked shocking documents that the mainstream media does NOT want you to see. According to anonymous sources close to the establishment, powerful elites have been secretly suppressing a miracle cure for years so they can keep making billions from sick people. The so-called experts are lying to you, and the fake news networks are working overtime to cover it up. Patriots who dared to speak out have been silenced, censored and banned from social media. Wake up, people! Share this article with everyone you know before it gets deleted, because they are already trying to shut this site down. Do your own research and never trust the corrupt media again.",
  },
  real: {
    title: "City Council Approves 2027 Budget After Two-Hour Debate",
    text: "The Riverton City Council voted 6-3 on Tuesday evening to approve a $214 million budget for the 2027 fiscal year, following a two-hour public hearing. The plan includes a 2.1 percent increase in spending on road maintenance and a new $4 million allocation for the public library system. Council member Dana Whitfield said the budget reflected feedback gathered at town meetings over the summer. Opponents argued the plan relies too heavily on projected sales tax revenue. The city finance office said it expects to publish the full budget document on its website next week, and the new fiscal year begins on January 1.",
  },
};

function updateButtons() {
  document.querySelectorAll("button[type=submit], .chip").forEach((b) => (b.disabled = !st.ready || st.busy));
}
function setError(msg) { $("error").textContent = msg || ""; $("error").hidden = !msg; }
function setBanner(msg, isError) {
  $("banner").textContent = msg || "";
  $("banner").className = "alert" + (isError ? "" : " alert-info");
  $("banner").hidden = !msg;
}
function countWords() {
  const n = ($("title").value + " " + $("body").value).trim().split(/\s+/).filter(Boolean).length;
  $("counter").textContent = `${n} word${n === 1 ? "" : "s"}`;
}
$("body").addEventListener("input", countWords);
$("title").addEventListener("input", countWords);

function setMode(mode) {
  ["manual", "url"].forEach((m) => {
    const on = m === mode;
    $("tab-" + m).setAttribute("aria-selected", on);
    $("tab-" + m).tabIndex = on ? 0 : -1;
    $("panel-" + m).hidden = !on;
  });
  setError(""); $("result").hidden = true;
}
document.querySelectorAll("[role=tab]").forEach((t) => {
  t.addEventListener("click", () => setMode(t.id.replace("tab-", "")));
  t.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      const next = t.id === "tab-manual" ? "url" : "manual";
      setMode(next); $("tab-" + next).focus();
    }
  });
});

function detailText(d) {
  if (Array.isArray(d)) return d.map((x) => x.msg).join(" ");
  return typeof d === "string" ? d : "";
}

async function submit(endpoint, payload, button) {
  const label = button.textContent;
  setError(""); st.busy = true; updateButtons();
  button.innerHTML = '<span class="spinner"></span>Analyzing...';
  try {
    const res = await fetch(API + endpoint, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(detailText(data.detail) || `Request failed (${res.status}).`);
    render(data);
  } catch (err) {
    $("result").hidden = true;
    setError(err instanceof TypeError ? "Can't reach the server. Is it running?" : err.message);
  } finally {
    st.busy = false; button.textContent = label; updateButtons();
  }
}

function animateWidth(el, pct) {
  el.style.transition = "none"; el.style.width = "0%";
  void el.offsetWidth;
  el.style.transition = "";
  requestAnimationFrame(() => requestAnimationFrame(() => (el.style.width = pct + "%")));
}

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text) n.textContent = text;
  return n;
}
const signed = (n) => (n > 0 ? "+" + n : String(n));

function evidenceCard(e) {
  const card = el("div", "ev");
  const head = el("div", "ev-head");
  head.append(el("span", "ev-dot ev-" + e.status), el("strong", "", e.name));
  card.append(head, el("p", "ev-sum", e.summary));
  if (e.items.length) {
    const ul = el("ul", "ev-items");
    e.items.forEach((i) => {
      const li = document.createElement("li");
      const a = el("a", "", i.title || i.url);
      a.href = /^https?:\/\//i.test(i.url) ? i.url : "#";
      a.target = "_blank"; a.rel = "noopener noreferrer";
      li.append(a, el("span", "ev-src", " " + [i.source, i.rating].filter(Boolean).join(" - ")));
      ul.append(li);
    });
    card.append(ul);
  }
  return card;
}

function renderVerification(v) {
  const o = v.overall;
  const kind = { reliable: "real", unreliable: "fake", mixed: "mixed" }[o.level];
  $("overall").className = "verdict verdict-" + kind;
  $("overall-icon").textContent = { reliable: "✓", unreliable: "!", mixed: "?" }[o.level];
  $("overall-label").textContent = o.headline;
  $("overall-sub").textContent = `Score ${signed(o.score)}. Reliable at +2 or more, unreliable at -2 or less.`;
  $("votes").replaceChildren(...o.votes.map((x) =>
    el("span", "vote " + (x.vote > 0 ? "pos" : x.vote < 0 ? "neg" : ""), `${signed(x.vote)} ${x.name}`)));
  $("evidence-list").replaceChildren(...v.evidence.map(evidenceCard));
}

function render(d) {
  const fake = d.label === "Fake";
  const v = d.verification;
  // Evidence votes other than the style model. When they are all zero (no API
  // keys, unknown source, no coverage), the style model is the only signal, so
  // its verdict headlines; otherwise the combined evidence verdict does.
  const evidenceActive = v && v.overall && v.overall.votes
    ? v.overall.votes.some((x) => x.name !== "Writing style" && x.vote !== 0)
    : false;
  const level = evidenceActive ? v.overall.level : null;
  // The writing-style model is a weak signal on out-of-domain articles, so when
  // live evidence exists the evidence verdict headlines the page instead.
  const headline = level === "reliable" ? { icon: "✓", label: "Likely Real", cls: "verdict-real",
      sub: "Live evidence supports this article." }
    : level === "unreliable" ? { icon: "!", label: "Likely Fake", cls: "verdict-fake",
      sub: "Live evidence counts against this article." }
    : level === "mixed" ? { icon: "?", label: "Inconclusive", cls: "verdict-mixed",
      sub: "Evidence is mixed or missing. Check the panel below and the sources yourself." }
    : null;
  if (v) renderVerification(v);
  if (headline) {
    $("verdict").className = "verdict " + headline.cls;
    $("verdict-icon").textContent = headline.icon;
    $("verdict-label").textContent = headline.label;
    $("verdict-sub").textContent = headline.sub;
  } else {
    $("verdict").className = "verdict " + (fake ? "verdict-fake" : "verdict-real");
    $("verdict-icon").textContent = fake ? "!" : "✓";
    $("verdict-label").textContent = fake ? "Fake News" : "Real News";
    $("verdict-sub").textContent = fake ? "This reads like misleading or fabricated content." : "This reads like reliable reporting.";
  }
  $("verdict-pill").textContent = headline
    ? `WRITING-STYLE MODEL: ${fake ? "FAKE" : "REAL"} ${d.confidence.toFixed(1)}% (weak signal)`
    : `${d.confidence.toFixed(1)}% CONFIDENCE`;
  const rows = { conf: d.confidence, real: d.probabilities.real, fake: d.probabilities.fake };
  for (const [k, v] of Object.entries(rows)) {
    $(k + "-value").textContent = v.toFixed(1) + "%";
    animateWidth($(k + "-fill"), v);
  }
  const a = d.article;
  $("scraped").hidden = !a;
  if (a) {
    $("scraped-title").textContent = a.title || "(No title found)";
    const bits = [a.domain, a.word_count + " words", a.authors.join(", "), a.publish_date && a.publish_date.slice(0, 10)];
    $("scraped-meta").textContent = bits.filter(Boolean).join(" | ");
    $("scraped-text").textContent = a.text_snippet;
  }
  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

$("form-manual").addEventListener("submit", (e) => {
  e.preventDefault();
  const btn = e.submitter || e.target.querySelector("button[type=submit]");
  submit("/predict/manual", { title: $("title").value, text: $("body").value }, btn);
});
$("form-url").addEventListener("submit", (e) => {
  e.preventDefault();
  const btn = e.submitter || e.target.querySelector("button[type=submit]");
  submit("/predict/url", { url: $("url").value }, btn);
});
document.querySelectorAll(".chip").forEach((c) =>
  c.addEventListener("click", () => {
    const s = SAMPLES[c.dataset.sample];
    $("title").value = s.title; $("body").value = s.text; countWords();
    $("form-manual").requestSubmit();
  })
);

async function checkHealth() {
  try {
    const h = await (await fetch(API + "/health")).json();
    st.ready = h.status === "ready";
    setBanner(st.ready ? "" : h.message, h.status === "unavailable");
    $("badge-dot").classList.toggle("ok", st.ready);
    const acc = h.metrics && h.metrics.accuracy;
    $("badge-text").textContent = st.ready ? `Model ready${acc ? `, ${(acc * 100).toFixed(1)}% test accuracy` : ""}` : "Model not ready";
    if (h.status === "loading" || h.status === "training") setTimeout(checkHealth, 4000);
  } catch {
    st.ready = false;
    setBanner("Can't reach the API. Start it with: python -m uvicorn backend.app:app", true);
    $("badge-text").textContent = "Server offline";
  }
  updateButtons();
}
checkHealth();
