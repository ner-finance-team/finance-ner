// 네이버 기사 페이지에서: 제목·본문 글자를 모아 NER API로 보내고, 결과를 같은 화면 위에 칠한다.
(() => {
  if (window.__nerx) return; window.__nerx = true;
  const SKIP = ".end_photo_org, .img_desc, figure, table, script, style, .vod_player_wrap, .ab_photo, .nerx-tag";
  const S = { api: "http://localhost:8000", labels: {}, tags: [], res: null, off: new Set(), min: 0, nodes: {} };

  // ---------- 1. 글자 모으기: 텍스트 조각마다 [시작, 끝) 위치를 기억 ----------
  function collect(root) {
    let text = "", map = [];
    const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT, {
      acceptNode: n => n.nodeType === 1 && n.matches(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
    for (let n = walk.nextNode(); n; n = walk.nextNode()) {
      if (n.nodeType === 1) { if (n.tagName === "BR" || /^(DIV|P|STRONG|H\d)$/.test(n.tagName) && text && !text.endsWith("\n")) text += "\n"; continue; }
      if (!n.nodeValue) continue;
      map.push({ node: n, start: text.length, end: text.length + n.nodeValue.length });
      text += n.nodeValue;
    }
    return { text, map };
  }

  // ---------- 2. 칠하기: 개체 범위가 걸친 텍스트 조각을 잘라 <mark>로 감싼다 ----------
  function paint(fieldId, map, ents) {
    const per = new Map();
    ents.forEach((e, k) => map.forEach(m => {
      const s = Math.max(e.start, m.start), t = Math.min(e.end, m.end);
      if (s < t) { if (!per.has(m)) per.set(m, []); per.get(m).push({ s: s - m.start, t: t - m.start, e, first: s === e.start, last: t === e.end }); }
    }));
    for (const [m, rs] of per) {
      rs.sort((a, b) => b.s - a.s || b.t - a.t);       // 뒤에서부터 잘라야 앞쪽 위치가 안 밀린다
      let node = m.node;
      for (const r of rs) {
        if (r.t > node.length || r.s >= r.t) continue;  // 이미 칠한 범위와 겹치면 건너뜀(서버가 정리하지만 혹시 몰라)
        const tail = node.splitText(r.t); const mid = node.splitText(r.s);
        const mk = document.createElement("mark");
        const l = S.labels[r.e.tag] || { bg: "#e5e7eb", fg: "#374151" };
        mk.className = "nerx-ent"; mk.style.setProperty("--nerx-bg", l.bg); mk.style.setProperty("--nerx-fg", l.fg);
        mk.dataset.tag = r.e.tag; mk.dataset.score = r.e.score;
        if (r.first) mk.id = `nerx-${fieldId}-${r.e.start}`;
        mid.replaceWith(mk); mk.appendChild(mid);
        if (r.last) { const sp = document.createElement("span"); sp.className = "nerx-tag"; sp.textContent = r.e.tag; mk.appendChild(sp); }
        void tail;
      }
    }
  }
  function clear() {
    document.querySelectorAll("mark.nerx-ent").forEach(mk => { mk.querySelector(".nerx-tag")?.remove(); const p = mk.parentNode; mk.replaceWith(...mk.childNodes); p.normalize(); });
    S.host?.shadowRoot.querySelector(".report")?.remove(); S.reportHost?.remove(); S.reportHost = null;
  }

  // ---------- 3. API (background.js 를 거쳐 호출) ----------
  const api = (path, body) => new Promise((ok, no) => chrome.runtime.sendMessage({ type: "api", url: S.api + path, body },
    r => r && r.ok ? ok(r.data) : no(new Error(r ? r.error : "확장 오류"))));

  // ---------- 4. 시간 ----------
  const kst = s => s ? new Date(s.replace(" ", "T") + "+09:00") : null;
  const ago = d => { if (!d) return ""; const m = Math.round((Date.now() - d) / 6e4);
    return m < 1 ? "방금 전" : m < 60 ? `${m}분 전` : m < 1440 ? `${Math.floor(m / 60)}시간 ${m % 60}분 전` : `${Math.floor(m / 1440)}일 전`; };
  const hhmm = d => d.toLocaleString("ko-KR", { timeZone: "Asia/Seoul", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
  const pubDate = () => kst(document.querySelector("span._ARTICLE_DATE_TIME")?.dataset.dateTime);

  // ---------- 5. 패널 (섀도 DOM: 네이버 CSS와 분리) ----------
  const CSS = `
  :host{all:initial}
  *{box-sizing:border-box;font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif}
  .panel{position:fixed;top:84px;right:20px;z-index:2147483646;width:300px;background:#fff;color:#1d2433;border:1px solid #e4e7ec;
    border-radius:16px;box-shadow:0 12px 32px rgba(16,24,40,.18);overflow:hidden;font-size:13px;line-height:1.5}
  .panel.min .body{display:none}
  .head{display:flex;align-items:center;gap:8px;padding:12px 14px;background:linear-gradient(135deg,#2563eb,#7c3aed);color:#fff;cursor:move}
  .head b{font-size:14px}.head .sp{flex:1}
  .head button{all:unset;cursor:pointer;opacity:.9;font-size:16px;padding:0 4px}
  .body{padding:12px 14px 14px}
  .model{font-size:12px;color:#667085;margin-bottom:10px;display:flex;align-items:center;gap:6px}
  .dot{width:8px;height:8px;border-radius:50%;background:#12b76a;flex:none}.dot.bad{background:#f04438}
  .model select{flex:1;min-width:0;font-size:12px;color:#1d2433;border:1px solid #e4e7ec;border-radius:8px;padding:4px 6px;background:#fff;cursor:pointer}
  .model select:disabled{color:#98a2b3;cursor:wait}
  .tagbtn{display:flex;justify-content:space-between;width:100%;margin:-4px 0 10px;border:1px solid #e4e7ec;background:#fff;border-radius:8px;padding:5px 9px;font-size:12px;color:#475467;cursor:pointer}
  .tagbtn:hover{border-color:#c7d2fe}.tagbtn.on{background:#eff4ff;border-color:#2563eb;color:#1d4ed8}
  .tags{display:none;margin:-6px 0 10px;border:1px solid #eaecf0;border-radius:10px;padding:4px}.tags.on{display:block}
  .tags .hint{font-size:11px;color:#98a2b3;padding:3px 6px 4px}
  .tg{display:flex;align-items:center;gap:8px;padding:4px 6px;border-radius:6px;cursor:pointer}.tg:hover{background:#f9fafb}
  .tg .nm{flex:1}.tg code{font-size:11px;color:#667085}.tg .n{font-weight:700;min-width:20px;text-align:right}.tg.off{opacity:.4}
  .run{width:100%;border:0;border-radius:10px;padding:11px;font-size:14px;font-weight:700;background:#2563eb;color:#fff;cursor:pointer}
  .run:hover{background:#1d4ed8}.run:disabled{background:#98a2b3;cursor:wait}
  .row2{display:flex;gap:6px;margin-top:6px}.row2 button{flex:1;border:1px solid #e4e7ec;background:#fff;border-radius:8px;padding:6px;cursor:pointer;color:#475467;font-size:12px}
  .times{display:flex;flex-direction:column;gap:4px;margin:12px 0 4px}
  .t{display:flex;justify-content:space-between;gap:8px;border:1px solid #eaecf0;border-radius:8px;padding:5px 9px;color:#475467}.t span{white-space:nowrap}
  .t b{color:#1d2433}.t.hot{border-color:#fda29b;background:#fef3f2}.t.hot b{color:#b42318}
  .sec{margin-top:12px;font-size:12px;font-weight:700;color:#667085}
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
  .chip{display:inline-flex;align-items:center;gap:5px;border-radius:999px;padding:3px 9px;font-size:12px;font-weight:700;cursor:pointer;
    border:1.5px solid var(--fg);background:var(--bg);color:#1d2433;user-select:none}
  .chip i{font-style:normal;opacity:.75}.chip.off{background:#fff;color:#98a2b3;border-style:dashed}
  .conf{display:flex;align-items:center;gap:8px;margin-top:10px;color:#475467}.conf input{flex:1}
  .msg{margin-top:10px;color:#667085}.err{color:#d92d20}
  .set{margin-top:10px;display:none}.set.on{display:block}.set input{width:100%;padding:6px 8px;border:1px solid #e4e7ec;border-radius:8px;font-size:12px}
  .spin{display:inline-block;width:12px;height:12px;border:2px solid #fff;border-top-color:transparent;border-radius:50%;animation:s 1s linear infinite;vertical-align:-2px;margin-right:6px}
  .spin.dark{border-color:#2563eb;border-top-color:transparent}
  @keyframes s{to{transform:rotate(360deg)}}
  .tip{position:fixed;z-index:2147483647;pointer-events:none;background:#111827;color:#fff;font-size:12.5px;padding:6px 10px;border-radius:8px;opacity:0;transition:opacity .1s;white-space:nowrap}
  /* 리포트 */
  .report{margin:28px 0 8px;border:1px solid #e4e7ec;border-radius:16px;overflow:hidden;background:#fff;color:#1d2433;box-shadow:0 4px 14px rgba(16,24,40,.06)}
  .rhead{padding:14px 18px;background:linear-gradient(135deg,#eff4ff,#f5f3ff);border-bottom:1px solid #e4e7ec}
  .rhead h3{margin:0;font-size:17px}.rhead p{margin:2px 0 0;color:#667085;font-size:12.5px}
  .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:14px 18px 4px}
  .stat{border:1px solid #eaecf0;border-radius:12px;padding:8px 10px}.stat .k{font-size:11.5px;color:#667085}.stat .v{font-size:20px;font-weight:800}
  .group{padding:12px 18px;border-top:1px solid #f2f4f7}
  .group h4{margin:0 0 8px;font-size:14px;display:flex;align-items:center;gap:8px}
  .sw{width:12px;height:12px;border-radius:4px;background:var(--bg);border:1.5px solid var(--fg)}
  .group h4 small{color:#667085;font-weight:500}
  .ents{display:flex;flex-wrap:wrap;gap:6px}
  .ent{border:1px solid #eaecf0;background:var(--bg);border-radius:8px;padding:3px 9px;font-size:13.5px;cursor:pointer}
  .ent:hover{border-color:var(--fg)}.ent small{color:#475467;margin-left:4px;font-weight:700}
  .group.off{opacity:.35}
  .foot{padding:10px 18px 14px;color:#98a2b3;font-size:11.5px}
  /* Claude 비교 */
  .cmp{padding:14px 18px;border-top:1px solid #e4e7ec;background:#fcfcfd}
  .cmpbtn{border:0;border-radius:10px;padding:9px 14px;font-size:13.5px;font-weight:700;background:#111827;color:#fff;cursor:pointer}
  .cmpbtn:hover{background:#374151}.cmpbtn:disabled{background:#98a2b3;cursor:wait}
  .cmphint{display:inline-block;margin-left:10px;color:#667085;font-size:12px}
  .score{display:flex;align-items:center;gap:16px;margin:12px 0 6px}
  .score .big{font-size:34px;font-weight:800;color:#1d4ed8}.score .sub{color:#475467;font-size:12.5px;line-height:1.6}
  .bar{display:flex;height:12px;border-radius:6px;overflow:hidden;margin:8px 0 4px;background:#f2f4f7}
  .bar span{display:block;height:100%}
  .leg{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:#475467;margin-bottom:6px}.leg i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:4px;vertical-align:-1px}
  .dsec{margin-top:10px}.dsec h5{margin:0 0 6px;font-size:13px;display:flex;gap:6px;align-items:center}.dsec h5 small{color:#667085;font-weight:500}
  .drow{display:flex;flex-wrap:wrap;gap:6px}
  .dent{border:1px solid #eaecf0;border-radius:8px;padding:3px 8px;font-size:12.5px;background:#fff;cursor:pointer}.dent:hover{border-color:#2563eb}
  .dent b{font-weight:700}.dent .o{color:#1d4ed8}.dent .c{color:#b45309}.dent .ar{color:#98a2b3;margin:0 3px}
  .cmpfoot{margin-top:10px;color:#98a2b3;font-size:11.5px}`;

  function el(html) { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; }
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const name = t => (S.labels[t] && S.labels[t].name) || t;
  const vars = t => { const l = S.labels[t] || { bg: "#e5e7eb", fg: "#374151" }; return `--bg:${l.bg};--fg:${l.fg}`; };

  function buildPanel() {
    const host = document.createElement("div"); host.id = "nerx-host"; document.documentElement.appendChild(host);
    const root = host.attachShadow({ mode: "open" }); S.host = host; S.root = root;
    root.innerHTML = `<style>${CSS}</style>
      <div class="panel" id="p">
        <div class="head" id="drag"><b>🔎 개체명 인식</b><span class="sp"></span>
          <button id="gear" title="서버 주소">⚙</button><button id="min" title="접기">–</button></div>
        <div class="body">
          <div class="model"><span class="dot" id="dot"></span><select id="model" title="모델 선택" disabled><option>서버 확인 중…</option></select></div>
          <button class="tagbtn" id="tagbtn" title="이 모델이 찾는 태그 목록"><span id="tagcnt">🔍 태그 목록</span><span>▾</span></button>
          <div class="tags" id="tags"></div>
          <div class="set" id="set"><input id="apiUrl" placeholder="http://localhost:8000"></div>
          <button class="run" id="run">이 기사에서 개체 찾기</button>
          <div class="row2"><button id="clr">표시 지우기</button><button id="rep">리포트로 이동 ↓</button></div>
          <div class="times" id="times"></div>
          <div id="legendBox"></div>
          <div class="msg" id="msg">버튼을 누르면 제목·본문의 개체를 태그 색으로 칠합니다.</div>
        </div>
      </div><div class="tip" id="tip"></div>`;
    const $ = s => root.getElementById(s);
    $("run").onclick = run; $("clr").onclick = () => { clear(); S.res = null; renderSide(); };
    $("rep").onclick = () => S.reportHost?.scrollIntoView({ behavior: "smooth", block: "start" });
    $("min").onclick = () => $("p").classList.toggle("min");
    $("gear").onclick = () => $("set").classList.toggle("on");
    $("tagbtn").onclick = () => { $("tagbtn").classList.toggle("on"); $("tags").classList.toggle("on"); };
    $("model").onchange = e => switchModel(e.target.value, e.target.selectedOptions[0].textContent);
    $("apiUrl").value = S.api;
    $("apiUrl").onchange = e => { S.api = e.target.value.replace(/\/$/, ""); chrome.storage.local.set({ api: S.api }); health(); };
    // 끌어서 옮기기
    let dx, dy; $("drag").onmousedown = e => { if (e.target.tagName === "BUTTON") return; const r = $("p").getBoundingClientRect(); dx = e.clientX - r.left; dy = e.clientY - r.top;
      const mv = ev => { Object.assign($("p").style, { left: ev.clientX - dx + "px", top: ev.clientY - dy + "px", right: "auto" }); };
      const up = () => { removeEventListener("mousemove", mv); removeEventListener("mouseup", up); };
      addEventListener("mousemove", mv); addEventListener("mouseup", up); };
    renderTimes();
  }

  function renderTimes(extra) {
    const d = pubDate(), hot = d && Date.now() - d < 3600e3, b = S.root.getElementById("times");
    b.innerHTML = (d ? `<div class="t${hot ? " hot" : ""}"><span>🕒 기사 입력</span><b>${hhmm(d)} · ${ago(d)}</b></div>` : "")
      + (extra ? `<div class="t"><span>⚡ 분석</span><b>${extra}</b></div>` : "");
  }

  function renderSide() {
    const box = S.root.getElementById("legendBox");
    renderTags();
    if (!S.res) { box.innerHTML = ""; return; }
    const g = S.res.report.by_tag;
    box.innerHTML = `<div class="sec">태그 (눌러서 켜기/끄기)</div><div class="chips">${g.map(x =>
      `<span class="chip${S.off.has(x.tag) ? " off" : ""}" data-tag="${x.tag}" style="${vars(x.tag)}">${esc(name(x.tag))} <i>${x.total}</i></span>`).join("")}</div>
      <div class="conf">확신도 <input type="range" id="conf" min="0" max="0.95" step="0.05" value="${S.min}"><b id="cv">${Math.round(S.min * 100)}%</b>↑</div>`;
    box.querySelectorAll(".chip").forEach(c => c.onclick = () => toggleTag(c.dataset.tag));
    S.root.getElementById("conf").oninput = e => { S.min = +e.target.value; S.root.getElementById("cv").textContent = Math.round(S.min * 100) + "%"; filter(); };
  }
  function toggleTag(t) { S.off.has(t) ? S.off.delete(t) : S.off.add(t); filter(); renderSide(); }

  // 🔍 태그 목록: 이 모델이 찾는 태그 전부(못 찾은 태그 포함) + 이 기사에서 찾은 횟수
  function renderTags() {
    const box = S.root.getElementById("tags"), cnt = {};
    S.res?.report.by_tag.forEach(x => cnt[x.tag] = x.total);
    box.innerHTML = `<div class="hint">이 모델이 찾는 태그 ${S.tags.length}종 · ${S.res ? "숫자는 이 기사에서 찾은 횟수, 눌러서 켜기/끄기" : "개체를 찾으면 횟수가 나옵니다"}</div>`
      + S.tags.map(t => `<div class="tg${S.off.has(t) ? " off" : ""}" data-tag="${t}" style="${vars(t)}"><span class="sw"></span><span class="nm">${esc(name(t))}</span><code>${t}</code><span class="n">${S.res ? cnt[t] || 0 : ""}</span></div>`).join("");
    if (S.res) box.querySelectorAll(".tg").forEach(r => r.onclick = () => toggleTag(r.dataset.tag));
  }

  function filter() {
    document.querySelectorAll("mark.nerx-ent").forEach(m => m.classList.toggle("nerx-off", S.off.has(m.dataset.tag) || +m.dataset.score < S.min));
    S.reportHost?.shadowRoot.querySelectorAll(".group").forEach(g => g.classList.toggle("off", S.off.has(g.dataset.tag)));
  }

  function buildReport(r, sec) {
    const body = document.querySelector("#dic_area"); if (!body) return;
    const host = document.createElement("div"); host.id = "nerx-report"; body.after(host); S.reportHost = host;
    const root = host.attachShadow({ mode: "open" }), g = r.report.by_tag;
    root.innerHTML = `<style>${CSS}</style><div class="report">
      <div class="rhead"><h3>📊 개체 리포트</h3><p>이 기사에서 모델이 찾은 개체를 태그별로 모았습니다. 개체를 누르면 본문 위치로 이동합니다.</p></div>
      <div class="stats">${[["찾은 개체", r.report.entities], ["태그 종류", g.length], ["문장 수", r.sentences], ["처리 시간", sec + "초"]]
        .map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("")}</div>
      ${g.map(x => `<div class="group" data-tag="${x.tag}" style="${vars(x.tag)}"><h4><span class="sw"></span>${esc(name(x.tag))} <small>${x.tag} · ${x.unique}종 ${x.total}회</small></h4>
        <div class="ents">${x.items.map(i => `<span class="ent" data-jump="nerx-${i.first.id}-${i.first.start}" title="평균 확신도 ${Math.round(i.score * 100)}%">${esc(i.text)}${i.count > 1 ? `<small>×${i.count}</small>` : ""}</span>`).join("")}</div></div>`).join("")}
      ${S.isApi ? "" : `<div class="cmp" id="cmp"><button class="cmpbtn" id="cmpbtn">🤖 Claude와 비교</button><span class="cmphint">같은 기사를 Claude(기준서 v1.2)에게 라벨링시켜 우리 모델과 얼마나 같은지 봅니다 · API 요금이 듭니다(같은 기사는 저장된 결과를 다시 써서 한 번만)</span><div id="cmpres"></div></div>`}
      <div class="foot">모델: ${esc(r.model)}${r.members ? ` (${r.members.map(esc).join(" + ")} 다수결)` : ""} · 자동 예측이라 틀린 표시가 있을 수 있습니다.</div></div>`;
    root.querySelectorAll("[data-jump]").forEach(e => e.onclick = () => jump(e.dataset.jump));
    const cb = root.getElementById("cmpbtn"); if (cb) cb.onclick = () => compare(root);
  }
  function jump(id) {
    const m = document.getElementById(id); if (!m) return;
    m.scrollIntoView({ behavior: "smooth", block: "center" }); m.classList.remove("nerx-flash"); void m.offsetWidth; m.classList.add("nerx-flash");
  }

  // ---------- Claude 비교: 같은 글을 Claude 에 보내 우리 결과와 비교 ----------
  async function compare(root, force = false) {
    const btn = root.getElementById("cmpbtn"), box = root.getElementById("cmpres");
    btn.disabled = true; btn.innerHTML = `<span class="spin"></span>Claude 결과 확인 중… (저장본이 없으면 10~40초)`; box.innerHTML = "";
    try {
      // 같은 기사의 Claude 결과는 서버에 저장되어, 다른 모델로 바꿔 비교해도 다시 호출하지 않는다(force=true 일 때만 새로 호출)
      const c = await api("/api/compare", { texts: S.texts, ours: S.res.texts.map(t => ({ id: t.id, entities: t.entities })), force });
      const n = { same: c.same.length, tag: c.tag.length, boundary: c.boundary.length, oo: c.only_ours.length, oc: c.only_claude.length };
      const tot = n.same + n.tag + n.boundary + n.oo + n.oc || 1, pct = v => (v / tot * 100).toFixed(1) + "%";
      const COL = { same: "#16a34a", tag: "#7c3aed", boundary: "#f59e0b", oo: "#2563eb", oc: "#d97706" };
      const tagN = t => esc(name(t));
      const item = (id, start, html) => `<span class="dent" data-jump="nerx-${id}-${start}">${html}</span>`;
      const sec = (title, sub, arr, fn) => arr.length ? `<div class="dsec"><h5>${title} <small>${sub} · ${arr.length}개</small></h5><div class="drow">${arr.map(fn).join("")}</div></div>` : "";
      box.innerHTML = `
        <div class="score"><div class="big">${(c.f1 * 100).toFixed(1)}%</div>
          <div class="sub">우리 모델과 Claude의 <b>일치도(F1)</b> · 위치·태그까지 같은 개체 ${c.match}개<br>
          우리 ${c.n_ours}개 중 ${(c.precision * 100).toFixed(0)}%, Claude ${c.n_claude}개 중 ${(c.recall * 100).toFixed(0)}%가 같음</div></div>
        <div class="bar">${Object.entries(n).map(([k, v]) => v ? `<span style="width:${pct(v)};background:${COL[k]}" title="${v}개"></span>` : "").join("")}</div>
        <div class="leg"><span><i style="background:${COL.same}"></i>같음 ${n.same}</span><span><i style="background:${COL.tag}"></i>태그만 다름 ${n.tag}</span>
          <span><i style="background:${COL.boundary}"></i>범위 다름 ${n.boundary}</span><span><i style="background:${COL.oo}"></i>우리만 ${n.oo}</span><span><i style="background:${COL.oc}"></i>Claude만 ${n.oc}</span></div>
        ${sec("🏷 태그만 다름", "같은 글자, 다른 태그", c.tag, x => item(x.id, x.start, `<b>${esc(x.text)}</b> <span class="o">우리 ${tagN(x.ours)}</span><span class="ar">↔</span><span class="c">Claude ${tagN(x.claude)}</span>`))}
        ${sec("📏 범위 다름", "겹치지만 경계가 다름", c.boundary, x => item(x.id, x.start, `<span class="o">우리 <b>${esc(x.ours_text)}</b></span><span class="ar">↔</span><span class="c">Claude <b>${esc(x.claude_text)}</b></span>`))}
        ${sec("🔵 우리 모델만 찾음", "Claude는 안 담", c.only_ours, x => item(x.id, x.start, `<b>${esc(x.text)}</b> <span class="o">${tagN(x.ours)}</span>`))}
        ${sec("🟠 Claude만 찾음", "우리 모델은 안 담", c.only_claude, x => `<span class="dent" title="본문 위치: ${x.id === "title" ? "제목" : "본문"}"><b>${esc(x.text)}</b> <span class="c">${tagN(x.claude)}</span></span>`)}
        ${c.cached ? `<div class="cmpfoot" style="color:#067647;font-weight:700">💾 저장된 Claude 결과 사용 · API 호출 없음 · 이번 요금 $0 (처음 받은 때 ${esc(c.saved_at || "")}) <button class="cmpbtn" id="cmpforce" style="padding:3px 9px;font-size:12px;margin-left:6px">Claude 새로 호출</button></div>` : ""}
        <div class="cmpfoot">${esc(c.claude)} · ${c.cached ? "처음 받을 때 " : ""}${c.seconds}초 · 토큰 입력 ${c.usage.input_tokens ?? "-"} / 출력 ${c.usage.output_tokens ?? "-"} / 캐시 ${c.usage.cache_read_input_tokens ?? 0} · ${c.cached ? "처음 받을 때 " : ""}약 $${c.cost_usd_est} (가격표 기준 추정) · Claude 라벨도 정답은 아닙니다</div>`;
      box.querySelectorAll("[data-jump]").forEach(e => e.onclick = () => jump(e.dataset.jump));
      const fb = root.getElementById("cmpforce"); if (fb) fb.onclick = () => compare(root, true);
      btn.textContent = "🤖 다시 비교";
    } catch (e) { box.innerHTML = `<div class="cmpfoot" style="color:#d92d20">비교 실패: ${esc(e.message)}</div>`; btn.textContent = "🤖 Claude와 비교"; }
    btn.disabled = false;
  }

  // ---------- 6. 실행 ----------
  async function health() {
    const $ = s => S.root.getElementById(s);
    try { const h = await api("/api/health"), m = await api("/api/models"); S.labels = await api("/api/labels"); S.tags = h.tags;
      S.isApi = h.model === "claude-api";
      const opt = x => `<option value="${esc(x.name)}"${x.name === h.model ? " selected" : ""}>${x.kind === "api"
        ? `☁ 상용 API 키 사용 (Claude · 금융 8태그)${x.ready ? "" : " · 키 미설정"}`
        : x.kind === "ensemble" ? `★ 앙상블 (03 + 12 + 03s 다수결)` : esc(x.name) + (x.pretok ? " · 형태소" : "")}</option>`;
      $("model").innerHTML = `<optgroup label="내 모델 (폴더)">${m.models.filter(x => x.kind === "local").map(opt).join("")}</optgroup>`
        + (m.models.some(x => x.kind === "ensemble") ? `<optgroup label="앙상블">${m.models.filter(x => x.kind === "ensemble").map(opt).join("")}</optgroup>` : "")
        + `<optgroup label="상용 API">${m.models.filter(x => x.kind === "api").map(opt).join("")}</optgroup>`;
      $("model").disabled = false; $("tagcnt").textContent = `🔍 이 모델의 태그 ${h.tags.length}종 보기`; $("dot").classList.remove("bad"); renderTags(); }
    catch (e) { $("model").innerHTML = `<option>서버 연결 안 됨</option>`; $("model").disabled = true;
      $("msg").innerHTML = `<span class="err">${esc(e.message)}</span>`; $("dot").classList.add("bad"); }
  }

  // 모델 바꾸기: 서버의 모델이 바뀌므로(웹 페이지 포함) 지금 표시는 지우고 다시 찾게 한다
  async function switchModel(nm, label) {
    const $ = s => S.root.getElementById(s);
    $("model").disabled = $("run").disabled = true;
    $("msg").innerHTML = `<span class="spin dark"></span>${esc(label)} 불러오는 중… (몇 초 걸립니다)`;
    try { await api("/api/model", { name: nm });
      clear(); S.res = null; S.off = new Set(); renderSide(); renderTimes();
      $("run").textContent = "이 기사에서 개체 찾기"; $("msg").textContent = `모델을 ${label}(으)로 바꿨습니다. 버튼을 눌러 개체를 찾으세요.`; }
    catch (e) { $("msg").innerHTML = `<span class="err">모델 교체 실패: ${esc(e.message)}</span>`; }
    $("run").disabled = false; await health();
  }

  async function run() {
    const $ = s => S.root.getElementById(s), btn = $("run");
    const tEl = document.querySelector("#title_area"), bEl = document.querySelector("#dic_area");
    if (!tEl || !bEl) { $("msg").innerHTML = `<span class="err">기사 제목·본문을 찾지 못했습니다.</span>`; return; }
    clear(); btn.disabled = true; btn.innerHTML = `<span class="spin"></span>개체 찾는 중…`; $("msg").textContent = "";
    try {
      const T = collect(tEl), B = collect(bEl), t0 = performance.now();
      S.texts = [{ id: "title", text: T.text }, { id: "body", text: B.text }];
      const r = await api("/api/ner", { texts: S.texts });
      const sec = ((performance.now() - t0) / 1000).toFixed(1);
      paint("title", T.map, r.texts[0].entities); paint("body", B.map, r.texts[1].entities);
      S.res = r; S.off = new Set(); renderSide(); filter(); buildReport(r, sec);
      renderTimes(`${new Date().toLocaleTimeString("ko-KR", { timeZone: "Asia/Seoul" })} · ${sec}초`);
      $("msg").textContent = `개체 ${r.report.entities}개를 찾았습니다. 마우스를 올리면 태그와 확신도가 보입니다.`;
    } catch (e) { $("msg").innerHTML = `<span class="err">실패: ${esc(e.message)}</span>`; }
    btn.disabled = false; btn.textContent = "다시 찾기";
  }

  // 형광펜 위에 마우스를 올리면 태그·확신도
  document.addEventListener("mouseover", e => {
    const tip = S.root?.getElementById("tip"), m = e.target.closest?.("mark.nerx-ent"); if (!tip) return;
    if (!m || m.classList.contains("nerx-off")) { tip.style.opacity = 0; return; }
    tip.textContent = `${name(m.dataset.tag)} (${m.dataset.tag}) · 확신도 ${Math.round(m.dataset.score * 100)}%`; tip.style.opacity = 1;
  });
  document.addEventListener("mousemove", e => { const tip = S.root?.getElementById("tip"); if (tip) { tip.style.left = e.clientX + 12 + "px"; tip.style.top = e.clientY + 14 + "px"; } });
  setInterval(() => S.root && renderTimes(S.root.querySelector("#times .t:nth-child(2) b")?.textContent), 30000);
  chrome.runtime.onMessage.addListener(m => { if (m.type === "toggle") S.host.style.display = S.host.style.display === "none" ? "" : "none"; });

  chrome.storage.local.get("api", v => { if (v.api) S.api = v.api; buildPanel(); health(); });
})();
