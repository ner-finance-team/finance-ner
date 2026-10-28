// 기사 페이지(https)에서 로컬 API(http)를 바로 부르지 않고, 여기서 대신 부른다.
chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg.type !== "api") return;
  fetch(msg.url, msg.body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(msg.body) } : {})
    .then(async r => { const j = await r.json().catch(() => ({})); reply(r.ok ? { ok: true, data: j } : { ok: false, error: j.detail || r.statusText }); })
    .catch(e => reply({ ok: false, error: "서버에 연결할 수 없습니다 (" + msg.url.split("/api")[0] + ")" }));
  return true;   // 비동기 응답
});
// 툴바 아이콘을 누르면 패널을 열고 닫는다
chrome.action.onClicked.addListener(tab => { chrome.tabs.sendMessage(tab.id, { type: "toggle" }).catch(() => {}); });
