"""Local web console for the Trade Desk ("push the button" UI).

Standard library only. Binds to 127.0.0.1, requires a random per-session token on every
API call and checks the Host header (DNS-rebinding guard). Nothing here can place an
order without an explicit click.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)


class ConsoleApp:
    def __init__(self, desk, refresh_fn, host="127.0.0.1", port=8765, token=None):
        self.desk = desk
        self.refresh_fn = refresh_fn          # callable() -> dict, loads data and calls desk.refresh
        self.host, self.port = host, port
        self.token = token or secrets.token_urlsafe(24)
        self.busy = False
        self.last_error = ""
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/?t={self.token}"

    # ---------------------------------------------------------------- actions
    def _refresh_bg(self):
        try:
            self.refresh_fn()
            self.last_error = ""
        except Exception as exc:  # surfaced in the UI
            self.last_error = f"Refresh failed: {exc}"
            log.error("refresh failed\n%s", traceback.format_exc())
        finally:
            self.busy = False

    def dispatch(self, action: str, body: dict) -> dict:
        d = self.desk
        if action == "refresh":
            with self._lock:
                if self.busy:
                    return {"message": "Refresh already running"}
                self.busy = True
            threading.Thread(target=self._refresh_bg, daemon=True).start()
            return {"message": "Refreshing - loading data and running the models..."}
        if action == "approve":
            return d.approve(int(body["id"]), body.get("override_reason") or None,
                             body.get("qty") and int(body["qty"]), body.get("limit_price") and float(body["limit_price"]))
        if action == "skip":
            d.skip(int(body["id"]), body.get("reason", ""))
            return {"message": "Skipped"}
        if action == "cancel":
            d.cancel(int(body["id"]))
            return {"message": "Cancelled"}
        if action == "fill":
            return d.confirm_fill(int(body["id"]), int(body["qty"]), float(body["price"]))
        if action == "exit":
            return d.exit_position(body["sleeve"], body["symbol"])
        if action == "stopped":
            return d.stopped_out(body["sleeve"], body["symbol"], float(body["price"]), body.get("reason", "stop"))
        if action == "stop":
            return d.set_stop(body["sleeve"], body["symbol"], float(body["stop"]) if body.get("stop") else None,
                              float(body["target"]) if body.get("target") else None)
        if action == "kill":
            return d.kill_switch(body.get("reason", "console"))
        if action == "resume":
            return d.resume()
        if action == "flatten":
            return d.flatten(body.get("confirm", ""))
        if action == "sync":
            return d.sync()
        if action == "rebalance":
            d.force_rebalance()
            return {"message": "Core rebalance will be proposed on the next refresh"}
        raise ValueError(f"Unknown action {action}")

    # ---------------------------------------------------------------- server
    def make_handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # quiet
                log.debug(fmt, *args)

            def _host_ok(self) -> bool:
                host = (self.headers.get("Host") or "").split(":")[0]
                return host in ("127.0.0.1", "localhost", app.host)

            def _send(self, code: int, payload, ctype="application/json"):
                body = payload if isinstance(payload, bytes) else json.dumps(payload, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(body)

            def _authed(self) -> bool:
                return secrets.compare_digest(self.headers.get("X-Desk-Token", ""), app.token)

            def do_GET(self):
                if not self._host_ok():
                    return self._send(403, {"error": "bad host"})
                u = urlparse(self.path)
                if u.path == "/":
                    if parse_qs(u.query).get("t", [""])[0] != app.token:
                        return self._send(403, b"Missing or wrong token - use the URL printed by `indquant console`.",
                                          "text/plain")
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if u.path == "/api/state":
                    if not self._authed():
                        return self._send(403, {"error": "bad token"})
                    st = app.desk.state()
                    st.update(busy=app.busy, error=app.last_error)
                    return self._send(200, st)
                return self._send(404, {"error": "not found"})

            def do_POST(self):
                if not self._host_ok() or not self._authed():
                    return self._send(403, {"error": "forbidden"})
                u = urlparse(self.path)
                if not u.path.startswith("/api/"):
                    return self._send(404, {"error": "not found"})
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    return self._send(200, {"ok": True, **(app.dispatch(u.path[5:], body) or {})})
                except (ValueError, KeyError) as exc:
                    return self._send(400, {"ok": False, "error": str(exc)})
                except Exception as exc:
                    log.error("action failed\n%s", traceback.format_exc())
                    return self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        return Handler

    def serve(self, open_browser: bool = True):
        server = ThreadingHTTPServer((self.host, self.port), self.make_handler())
        print(f"\nTrade desk console: {self.url}\n(Ctrl+C to stop. The link contains your session token - don't share it.)")
        if open_browser:
            import webbrowser

            threading.Timer(0.8, lambda: webbrowser.open(self.url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return server


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trade Desk</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#14171a;--muted:#5f6b76;--line:#dde2e7;--accent:#1f6feb;--buy:#1a7f37;--sell:#cf222e;
--warn:#9a6700;--warnbg:#fff4d6;--vetobg:#ffe5e5;--okbg:#e6f6ea;--btn:#eef1f4}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--muted:#8d96a0;--line:#30363d;--accent:#4493f8;
--buy:#3fb950;--sell:#f85149;--warn:#d29922;--warnbg:#2b2111;--vetobg:#3a1517;--okbg:#12261a;--btn:#21262d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;
flex-wrap:wrap;gap:10px;align-items:center}
h1{font-size:17px;margin:0 12px 0 0}.pill{border:1px solid var(--line);border-radius:99px;padding:2px 10px;font-size:12px;color:var(--muted)}
.pill.live{color:var(--sell);border-color:var(--sell)}.spacer{flex:1}
main{max-width:1200px;margin:0 auto;padding:16px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;margin-bottom:16px;overflow:hidden}
section h2{font-size:15px;margin:0;padding:10px 14px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center}
.wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}
th{font-size:12px;color:var(--muted);font-weight:600}td.num{text-align:right}
button{font:inherit;border:1px solid var(--line);background:var(--btn);color:var(--fg);border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover{filter:brightness(1.05)}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.ok{background:var(--buy);border-color:var(--buy);color:#fff}button.danger{background:var(--sell);border-color:var(--sell);color:#fff}
button.big{font-size:15px;padding:8px 16px;font-weight:600}button:disabled{opacity:.5;cursor:default}
.side{font-weight:700}.BUY{color:var(--buy)}.SELL{color:var(--sell)}
.chip{display:inline-block;border-radius:5px;padding:1px 7px;font-size:12px;font-weight:600;cursor:pointer}
.PASS{background:var(--okbg);color:var(--buy)}.CAUTION,.REVIEW{background:var(--warnbg);color:var(--warn)}
.VETO{background:var(--vetobg);color:var(--sell)}.SKIPPED,.none{background:var(--btn);color:var(--muted)}
.note{white-space:normal;color:var(--muted);font-size:12.5px;max-width:560px}
.alert{background:var(--warnbg);color:var(--fg);border-left:4px solid var(--warn);padding:8px 12px;margin:8px 14px;border-radius:4px}
.kill{background:var(--vetobg);border-left-color:var(--sell);font-weight:600}
.empty{color:var(--muted);padding:14px}input{font:inherit;width:90px;padding:4px 6px;border:1px solid var(--line);border-radius:5px;
background:var(--bg);color:var(--fg)}.details{white-space:normal;background:var(--bg);font-size:12.5px}
.details li{margin:2px 0}.muted{color:var(--muted)}#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);
background:var(--fg);color:var(--bg);padding:10px 16px;border-radius:8px;max-width:90vw;display:none;z-index:9}
.kpis{display:flex;flex-wrap:wrap;gap:18px;padding:12px 14px}.kpi b{display:block;font-size:18px}
.sub{color:var(--muted);font-size:12px;white-space:nowrap}input{width:78px}td.inst,td.act{white-space:normal}td.act{min-width:220px}
@media (max-width:760px){header{gap:6px}h1{width:100%}table.resp tr.head{display:none}table.resp tr{display:block;border-bottom:1px solid var(--line);padding:6px 0}table.resp td{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;border:none;padding:3px 14px;white-space:normal;text-align:right}table.resp td::before{content:attr(data-l);color:var(--muted);font-size:12px;text-align:left}table.resp td.details,table.resp td.inst{display:block;text-align:left}table.resp td.details::before,table.resp td.inst::before{content:none}}
</style></head><body>
<header><h1>Trade Desk</h1><span class="pill" id="date">–</span><span class="pill" id="broker">–</span>
<span class="pill" id="ai">–</span><span class="spacer"></span>
<button class="primary big" id="refreshBtn" onclick="act('refresh')">Refresh plan</button>
<button class="danger big" id="killBtn" onclick="kill()">STOP ALL</button></header>
<main>
<div id="banner"></div>
<section><h2>Account</h2><div class="kpis" id="kpis"></div></section>
<section><h2>Next-session trade plan <span class="muted" style="font-weight:400;font-size:12px">orders execute at the next
open · approve each one</span></h2><div id="alerts"></div><div class="wrap" id="plan"></div></section>
<section><h2>Open positions</h2><div class="wrap" id="positions"></div></section>
<section><h2>Activity log</h2><div class="wrap" id="events"></div></section>
<section><h2>Danger zone</h2><div style="padding:12px 14px;display:flex;gap:10px;flex-wrap:wrap">
<button onclick="act('sync')">Sync with broker</button><button onclick="act('rebalance')">Force core rebalance</button>
<button class="danger" onclick="flatten()">Exit ALL positions…</button></div></section>
</main><div id="toast"></div>
<script>
const T=new URLSearchParams(location.search).get('t');let S=null,expanded={};
const WHY={rebalance_entry:'Core: new holding',rebalance:'Core: resize',rebalance_exit:'Core: dropped from top list',entry_ml:'Swing: model edge after costs',entry_rule:'Swing: trend rule',time_stop:'Swing: time stop',stop_breach:'Closed below stop',manual_exit:'Manual exit',flatten:'Exit all'};const why=r=>WHY[r]||r;
const inr=v=>v==null?'–':'₹'+Number(v).toLocaleString('en-IN',{maximumFractionDigits:2});
const pct=v=>v==null?'–':(v*100).toFixed(2)+'%';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function toast(m){const t=document.getElementById('toast');t.textContent=m;t.style.display='block';clearTimeout(t._h);
t._h=setTimeout(()=>t.style.display='none',6000)}
async function api(path,body){const r=await fetch('/api/'+path,{method:body?'POST':'GET',headers:{'X-Desk-Token':T,
'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const j=await r.json();
if(!r.ok||j.ok===false)throw new Error(j.error||r.statusText);return j}
async function act(a,b={}){try{const j=await api(a,b);toast(j.message||j.status||'Done');}catch(e){toast('⚠ '+e.message)}load()}
function kill(){if(S&&S.kill)return act('resume');if(confirm('STOP ALL: cancel pending buys and block new trades? Protective stops stay active.'))act('kill',{reason:'console'})}
function exitpos(i){const x=X_(i);if(confirm('Exit '+x.symbol+'?'))act('exit',{sleeve:x.sleeve,symbol:x.symbol})}
function flatten(){const c=prompt('This creates SELL orders for EVERY position. Type SELL ALL to confirm:');if(c)act('flatten',{confirm:c})}
function P_(id){return S.proposals.find(p=>p.id===id)}function X_(i){return S.positions[i]}
function approve(id){const p=P_(id);let body={id:p.id};if(p.status==='blocked_ai'){const r=prompt('The AI reviewer vetoed this trade. Reason for overriding?');
if(!r)return;body.override_reason=r}act('approve',body)}
function fill(id){const p=P_(id);const q=document.getElementById('q'+p.id).value,px=document.getElementById('p'+p.id).value;
if(!px)return toast('Enter your actual fill price');act('fill',{id:p.id,qty:q,price:px})}
function stopped(i){const x=X_(i);const px=prompt(`Price at which ${x.symbol} was sold by your stop/target order?`,x.stop||x.last);if(px)act('stopped',{sleeve:x.sleeve,symbol:x.symbol,price:px})}
function movestop(i){const x=X_(i);const s=prompt(`New stop-loss for ${x.symbol} (last ${x.last})`,x.stop||'');if(s===null)return;
const t=prompt('Target (blank = keep)',x.target||'');act('stop',{sleeve:x.sleeve,symbol:x.symbol,stop:s,target:t})}
function review(p){const r=p.review;if(!r)return '<span class="chip none">not reviewed</span>';
return `<span class="chip ${r.verdict}" onclick="expanded[${p.id}]=!expanded[${p.id}];render()">AI: ${r.verdict}</span>`}
function details(p){const r=p.review;if(!r||!expanded[p.id])return '';const f=(r.flags||[]).map(x=>`<li><b>${esc(x.severity)}</b> ${esc(x.category)}: ${esc(x.explanation)} <span class="muted">[${x.evidence_ids.join(', ')}]</span></li>`).join('');
const h=(r.headlines||[]).map((x,i)=>`<li><span class="muted">[${i+1}] ${esc(String(x.timestamp).slice(0,10))}</span> ${esc(x.title)}</li>`).join('');
return `<tr><td colspan="10" class="details"><b>${esc(r.summary)}</b>${r.injection_suspected?' <span class="chip VETO">injection attempt detected</span>':''}
${f?'<ul>'+f+'</ul>':''}${h?'<details><summary>Headlines reviewed</summary><ol style="list-style:none;padding-left:0">'+h+'</ol></details>':''}
<div class="muted">${esc(r.model||'')}${r.challenged?' · veto challenged by second reviewer':''}</div></td></tr>`}
function render(){if(!S)return;document.getElementById('date').textContent='Data: '+(S.data_date||'never refreshed');
const b=document.getElementById('broker');b.textContent=S.broker==='manual'?'Manual broker (you place orders)':(S.dry_run?'Zerodha · DRY RUN':'Zerodha · LIVE');
b.className='pill'+(S.broker!=='manual'&&!S.dry_run?' live':'');
document.getElementById('ai').textContent=S.reviewer.enabled?`AI reviewer: ${S.reviewer.model} · $${S.ai_cost_usd.last_day_usd} today`:'AI reviewer off';
const kb=document.getElementById('killBtn');kb.textContent=S.kill?'RESUME trading':'STOP ALL';kb.className=S.kill?'ok big':'danger big';
const rb=document.getElementById('refreshBtn');rb.disabled=S.busy;rb.textContent=S.busy?'Refreshing…':'Refresh plan';
const dr=S.drift||{};const drift=(dr.status==='below'||dr.status==='above')?`<div class="alert ${dr.status==='below'?'kill':''}">${esc(dr.message)}</div>`:'';
document.getElementById('banner').innerHTML=drift+(S.kill?'<div class="alert kill">KILL SWITCH ON - no new buys will be proposed or sent. Protective stops remain active.</div>':'')+(S.error?`<div class="alert kill">${esc(S.error)}</div>`:'');
const sl=Object.entries(S.sleeves).map(([k,v])=>`<div class="kpi"><span class="muted">${esc(k)}</span><b>${inr(v.equity)}</b><span class="muted">cash ${inr(v.cash)}${v.halted?' · HALTED':''}${v.loss_streak?` · loss streak ${v.loss_streak}`:''}</span></div>`).join('');
const tc=S.tca||{};const tcHtml=tc.n?`<div class="kpi"><span class="muted">Execution cost (${tc.n} fills)</span><b>${tc.median_bps.toFixed(1)} bps</b><span class="muted">model ${tc.model_bps.toFixed(1)} bps</span></div>`:'';
document.getElementById('kpis').innerHTML=`<div class="kpi"><span class="muted">Total equity</span><b>${inr(S.equity)}</b><span class="muted">start ${inr(S.capital)}</span></div>`+sl+tcHtml+(dr.message?`<div class="kpi" style="max-width:420px"><span class="muted">Live vs backtest</span><span class="note">${esc(dr.message)}</span></div>`:'');
document.getElementById('alerts').innerHTML=(S.alerts||[]).map(a=>`<div class="alert">${esc(a)}</div>`).join('');
const P=S.proposals;document.getElementById('plan').innerHTML=!P.length?'<div class="empty">No orders proposed. Press “Refresh plan” after the market closes.</div>':
`<table class="resp"><tr class="head"><th>Sleeve</th><th>Order</th><th class="num">Qty</th><th class="num">Limit</th><th class="num">Stop-loss</th><th class="num">Target</th><th class="num">Risk</th><th>Why</th><th>Check</th><th>Action</th></tr>`+
P.map(p=>{const st=p.stop_dist?p.ref_price-p.stop_dist:null,tg=p.target_dist?p.ref_price+p.target_dist:null;let a='';
if(p.status==='pending')a=`<button class="ok" onclick="approve(${p.id})">Approve</button> <button onclick="act('skip',{id:${p.id}})">Skip</button>`;
else if(p.status==='blocked_ai')a=`<button onclick="approve(${p.id})">Override…</button> <button onclick="act('skip',{id:${p.id}})">Skip</button>`;
else if(['awaiting_fill','submitted','dry_run'].includes(p.status))a=`<span>Qty <input id="q${p.id}" value="${p.qty}"> @ ₹<input id="p${p.id}" placeholder="fill price"></span> <button class="ok" onclick="fill(${p.id})">Mark filled</button> <button onclick="act('cancel',{id:${p.id}})">Cancel</button>`;
else a=`<span class="chip VETO">${esc(p.status)}</span><div class="note">${esc(p.message)}</div>`;
const inst=['awaiting_fill','submitted','dry_run'].includes(p.status)&&p.message?`<tr><td colspan="10" class="inst note">➜ ${esc(p.message)}</td></tr>`:'';
return `<tr><td data-l="Sleeve">${esc(p.sleeve)}</td><td data-l="Order"><span><span class="side ${p.side}">${p.side}</span> <b>${esc(p.symbol)}</b>${p.company&&p.company!==p.symbol?'<div class="sub">'+esc(p.company)+'</div>':''}</span></td>
<td class="num" data-l="Qty">${p.qty}</td><td class="num" data-l="Limit">${inr(p.limit_price)}</td><td class="num" data-l="Stop-loss">${inr(st)}</td><td class="num" data-l="Target">${inr(tg)}</td>
<td class="num" data-l="Risk">${p.risk_inr?'<span>'+inr(Math.round(p.risk_inr))+'<div class="sub">'+pct(p.risk_pct)+' of capital</div></span>':'–'}</td>
<td data-l="Why">${esc(why(p.reason))}</td><td data-l="Check">${p.side==='BUY'?review(p):'<span class="chip none">exit</span>'}</td><td class="act" data-l="Action"><span>${a}</span></td></tr>`+inst+details(p)}).join('')+'</table>';
const X=S.positions;document.getElementById('positions').innerHTML=!X.length?'<div class="empty">No open positions.</div>':
`<table class="resp"><tr class="head"><th>Sleeve</th><th>Stock</th><th class="num">Qty</th><th class="num">Avg cost</th><th class="num">Last</th><th class="num">P&amp;L</th><th class="num">Stop</th><th class="num">Target</th><th class="num">Risk to stop</th><th>Action</th></tr>`+
X.map((x,i)=>`<tr><td data-l="Sleeve">${esc(x.sleeve)}</td><td data-l="Stock"><span><b>${esc(x.symbol)}</b><div class="sub">since ${esc(x.entry_date)}</div></span></td><td class="num" data-l="Qty">${x.shares}</td>
<td class="num" data-l="Avg cost">${inr(x.avg_cost)}</td><td class="num" data-l="Last">${inr(x.last)}</td><td class="num ${x.pnl>=0?'BUY':'SELL'}" data-l="P&amp;L"><span>${inr(x.pnl)}<div class="sub">${pct(x.pnl_pct)}</div></span></td>
<td class="num" data-l="Stop">${inr(x.stop)}</td><td class="num" data-l="Target">${inr(x.target)}</td><td class="num" data-l="Risk to stop">${inr(x.risk_to_stop)}</td>
<td data-l="Action" style="white-space:normal"><button onclick="movestop(${i})">Move stop</button> <button onclick="stopped(${i})">Stopped out</button> <button class="danger" onclick="exitpos(${i})">Exit</button>
${x.protection?`<div class="note">${esc(x.protection)}</div>`:''}</td></tr>`).join('')+'</table>';
document.getElementById('events').innerHTML='<table><tr><th>Time</th><th>Action</th><th>Detail</th></tr>'+S.events.map(e=>`<tr><td>${esc(e.ts)}</td><td>${esc(e.action)}</td><td class="note">${esc(String(e.detail||'').slice(0,300))}</td></tr>`).join('')+'</table>'}
async function load(){try{S=await api('state');render()}catch(e){toast('⚠ '+e.message)}}
load();setInterval(load,8000);
</script></body></html>"""
