"""Minimal JSON API (stdlib only). Endpoints:
  GET  /            search UI
  GET  /health
  POST /search   {"query": "...", "modality": "sat", "place": "malibu", "date_from": "...", "k": 10}
  POST /similar  {"item_id": 123, "modality": "cell", "k": 10}
  GET  /item/<id>
  POST /agent    {"question": "..."}
"""
from __future__ import annotations
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from . import agent as A


UI = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRISM</title><style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--line:#e7e5e4;--acc:#0f766e}
@media(prefers-color-scheme:dark){:root{--bg:#0c0a09;--fg:#f5f5f4;--mut:#a8a29e;--card:#1c1917;--line:#292524;--acc:#2dd4bf}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:28px 16px 60px}h1{margin:0;font-size:22px;letter-spacing:.02em}
.sub{color:var(--mut);margin:2px 0 18px}form{display:grid;gap:8px;grid-template-columns:1fr auto auto}
input,select,button{font:inherit;padding:9px 11px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
button{cursor:pointer;background:var(--acc);color:#fff;border:0}button.alt{background:var(--card);color:var(--fg);border:1px solid var(--line)}
.row{display:flex;gap:8px;margin:8px 0 18px;flex-wrap:wrap}.row>*{flex:1;min-width:140px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:8px 0}
.tag{display:inline-block;font-size:12px;padding:1px 8px;border-radius:99px;background:var(--line);margin-right:6px}
.mut{color:var(--mut);font-size:13px}pre{white-space:pre-wrap;word-break:break-word;font:13px/1.5 ui-monospace,monospace;margin:0}
#meta{color:var(--mut);font-size:13px;margin:6px 0}@media(max-width:560px){form{grid-template-columns:1fr}}
</style></head><body><main>
<h1>PRISM</h1><p class="sub">One query across satellite imagery, DNA and single-cell data.</p>
<form id="f"><input id="q" placeholder="e.g. wildfire burn scars near Malibu in 2025" autofocus>
<button type="submit">Search</button><button type="button" class="alt" id="ask">Ask agent</button></form>
<div class="row"><select id="mod"><option value="">all modalities</option><option value="sat">satellite</option>
<option value="dna">DNA</option><option value="cell">single-cell</option></select>
<select id="place"><option value="">anywhere</option><option>malibu</option><option>los angeles</option>
<option>angeles national forest</option><option>mojave</option><option>imperial valley</option><option>southern california</option></select>
<input id="from" type="date" title="from"><input id="to" type="date" title="to"></div>
<div id="meta"></div><div id="out"></div></main><script>
const $=s=>document.querySelector(s),out=$('#out'),meta=$('#meta');
const post=async(p,b)=>{const r=await fetch(p,{method:'POST',body:JSON.stringify(b)});return r.json()};
const el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e};
function hit(h){const c=el('div','card'),top=el('div');top.append(el('span','tag',h.modality),el('b',null,'#'+h.id),
 el('span','mut','  score '+h.score));c.append(top);
 const f=['region','date','ndvi','nbr','gc','top_marker','genes_detected'].filter(k=>h[k]!=null).map(k=>k+' '+h[k]).join(' · ');
 c.append(el('div','mut',f),el('div','mut',h.uri));
 const b=el('button','alt','inspect');b.style.marginTop='6px';b.onclick=async()=>{b.disabled=true;
  const d=await (await fetch('/item/'+h.id)).json();c.append(el('pre',null,JSON.stringify(d,null,1)))};c.append(b);return c}
function body(){const b={query:$('#q').value,k:10};if($('#mod').value)b.modality=$('#mod').value;
 if($('#place').value)b.place=$('#place').value;if($('#from').value)b.date_from=$('#from').value;
 if($('#to').value)b.date_to=$('#to').value;return b}
$('#f').onsubmit=async e=>{e.preventDefault();out.replaceChildren();meta.textContent='searching…';const t=performance.now();
 const r=await post('/search',body());if(r.error){meta.textContent=r.error;return}
 meta.textContent=r.hits.length+' hits · '+Math.round(performance.now()-t)+' ms · scanned '+r.probe.scanned.toLocaleString()+' vectors';
 out.append(...r.hits.map(hit))};
$('#ask').onclick=async()=>{out.replaceChildren();meta.textContent='agent working…';const t=performance.now();
 const r=await post('/agent',{question:$('#q').value});if(r.error){meta.textContent=r.error;return}
 meta.textContent=(r.grounded?'grounded':'UNGROUNDED '+r.ungrounded_ids)+' · '+r.steps+' tool calls · '+Math.round(performance.now()-t)+' ms · trace '+r.trace;
 const c=el('div','card');c.append(el('pre',null,r.answer.replace(/\\*\\*/g,'')));out.append(c)};
</script></body></html>"""


def serve(eng, host="0.0.0.0", port=8080, trace_dir="traces"):
    lock = threading.Lock()  # harness keeps per-run state; serialize agent runs

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, default=str).encode()
            self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("content-length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            try:
                if self.path in ("/", "/index.html"):
                    body = UI.encode()
                    self.send_response(200); self.send_header("content-type", "text/html; charset=utf-8")
                    self.send_header("content-length", str(len(body))); self.end_headers()
                    return self.wfile.write(body)
                if self.path == "/health":
                    return self._send(200, {"ok": True, "items": len(eng)})
                if self.path.startswith("/item/"):
                    return self._send(200, eng.inspect(int(self.path.split("/")[-1])))
                self._send(404, {"error": "not found"})
            except Exception as e:
                self._send(400, {"error": str(e)})

        def do_POST(self):
            try:
                b = self._body()
                if self.path == "/search":
                    return self._send(200, eng.search(**b))
                if self.path == "/similar":
                    return self._send(200, eng.similar(**b))
                if self.path == "/agent":
                    with lock:
                        h = A.Harness(eng, trace_dir=trace_dir)
                        return self._send(200, A.default_planner().run(h, b["question"]))
                self._send(404, {"error": "not found"})
            except Exception as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer((host, port), H)
    print(f"PRISM serving {len(eng):,} items on http://{host}:{port}")
    srv.serve_forever()
