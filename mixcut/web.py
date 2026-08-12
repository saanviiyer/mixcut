"""FastAPI web app for mixcut.

Flow: upload -> /analyze returns section JSON -> user adjusts span + crossfade
in the timeline -> /render produces wav+mp3 -> before/after preview + download.

Run:  uvicorn mixcut.web:app --reload
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from .analysis import analyze
from .render import render_cut, write_outputs

MAX_UPLOAD_BYTES = 40 * 1024 * 1024  # 40 MB cap
ALLOWED_EXT = {".wav", ".aiff", ".aif", ".flac", ".mp3", ".m4a", ".ogg"}

WORK_DIR = os.environ.get(
    "MIXCUT_WORK", os.path.join(tempfile.gettempdir(), "mixcut_work")
)
os.makedirs(WORK_DIR, exist_ok=True)

app = FastAPI(title="mixcut")

# job_id -> {"input": path, "analysis": dict}
JOBS: dict[str, dict] = {}


def _job_dir(job_id: str) -> str:
    d = os.path.join(WORK_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return d


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type {ext!r}. "
                                 f"Allowed: {sorted(ALLOWED_EXT)}")
    job_id = uuid.uuid4().hex[:12]
    jd = _job_dir(job_id)
    in_path = os.path.join(jd, "input" + ext)

    size = 0
    with open(in_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                shutil.rmtree(jd, ignore_errors=True)
                raise HTTPException(413, "Upload too large (max 40 MB).")
            f.write(chunk)

    try:
        a = analyze(in_path)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")

    JOBS[job_id] = {"input": in_path, "analysis": a.to_dict()}
    payload = {"job_id": job_id, **a.to_dict()}
    return JSONResponse(payload)


@app.post("/render")
async def render_endpoint(
    job_id: str = Form(...),
    remove_start: float = Form(...),
    remove_end: float = Form(...),
    crossfade_bars: float = Form(1.0),
    crossfade_seconds: float = Form(0.0),
):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job_id (re-upload).")
    a = job["analysis"]
    jd = _job_dir(job_id)
    out_base = os.path.join(jd, "mixcut_out")
    try:
        result = render_cut(
            job["input"],
            remove_start=remove_start,
            remove_end=remove_end,
            crossfade_seconds=crossfade_seconds,
            crossfade_bars=crossfade_bars,
            tempo=a["tempo"],
            beats_per_bar=a.get("beats_per_bar", 4),
        )
    except Exception as e:
        raise HTTPException(400, f"Render failed: {e}")

    outs = write_outputs(result, out_base)
    return JSONResponse({
        "job_id": job_id,
        "orig_duration": round(result.orig_duration, 3),
        "out_duration": round(result.out_duration, 3),
        "removed_duration": round(result.removed_duration, 3),
        "crossfade_seconds": round(result.crossfade_seconds, 4),
        "join_discontinuity": round(result.join_discontinuity, 6),
        "wav_url": f"/file/{job_id}/out.wav",
        "mp3_url": f"/file/{job_id}/out.mp3" if outs["mp3"] else None,
    })


@app.get("/file/{job_id}/{name}")
def get_file(job_id: str, name: str):
    jd = _job_dir(job_id)
    if name == "input":
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job.")
        return FileResponse(job["input"])
    mapping = {"out.wav": "mixcut_out.wav", "out.mp3": "mixcut_out.mp3"}
    fname = mapping.get(name)
    if not fname:
        raise HTTPException(404, "Not found.")
    path = os.path.join(jd, fname)
    if not os.path.exists(path):
        raise HTTPException(404, "File not rendered yet.")
    return FileResponse(path, filename=fname)


@app.get("/original/{job_id}")
def get_original(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    return FileResponse(job["input"])


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mixcut - shorten a song for your mix</title>
<style>
  :root { --bg:#12131a; --panel:#1c1e29; --fg:#e9ecf5; --muted:#8b90a6;
          --chorus:#e0533d; --verse:#3d7de0; --other:#5a6072; --remove:#f5c542; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--fg); }
  header { padding:20px 24px; border-bottom:1px solid #2a2d3c; }
  h1 { margin:0; font-size:20px; }
  .sub { color:var(--muted); font-size:13px; margin-top:4px; }
  main { max-width:940px; margin:0 auto; padding:24px; }
  .card { background:var(--panel); border:1px solid #2a2d3c; border-radius:10px;
          padding:18px; margin-bottom:18px; }
  button { background:#2f3550; color:var(--fg); border:1px solid #414868;
           padding:8px 14px; border-radius:7px; cursor:pointer; font-size:14px; }
  button:hover { background:#3a4166; }
  button.primary { background:var(--chorus); border-color:var(--chorus); }
  button:disabled { opacity:.5; cursor:default; }
  label { font-size:13px; color:var(--muted); }
  input[type=number]{ width:90px; background:#0f1017; color:var(--fg);
     border:1px solid #414868; border-radius:6px; padding:5px 7px; }
  #timeline { position:relative; height:70px; background:#0f1017;
     border-radius:8px; overflow:hidden; margin-top:10px; user-select:none; }
  .seg { position:absolute; top:0; height:100%; display:flex; align-items:center;
     justify-content:center; font-size:11px; color:#fff; overflow:hidden;
     border-right:1px solid rgba(0,0,0,.4); white-space:nowrap; }
  .seg.chorus { background:var(--chorus); }
  .seg.verse  { background:var(--verse); }
  .seg.other  { background:var(--other); }
  #removeBand { position:absolute; top:0; height:100%;
     background:rgba(245,197,66,.28); border-left:2px solid var(--remove);
     border-right:2px solid var(--remove); }
  .handle { position:absolute; top:0; width:10px; height:100%; cursor:ew-resize;
     background:var(--remove); opacity:.9; }
  .legend { font-size:12px; color:var(--muted); margin-top:10px; }
  .legend span{ display:inline-block; width:11px; height:11px; border-radius:2px;
     vertical-align:middle; margin:0 4px 0 12px; }
  .row { display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin-top:12px; }
  .stat { font-size:13px; color:var(--muted); }
  .stat b { color:var(--fg); }
  audio { width:100%; margin-top:8px; }
  .note { font-size:12px; color:var(--muted); line-height:1.5; }
  .err { color:#ff8080; font-size:13px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media(max-width:640px){ .grid2{grid-template-columns:1fr;} }
</style>
</head>
<body>
<header>
  <h1>mixcut</h1>
  <div class="sub">Auto-shorten a track by cutting the 2nd verse + chorus, spliced beat-aligned with an equal-power crossfade. Detection is approximate &mdash; review the span before exporting.</div>
</header>
<main>
  <div class="card">
    <label>1. Upload a song (wav / aiff / flac / mp3, &le;40 MB)</label><br><br>
    <input type="file" id="file" accept=".wav,.aiff,.aif,.flac,.mp3,.m4a,.ogg">
    <button id="analyzeBtn" class="primary">Analyze</button>
    <span id="analyzeStatus" class="stat"></span>
    <div id="analyzeErr" class="err"></div>
  </div>

  <div class="card" id="structCard" style="display:none">
    <label>2. Detected structure &mdash; drag the yellow handles or type to adjust the removal span</label>
    <div id="timeline"></div>
    <div class="legend">
      <span style="background:var(--chorus)"></span>chorus
      <span style="background:var(--verse)"></span>verse
      <span style="background:var(--other)"></span>other
      <span style="background:var(--remove)"></span>to remove
    </div>
    <div class="row">
      <div><label>remove start (s)</label><br><input type="number" id="remStart" step="0.1"></div>
      <div><label>remove end (s)</label><br><input type="number" id="remEnd" step="0.1"></div>
      <div><label>crossfade (bars)</label><br><input type="number" id="xfBars" step="0.25" value="1"></div>
      <div style="align-self:flex-end"><button id="renderBtn" class="primary">Render preview</button></div>
    </div>
    <div class="stat" id="reason" style="margin-top:10px"></div>
    <div id="renderErr" class="err"></div>
  </div>

  <div class="card" id="previewCard" style="display:none">
    <label>3. Before / After preview</label>
    <div class="grid2">
      <div>
        <div class="stat">Original &mdash; <b id="origDur"></b></div>
        <audio id="origAudio" controls preload="none"></audio>
      </div>
      <div>
        <div class="stat">mixcut &mdash; <b id="outDur"></b></div>
        <audio id="outAudio" controls preload="none"></audio>
      </div>
    </div>
    <div class="row">
      <div class="stat">removed <b id="removedDur"></b> &middot; crossfade <b id="xfMs"></b> &middot; join step <b id="joinDisc"></b></div>
    </div>
    <div class="row">
      <a id="dlWav"><button>Download WAV</button></a>
      <a id="dlMp3"><button id="mp3btn">Download MP3</button></a>
    </div>
    <div class="note" style="margin-top:10px">
      A low "join step" means the splice is click-free. If the transition sounds off,
      nudge the removal boundaries to land on downbeats and re-render.
    </div>
  </div>
</main>

<script>
let A = null;      // analysis JSON
let jobId = null;

const $ = id => document.getElementById(id);

$("analyzeBtn").onclick = async () => {
  const f = $("file").files[0];
  $("analyzeErr").textContent = "";
  if (!f) { $("analyzeErr").textContent = "Choose a file first."; return; }
  $("analyzeStatus").textContent = "analyzing…";
  $("analyzeBtn").disabled = true;
  const fd = new FormData();
  fd.append("file", f);
  try {
    const r = await fetch("/analyze", { method:"POST", body:fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    A = await r.json();
    jobId = A.job_id;
    $("analyzeStatus").textContent = `tempo ${A.tempo.toFixed(0)} BPM, ${A.duration.toFixed(1)}s, ${A.sections.length} sections`;
    buildTimeline();
    $("remStart").value = A.remove_start.toFixed(2);
    $("remEnd").value = A.remove_end.toFixed(2);
    $("reason").textContent = A.remove_reason;
    $("structCard").style.display = "block";
    drawRemoveBand();
  } catch (e) {
    $("analyzeErr").textContent = "Analyze failed: " + e.message;
  } finally {
    $("analyzeBtn").disabled = false;
    $("analyzeStatus").textContent = $("analyzeStatus").textContent.replace("analyzing…","");
  }
};

function segClass(s){
  if (s.is_chorus) return "chorus";
  if (s.label === "verse") return "verse";
  return "other";
}

function buildTimeline(){
  const tl = $("timeline");
  tl.innerHTML = "";
  const D = A.duration;
  for (const s of A.sections){
    const el = document.createElement("div");
    el.className = "seg " + segClass(s);
    el.style.left = (100*s.start/D) + "%";
    el.style.width = (100*(s.end-s.start)/D) + "%";
    el.title = `${s.label} ${s.start.toFixed(1)}-${s.end.toFixed(1)}s`;
    el.textContent = s.label;
    tl.appendChild(el);
  }
  const band = document.createElement("div");
  band.id = "removeBand"; tl.appendChild(band);
  const h1 = document.createElement("div"); h1.className="handle"; h1.id="hStart";
  const h2 = document.createElement("div"); h2.className="handle"; h2.id="hEnd";
  tl.appendChild(h1); tl.appendChild(h2);
  makeDraggable(h1, "start");
  makeDraggable(h2, "end");
}

function drawRemoveBand(){
  const D = A.duration;
  let s = parseFloat($("remStart").value), e = parseFloat($("remEnd").value);
  s = Math.max(0, Math.min(s, D)); e = Math.max(0, Math.min(e, D));
  const band = $("removeBand");
  band.style.left = (100*s/D) + "%";
  band.style.width = (100*Math.max(0,e-s)/D) + "%";
  $("hStart").style.left = "calc(" + (100*s/D) + "% - 5px)";
  $("hEnd").style.left = "calc(" + (100*e/D) + "% - 5px)";
}

function makeDraggable(handle, which){
  handle.onmousedown = (ev) => {
    ev.preventDefault();
    const tl = $("timeline");
    const rect = tl.getBoundingClientRect();
    const move = (e) => {
      let frac = (e.clientX - rect.left) / rect.width;
      frac = Math.max(0, Math.min(1, frac));
      let t = frac * A.duration;
      // snap to nearest beat
      if (A.beats && A.beats.length){
        let best = t, bd = 1e9;
        for (const b of A.beats){ const d = Math.abs(b-t); if (d<bd){bd=d;best=b;} }
        t = best;
      }
      if (which === "start") $("remStart").value = t.toFixed(2);
      else $("remEnd").value = t.toFixed(2);
      drawRemoveBand();
    };
    const up = () => { document.removeEventListener("mousemove", move);
                       document.removeEventListener("mouseup", up); };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  };
}

$("remStart").oninput = drawRemoveBand;
$("remEnd").oninput = drawRemoveBand;

$("renderBtn").onclick = async () => {
  $("renderErr").textContent = "";
  $("renderBtn").disabled = true;
  $("renderBtn").textContent = "rendering…";
  const fd = new FormData();
  fd.append("job_id", jobId);
  fd.append("remove_start", $("remStart").value);
  fd.append("remove_end", $("remEnd").value);
  fd.append("crossfade_bars", $("xfBars").value || "1");
  try {
    const r = await fetch("/render", { method:"POST", body:fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const d = await r.json();
    const bust = "?t=" + Date.now();
    $("origAudio").src = "/original/" + jobId + bust;
    $("outAudio").src = d.wav_url + bust;
    $("origDur").textContent = d.orig_duration.toFixed(1) + "s";
    $("outDur").textContent = d.out_duration.toFixed(1) + "s";
    $("removedDur").textContent = d.removed_duration.toFixed(1) + "s";
    $("xfMs").textContent = (d.crossfade_seconds*1000).toFixed(0) + "ms";
    $("joinDisc").textContent = d.join_discontinuity.toFixed(4);
    $("dlWav").href = d.wav_url;
    if (d.mp3_url){ $("dlMp3").href = d.mp3_url; $("mp3btn").disabled=false; }
    else { $("mp3btn").disabled = true; $("dlMp3").removeAttribute("href"); }
    $("previewCard").style.display = "block";
  } catch (e) {
    $("renderErr").textContent = "Render failed: " + e.message;
  } finally {
    $("renderBtn").disabled = false;
    $("renderBtn").textContent = "Render preview";
  }
};
</script>
</body>
</html>
"""
