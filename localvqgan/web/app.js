const $ = (id) => document.getElementById(id);
const losses = [];
let uploadedInit = null, uploadedPrompt = null, sysinfo = null;

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return r.json();
}

async function loadCheckpoints() {
  const cks = await api("/api/checkpoints");
  const sel = $("checkpoint");
  const prev = sel.value;
  sel.innerHTML = "";
  for (const c of cks) {
    const o = document.createElement("option");
    o.value = c.name;
    o.textContent = c.downloaded ? c.name
      : c.mirror_offline ? `${c.name} (mirrors offline)`
      : `${c.name} (download ${c.size_mb} MB)`;
    o.dataset.downloaded = c.downloaded;
    o.dataset.offline = c.mirror_offline;
    sel.appendChild(o);
  }
  sel.value = prev || "imagenet_16384";
  updateDownloadUI();
}

function updateDownloadUI() {
  const opt = $("checkpoint").selectedOptions[0];
  if (!opt) return;
  const downloaded = opt.dataset.downloaded === "true";
  const offline = opt.dataset.offline === "true";
  $("ckpt-download").classList.toggle("hidden", downloaded || offline);
  $("ckpt-offline").classList.toggle("hidden", downloaded || !offline);
  $("generate").disabled = (!downloaded && offline) || !sizeOk();
}

async function loadSystem() {
  sysinfo = await api("/api/system");
  const eng = $("engine");
  for (const e of sysinfo.engines) {
    const o = document.createElement("option");
    o.value = e; o.textContent = e;
    eng.appendChild(o);
  }
  $("sysinfo").textContent =
    `device: ${sysinfo.device} · ram: ${sysinfo.total_ram_gb} GB · max side: ${sysinfo.max_recommended_side}px`;
  checkSize();
}

function sizeOk() {
  if (!sysinfo) return true;
  const m = sysinfo.max_recommended_side;
  return +$("width").value <= m && +$("height").value <= m;
}

function checkSize() {
  const ok = sizeOk();
  $("size-warning").classList.toggle("hidden", ok);
  if (!ok) $("size-warning").textContent =
    `Above ${sysinfo.max_recommended_side}px this machine will likely run out of memory.`;
  updateDownloadUI();
}

function fileToDataURL(input) {
  const f = input.files[0];
  if (!f) return Promise.resolve(null);
  return new Promise((res) => {
    const rd = new FileReader();
    rd.onload = () => res(rd.result);
    rd.readAsDataURL(f);
  });
}

async function uploadIfAny() {
  uploadedInit = await fileToDataURL($("init_image"));
  uploadedPrompt = await fileToDataURL($("image_prompt"));
}

function settingsFromForm() {
  return {
    prompts: $("prompts").value,
    width: +$("width").value, height: +$("height").value,
    iterations: +$("iterations").value, cutouts: +$("cutouts").value,
    cut_pow: +$("cut_pow").value,
    step_size: +$("step_size").value, seed: +$("seed").value,
    init_weight: +$("init_weight").value,
    checkpoint: $("checkpoint").value, clip_model: $("clip_model").value,
    engine: $("engine").value,
    display_freq: +$("display_freq").value,
    precision: $("precision").value,
    fast_mode: $("fast_mode").checked,
  };
}

function setRunning(running) {
  // while a job runs, Generate stays available and queues the next job
  $("generate").textContent = running ? "Queue next" : "Generate";
  $("stop").classList.toggle("hidden", !running);
}

async function startJob(body) {
  losses.length = 0;
  body.queue = true; // server runs jobs one at a time; extra submits wait in line
  await api("/api/jobs", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  setRunning(true);
  $("error").textContent = "";
}

$("generate").onclick = async () => {
  try {
    await uploadIfAny();
    const settings = settingsFromForm();
    const batch = Math.min(16, Math.max(1, +($("batch")?.value || 1)));
    for (let i = 0; i < batch; i++) {
      // batches vary only the seed: explicit seeds step by one, random stays random
      const s = batch === 1 ? settings
        : { ...settings, seed: settings.seed >= 0 ? settings.seed + i : -1 };
      const body = { type: "still", settings: s };
      if (uploadedInit) body.init_image_data = uploadedInit;
      if (uploadedPrompt) body.image_prompt_data = uploadedPrompt;
      await startJob(body);
    }
  } catch (e) { $("error").textContent = e.message; }
};

$("stop").onclick = () => api("/api/jobs/cancel", { method: "POST" });

$("download-btn").onclick = async () => {
  try {
    await api(`/api/checkpoints/${$("checkpoint").value}/download`, { method: "POST" });
    $("download-progress").classList.remove("hidden");
  } catch (e) { $("error").textContent = e.message; }
};

$("checkpoint").onchange = updateDownloadUI;
$("width").oninput = checkSize;
$("height").oninput = checkSize;

function drawSpark() {
  const c = $("loss-spark"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (losses.length < 2) return;
  const min = Math.min(...losses), max = Math.max(...losses), span = max - min || 1;
  ctx.strokeStyle = "#d8a24a";
  ctx.beginPath();
  losses.forEach((v, i) => {
    const x = (i / (losses.length - 1)) * c.width;
    const y = c.height - ((v - min) / span) * (c.height - 4) - 2;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

function onMessage(msg) {
  if (msg.queued !== undefined)
    $("queued").textContent = msg.queued > 0 ? `· ${msg.queued} queued` : "";
  if (msg.type === "queue") return;
  if (msg.type === "download") {
    const p = $("download-progress");
    p.classList.remove("hidden");
    p.value = msg.total ? (100 * msg.done / msg.total) : 0;
    if (msg.total && msg.done >= msg.total) {
      p.classList.add("hidden");
      loadCheckpoints();
    }
    return;
  }
  if (msg.image_b64) {
    $("preview").src = "data:image/jpeg;base64," + msg.image_b64;
    $("idle-hint").style.display = "none";
  }
  if (msg.loss !== undefined) { losses.push(msg.loss); drawSpark(); }
  if (msg.iteration !== undefined) {
    // a finished/errored job keeps its last iteration fields in the status
    // snapshot; don't let a reload present them as still "generating"
    const phase = msg.state === "done" ? "done" :
                  msg.state === "error" ? "failed" : (msg.phase || "");
    $("progress-text").textContent = `${phase} ${msg.iteration}/${msg.total}` + (msg.engine ? ` · ${msg.engine}` : "");
  }
  if (msg.its_per_sec !== undefined) $("speed").textContent = `${msg.its_per_sec} it/s`;
  if (msg.engine) $("speed").title = `engine: ${msg.engine} (${msg.engine_reason || ""})`;
  if (msg.engine) $("progress-text").dataset.engine = msg.engine;
  if (msg.error) $("error").textContent = msg.error;
  if (msg.state === "running") setRunning(true);
  if (msg.state === "done" || msg.state === "error" || msg.state === "idle") {
    setRunning(false);
    if (msg.state === "done") loadGallery();
  }
}

function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connectWS, 1500);
}

async function loadGallery() {
  const runs = await api("/api/gallery");
  const g = $("gallery");
  g.innerHTML = "";
  for (const r of runs) {
    if (!r.final) continue;
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <img src="/api/gallery/${r.run_id}/final.png" loading="lazy">
      <div class="actions">
        <a href="/api/gallery/${r.run_id}/final.png" download>PNG</a>
        <a href="/api/gallery/${r.run_id}/timelapse.mp4">MP4</a>
        <a href="/api/gallery/${r.run_id}/settings.json" target="_blank">JSON</a>
        <button data-run="${r.run_id}" class="reuse">Reuse</button>
        <button data-run="${r.run_id}" class="upscale"
          title="Re-render at double size, refining this image (the classic init-image upscale: keeps the composition, adds detail)">Upscale 2&times;</button>
        <button data-run="${r.run_id}" class="delete" title="Delete this run">&times;</button>
      </div>`;
    card.querySelector(".reuse").onclick = async (ev) => {
      const s = await api(`/api/gallery/${ev.target.dataset.run}/settings.json`);
      for (const k of ["prompts", "width", "height", "iterations", "cutouts",
                       "cut_pow", "step_size", "seed", "init_weight", "clip_model",
                       "engine", "display_freq", "precision"])
        if (s[k] !== undefined && $(k)) $(k).value = s[k];
      // a seed=-1 run recorded the seed it actually drew; reuse that for
      // an exact re-run instead of rolling a new one
      if (s.seed === -1 && s.seed_used != null) $("seed").value = s.seed_used;
      if (s.checkpoint) $("checkpoint").value = s.checkpoint;
      if (s.fast_mode !== undefined) $("fast_mode").checked = s.fast_mode;
      checkSize();
      window.scrollTo({ top: 0, behavior: "smooth" });
    };
    card.querySelector(".upscale").onclick = async (ev) => {
      try {
        const runId = ev.target.dataset.run;
        const s = await api(`/api/gallery/${runId}/settings.json`);
        const w = (s.width || 256) * 2, h = (s.height || 256) * 2;
        if (sysinfo && Math.max(w, h) > sysinfo.max_recommended_side) {
          $("error").textContent =
            `Upscale target ${w}×${h} is above this machine's recommended ${sysinfo.max_recommended_side}px.`;
          return;
        }
        const blob = await (await fetch(`/api/gallery/${runId}/final.png`)).blob();
        const dataUrl = await new Promise((res) => {
          const rd = new FileReader(); rd.onload = () => res(rd.result); rd.readAsDataURL(blob);
        });
        const settings = { ...s, width: w, height: h, init_image: null,
          // the fine-stage share of the budget: enough to add 2x detail
          // without re-deciding the composition the draft already settled
          iterations: Math.max(60, Math.round((s.iterations || 300) * 0.4)),
          fast_mode: false };
        delete settings.engine_used;
        await startJob({ type: "still", settings, init_image_data: dataUrl });
      } catch (e) { $("error").textContent = e.message; }
    };
    card.querySelector(".delete").onclick = async (ev) => {
      const runId = ev.target.dataset.run;
      if (!confirm(`Delete ${runId}? This removes its files.`)) return;
      try {
        await api(`/api/gallery/${runId}`, { method: "DELETE" });
        card.remove();
      } catch (e) { $("error").textContent = e.message; }
    };
    g.appendChild(card);
  }
}

// --- animation mode ---
function kfRow(kf = { prompts: "", frames: 30, zoom: 1.02, pan_x: 0, pan_y: 0,
                     iterations_per_frame: 8 }) {
  const div = document.createElement("div");
  div.className = "kf";
  div.innerHTML = `
    <textarea class="kf-prompts" rows="2" placeholder="prompts for this keyframe">${kf.prompts}</textarea>
    <div class="row">
      <label>Frames <input class="kf-frames" type="number" value="${kf.frames}" min="1"></label>
      <label>Iters/frame <input class="kf-ipf" type="number" value="${kf.iterations_per_frame}" min="1"></label>
    </div>
    <div class="row">
      <label>Zoom <input class="kf-zoom" type="number" step="0.01" value="${kf.zoom}"></label>
      <label>Pan x/y <span><input class="kf-px" type="number" value="${kf.pan_x}" style="width:45%">
        <input class="kf-py" type="number" value="${kf.pan_y}" style="width:45%"></span></label>
    </div>
    <button class="kf-del">remove</button><hr>`;
  div.querySelector(".kf-del").onclick = () => div.remove();
  return div;
}

$("add-kf").onclick = () => $("keyframes").appendChild(kfRow());

$("animate").onclick = async () => {
  const keyframes = [...document.querySelectorAll("#keyframes .kf")].map((d) => ({
    prompts: d.querySelector(".kf-prompts").value,
    frames: +d.querySelector(".kf-frames").value,
    iterations_per_frame: +d.querySelector(".kf-ipf").value,
    zoom: +d.querySelector(".kf-zoom").value,
    pan_x: +d.querySelector(".kf-px").value,
    pan_y: +d.querySelector(".kf-py").value,
  }));
  if (!keyframes.length) { $("error").textContent = "Add at least one keyframe"; return; }
  try {
    await uploadIfAny();
    const body = { type: "animation", settings: settingsFromForm(), keyframes };
    if (uploadedInit) body.init_image_data = uploadedInit;
    await startJob(body);
  } catch (e) { $("error").textContent = e.message; }
};
$("keyframes").appendChild(kfRow());

loadCheckpoints(); loadSystem(); loadGallery(); connectWS();
