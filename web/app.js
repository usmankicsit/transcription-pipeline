const fileInput = document.querySelector("#file");
const drop = document.querySelector("#drop");
const fileLabel = document.querySelector("#file-label");
const languageInput = document.querySelector("#language");
const sampleButton = document.querySelector("#sample");
const goButton = document.querySelector("#go");
const statusNode = document.querySelector("#status");
const engineNode = document.querySelector("#engine");
const resultNode = document.querySelector("#result");
const metaNode = document.querySelector("#meta");
const fullText = document.querySelector("#full-text");
const player = document.querySelector("#player");
const timeline = document.querySelector("#timeline");
const segmentsNode = document.querySelector("#segments");

const LANGUAGE_NAMES = {
  en: "english",
  es: "spanish",
  fr: "french",
  de: "german",
  hi: "hindi",
  ur: "urdu",
  ar: "arabic",
  zh: "chinese",
  pt: "portuguese",
  ja: "japanese",
};

let chosenFile = null;
let transcript = null;
let objectUrl = null;
let browserEngine = null;
let useServer = false;

status("Checking which engine is available.");
detectEngine();

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) selectFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((name) => {
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.add("hot");
  });
});

["dragleave", "drop"].forEach((name) => {
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.remove("hot");
  });
});

drop.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) selectFile(file);
});

sampleButton.addEventListener("click", loadSample);
goButton.addEventListener("click", transcribe);
player.addEventListener("timeupdate", highlightCurrent);

document.querySelectorAll("[data-format]").forEach((button) => {
  button.addEventListener("click", () => download(button.dataset.format));
});

function selectFile(file) {
  chosenFile = file;
  fileLabel.textContent = file.name;
  status(`${file.name} is ready.`);
  attachPlayer(file);
}

async function loadSample() {
  status("Loading the spoken sample.");
  const response = await fetch("sample.mp3");
  if (!response.ok) {
    status("The sample file is missing.");
    return;
  }
  const blob = await response.blob();
  selectFile(new File([blob], "sample.mp3", { type: "audio/mpeg" }));
  await transcribe();
}

async function detectEngine() {
  try {
    const response = await fetch("/health");
    const body = await response.json();
    useServer = response.ok && body.status === "ok";
  } catch {
    useServer = false;
  }
  engineNode.textContent = useServer
    ? "Connected to the Python service. ffmpeg decodes the file, then faster-whisper writes the timestamps."
    : "Public demo. Whisper runs in this browser so the page can be opened without installing anything. The Python service in the repository is the production path for long files.";
  status(useServer ? "Ready." : "Ready. The first transcription downloads the small Whisper model.");
}

async function transcribe() {
  if (!chosenFile) {
    status("Choose an audio file first.");
    return;
  }
  if (chosenFile.size > 25 * 1024 * 1024 && !useServer) {
    status("This demo keeps files under 25 MB. Use the Python service for a long recording.");
    return;
  }
  setBusy(true);
  try {
    transcript = useServer ? await transcribeOnServer(chosenFile) : await transcribeInBrowser(chosenFile);
    renderTranscript(transcript);
    status("Done.");
  } catch (error) {
    status(error.message || "Transcription failed.");
  } finally {
    setBusy(false);
  }
}

async function transcribeOnServer(file) {
  status("Transcribing on the service.");
  const body = new FormData();
  body.append("file", file);
  const language = languageInput.value.trim();
  if (language) body.append("language", language);
  body.append("response_format", "json");
  const response = await fetch("/v1/transcriptions", { method: "POST", body });
  const payload = await response.json();
  if (response.status === 202) return pollJob(payload.id);
  if (!response.ok) throw new Error(payload.detail || "The service rejected this file.");
  return payload;
}

async function pollJob(id) {
  status("This file is long, so the service is working on it in the background.");
  for (let attempt = 0; attempt < 600; attempt += 1) {
    const response = await fetch(`/v1/transcriptions/${id}`);
    const job = await response.json();
    if (job.status === "succeeded") return job.transcript;
    if (job.status === "failed") throw new Error(job.error || "The job failed.");
    await wait(1000);
  }
  throw new Error("The job is still running. Poll the status URL from the API.");
}

async function transcribeInBrowser(file) {
  status("Decoding the audio to 16 kHz mono.");
  const samples = await decodeToMono16k(file);
  const engine = await loadBrowserEngine();
  status("Transcribing. A first run is slower while the model loads.");
  const language = languageInput.value.trim().toLowerCase();
  const options = {
    return_timestamps: true,
    chunk_length_s: 30,
    stride_length_s: 5,
    task: "transcribe",
  };
  if (LANGUAGE_NAMES[language]) options.language = LANGUAGE_NAMES[language];
  const output = await engine(samples, options);
  const segments = (output.chunks || [])
    .map((chunk) => {
      const [start, end] = chunk.timestamp || [0, null];
      return {
        start: Number(start || 0),
        end: end == null ? Number(start || 0) : Number(end),
        text: String(chunk.text || "").trim(),
        words: [],
      };
    })
    .filter((segment) => segment.text);
  return {
    language: language || null,
    duration: Math.round((samples.length / 16000) * 1000) / 1000,
    text: String(output.text || "").trim(),
    segments,
    source_format: extensionOf(file.name),
    source_codec: "browser-decode",
  };
}

async function loadBrowserEngine() {
  if (browserEngine) return browserEngine;
  status("Loading Whisper in the browser.");
  const { pipeline, env } = await import("https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2");
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  browserEngine = await pipeline("automatic-speech-recognition", "Xenova/whisper-tiny", {
    progress_callback: (update) => {
      if (update.status === "progress" && update.file) {
        const percent = update.progress == null ? "" : ` ${Math.round(update.progress)}%`;
        status(`Downloading ${update.file}${percent}`);
      }
    },
  });
  return browserEngine;
}

async function decodeToMono16k(file) {
  const context = new AudioContext();
  const decoded = await context.decodeAudioData(await file.arrayBuffer());
  const length = Math.ceil(decoded.duration * 16000);
  const offline = new OfflineAudioContext(1, length, 16000);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  await context.close();
  return rendered.getChannelData(0);
}

function renderTranscript(body) {
  transcript = body;
  resultNode.hidden = false;
  const codec = body.source_codec ? ` · ${body.source_codec}` : "";
  metaNode.textContent = `${formatClock(0).slice(3)} to ${formatClock(body.duration || 0)} · ${body.language || "detected"}${codec}`;
  fullText.textContent = body.text || "No speech detected.";
  timeline.replaceChildren();
  segmentsNode.replaceChildren();
  const duration = body.duration || lastEnd(body.segments);
  body.segments.forEach((segment, index) => {
    const width = duration > 0 ? Math.max(2, ((segment.end - segment.start) / duration) * 100) : 2;
    const mark = document.createElement("button");
    mark.type = "button";
    mark.style.flex = `${width} 1 0`;
    mark.title = segment.text;
    mark.addEventListener("click", () => seek(segment.start, index));
    timeline.append(mark);

    const item = document.createElement("li");
    item.innerHTML = `<div class="time">${formatClock(segment.start)} – ${formatClock(segment.end)}</div>`;
    const copy = document.createElement("div");
    const phrase = document.createElement("p");
    phrase.className = "phrase";
    phrase.textContent = segment.text;
    copy.append(phrase);
    if (segment.words && segment.words.length) {
      const words = document.createElement("div");
      words.className = "words";
      segment.words.forEach((word) => {
        const chip = document.createElement("span");
        chip.textContent = `${word.text} ${formatClock(word.start)}`;
        words.append(chip);
      });
      copy.append(words);
    }
    item.append(copy);
    item.addEventListener("click", () => seek(segment.start, index));
    segmentsNode.append(item);
  });
}

function highlightCurrent() {
  if (!transcript) return;
  const items = segmentsNode.querySelectorAll("li");
  const marks = timeline.querySelectorAll("button");
  transcript.segments.forEach((segment, index) => {
    const on = player.currentTime >= segment.start && player.currentTime < segment.end;
    items[index].classList.toggle("on", on);
    marks[index].classList.toggle("on", on);
  });
}

function seek(seconds, index) {
  player.currentTime = seconds;
  player.play();
  highlightCurrent();
  segmentsNode.querySelectorAll("li")[index].scrollIntoView({ block: "nearest" });
}

function attachPlayer(file) {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  player.src = objectUrl;
}

function download(format) {
  if (!transcript) return;
  const rendered = renderExport(transcript, format);
  const blob = new Blob([rendered], { type: "text/plain" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `transcript.${format === "txt" ? "txt" : format}`;
  link.click();
}

function renderExport(body, format) {
  if (format === "json") return JSON.stringify(body, null, 2) + "\n";
  if (format === "txt") return body.segments.map((segment) => segment.text).join("\n") + "\n";
  const decimal = format === "vtt" ? "." : ",";
  const cues = body.segments
    .map((segment, index) => `${index + 1}\n${stamp(segment.start, decimal)} --> ${stamp(segment.end, decimal)}\n${segment.text}\n`)
    .join("\n");
  return format === "vtt" ? `WEBVTT\n\n${cues}` : cues;
}

function stamp(seconds, decimal) {
  const total = Math.max(0, Math.round(seconds * 1000));
  const hours = Math.floor(total / 3600000);
  const minutes = Math.floor((total % 3600000) / 60000);
  const secs = Math.floor((total % 60000) / 1000);
  const millis = total % 1000;
  const pad = (value, size = 2) => String(value).padStart(size, "0");
  return `${pad(hours)}:${pad(minutes)}:${pad(secs)}${decimal}${pad(millis, 3)}`;
}

function formatClock(seconds) {
  return stamp(seconds, ".").replace(/^00:/, "");
}

function lastEnd(segments) {
  return segments.reduce((end, segment) => Math.max(end, segment.end), 0);
}

function extensionOf(name) {
  const match = String(name).toLowerCase().match(/\.([a-z0-9]+)$/);
  return match ? match[1] : "audio";
}

function status(message) {
  statusNode.textContent = message;
}

function setBusy(busy) {
  goButton.disabled = busy;
  sampleButton.disabled = busy;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
