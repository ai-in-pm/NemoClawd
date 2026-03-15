
// ── Config ──────────────────────────────────────────────────────────────────
const GW_HOST   = location.hostname || '127.0.0.1';
const GW_PORT   = location.port || '18789';
const GW_WS     = `ws://${GW_HOST}:${GW_PORT}`;
const GW_TOKEN  = '00ab32e78002bfa24d14d15e5cab1c82a0a2d1af54b2b249';
const SESSION   = 'nemoclawd-main';

// ── DOM refs ─────────────────────────────────────────────────────────────────
const dot        = document.getElementById('dot');
const statusText = document.getElementById('status-text');
const sessionLbl = document.getElementById('session-label');
const logEl      = document.getElementById('log');
const chatIn     = document.getElementById('chat-in');
const btnSend    = document.getElementById('btn-send');
const actionBtns = document.querySelectorAll('.btn');
const btnStt      = document.getElementById('btn-stt');
const btnPtt      = document.getElementById('btn-ptt');
const btnDictation = document.getElementById('btn-dictation');
const btnV2V      = document.getElementById('btn-v2v');
const btnTts      = document.getElementById('btn-tts');
const btnClear    = document.getElementById('btn-clear');
const voiceStatus = document.getElementById('voice-status');

// ── Log helper ───────────────────────────────────────────────────────────────
function log(msg, cls = 'sys') {
  const p = document.createElement('p');
  p.className = 'log-line ' + cls;
  const now = new Date().toLocaleTimeString();
  p.textContent = `[${now}] ${msg}`;
  if (logEl.children.length === 1 && logEl.firstChild.textContent.includes('Initialising')) {
    logEl.innerHTML = '';
  }
  logEl.appendChild(p);
  logEl.scrollTop = logEl.scrollHeight;
}

// ── State ────────────────────────────────────────────────────────────────────
let ws, connected = false;
function uuid() { return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => { const r = Math.random() * 16 | 0; return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16); }); }
const pending = {};
const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition || null;
const voice = {
  sttSupported: Boolean(SpeechRecognitionCtor),
  ttsSupported: 'speechSynthesis' in window,
  ttsEnabled: 'speechSynthesis' in window,
  dictationEnabled: false,
  v2vEnabled: false,
  recognition: null,
  listeningMode: null,
  manualStop: false,
  finalText: '',
  finalSent: false,
  pttPressing: false,
};
const nudge = {
  idleMs: 120000,
  timer: null,
  awaitingUserReply: false,
};

function clearNudgeTimer() {
  if (!nudge.timer) return;
  clearTimeout(nudge.timer);
  nudge.timer = null;
}

function disarmNudge() {
  nudge.awaitingUserReply = false;
  clearNudgeTimer();
}

function armNudge() {
  nudge.awaitingUserReply = true;
  clearNudgeTimer();
  nudge.timer = setTimeout(() => {
    if (!nudge.awaitingUserReply || !connected) return;
    if (voice.listeningMode || voice.pttPressing || voice.dictationEnabled) {
      armNudge();
      return;
    }
    log("Nudge: still there? I can continue when you are ready.", "sys");
    nudge.awaitingUserReply = false;
    nudge.timer = null;
  }, nudge.idleMs);
}

function markUserActivity() {
  disarmNudge();
}

function setVoiceStatus(text) {
  voiceStatus.textContent = `Voice: ${text}`;
}

function updateVoiceButtons() {
  const allowListen = connected && voice.sttSupported;
  btnStt.disabled = !allowListen || voice.listeningMode !== null;
  btnPtt.disabled = !allowListen;
  btnDictation.disabled = !allowListen;
  btnV2V.disabled = !allowListen;
  btnTts.disabled = !voice.ttsSupported;

  btnTts.classList.toggle('active', voice.ttsEnabled);
  btnDictation.classList.toggle('active', voice.dictationEnabled);
  btnV2V.classList.toggle('active', voice.v2vEnabled);
  btnPtt.classList.toggle('active', voice.pttPressing);
}

function stopSpeaking() {
  if (!voice.ttsSupported) return;
  window.speechSynthesis.cancel();
}

function speakText(text) {
  if (!voice.ttsSupported || !voice.ttsEnabled || !text.trim()) return Promise.resolve();
  stopSpeaking();
  return new Promise((resolve) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.onend = () => resolve();
    utterance.onerror = () => resolve();
    window.speechSynthesis.speak(utterance);
  });
}

function addToInput(text) {
  const trimmed = text.trim();
  if (!trimmed) return;
  const current = chatIn.value.trim();
  chatIn.value = current ? `${current} ${trimmed}` : trimmed;
}

function ensureRecognition() {
  if (!voice.sttSupported || voice.recognition) return;
  const recognition = new SpeechRecognitionCtor();
  recognition.lang = 'en-US';
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  recognition.onresult = (event) => {
    let finalChunk = '';
    let interimChunk = '';

    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const transcript = event.results[i][0]?.transcript ?? '';
      if (event.results[i].isFinal) finalChunk += transcript;
      else interimChunk += transcript;
    }

    if (finalChunk) {
      voice.finalText = `${voice.finalText} ${finalChunk}`.trim();
    }

    if (voice.listeningMode === 'dictation') {
      if (finalChunk) addToInput(finalChunk);
      setVoiceStatus(interimChunk ? `dictating… ${interimChunk.trim()}` : 'dictation listening');
      return;
    }

    const composed = `${voice.finalText} ${interimChunk}`.trim();
    if (composed) chatIn.value = composed;

    if (voice.finalText && !voice.finalSent && (voice.listeningMode === 'stt' || voice.listeningMode === 'ptt')) {
      voice.finalSent = true;
      sendMessage(voice.finalText);
      chatIn.value = '';
      setVoiceStatus('sent speech message');
    }
  };

  recognition.onerror = (event) => {
    setVoiceStatus(`speech error: ${event.error || 'unknown'}`);
  };

  recognition.onend = () => {
    const endedMode = voice.listeningMode;
    voice.listeningMode = null;
    updateVoiceButtons();

    if (endedMode === 'dictation' && voice.dictationEnabled && connected && !voice.manualStop) {
      startListening('dictation', true);
      return;
    }

    if (!voice.manualStop && voice.finalText && !voice.finalSent && (endedMode === 'stt' || endedMode === 'ptt')) {
      sendMessage(voice.finalText);
      chatIn.value = '';
    }

    voice.manualStop = false;
    voice.finalText = '';
    voice.finalSent = false;
    if (!voice.dictationEnabled) setVoiceStatus('idle');
  };

  voice.recognition = recognition;
}

function startListening(mode, autoRestart = false) {
  if (!voice.sttSupported || !connected) return;
  ensureRecognition();
  if (!voice.recognition || voice.listeningMode) return;

  voice.listeningMode = mode;
  voice.manualStop = false;
  voice.finalText = '';
  voice.finalSent = false;
  voice.recognition.continuous = mode === 'dictation';

  setVoiceStatus(mode === 'dictation' ? 'dictation listening' : 'listening');
  updateVoiceButtons();

  try {
    voice.recognition.start();
  } catch {
    voice.listeningMode = null;
    updateVoiceButtons();
    if (!autoRestart) setVoiceStatus('microphone busy; retry');
  }
}

function stopListening(manual = true) {
  if (!voice.recognition || !voice.listeningMode) return;
  voice.manualStop = manual;
  try {
    voice.recognition.stop();
  } catch {
    voice.listeningMode = null;
    updateVoiceButtons();
  }
}

function setConnected(ok, label = '') {
  connected = ok;
  dot.className = 'dot ' + (ok ? 'ok' : 'err');
  statusText.textContent = ok ? `Connected · gateway at ${GW_HOST}:${GW_PORT}` : (label || 'Disconnected');
  sessionLbl.textContent = ok ? `session: ${SESSION}` : '';
  actionBtns.forEach(b => b.disabled = !ok);
  chatIn.disabled = !ok;
  btnSend.disabled = !ok;
  if (!ok) {
    voice.dictationEnabled = false;
    stopListening(true);
    disarmNudge();
  }
  updateVoiceButtons();
}

// ── Gateway protocol ─────────────────────────────────────────────────────────
function request(method, params) {
  return new Promise((resolve, reject) => {
    const id = uuid();
    pending[id] = { resolve, reject };
    ws.send(JSON.stringify({ type: 'req', id, method, params }));
    setTimeout(() => { delete pending[id]; reject(new Error(`timeout: ${method}`)); }, 25000);
  });
}

async function sendMessage(text) {
  if (!connected) { log('Not connected', 'err'); return; }
  markUserActivity();
  log(`You: ${text}`, 'user');
  try {
    await request('chat.send', {
      sessionKey: SESSION,
      message: text,
      thinking: 'low',
      idempotencyKey: `canvas-${Date.now()}`,
      timeoutMs: 60000
    });
  } catch(e) {
    log(`Error: ${e.message}`, 'err');
  }
}

// ── Button actions ────────────────────────────────────────────────────────────
document.getElementById('btn-hello').onclick  = () => sendMessage('Hello NemoClawd! Introduce yourself briefly.');
document.getElementById('btn-time').onclick   = () => sendMessage('What is the current date and time?');
document.getElementById('btn-status').onclick = () => sendMessage('Give me a quick project status heartbeat check.');
document.getElementById('btn-tasks').onclick  = () => sendMessage('Show me any open tasks or action items from my workspace.');
btnSend.onclick = () => { const v = chatIn.value.trim(); if (v) { sendMessage(v); chatIn.value = ''; } };
chatIn.addEventListener('keydown', e => { if (e.key === 'Enter') btnSend.click(); });
chatIn.addEventListener('input', () => {
  if (chatIn.value.trim().length > 0) markUserActivity();
});

btnStt.onclick = () => startListening('stt');
btnDictation.onclick = () => {
  markUserActivity();
  voice.dictationEnabled = !voice.dictationEnabled;
  if (voice.dictationEnabled) {
    startListening('dictation');
  } else {
    stopListening(true);
    setVoiceStatus('dictation off');
  }
  updateVoiceButtons();
};

btnV2V.onclick = () => {
  voice.v2vEnabled = !voice.v2vEnabled;
  if (voice.v2vEnabled) voice.ttsEnabled = true;
  setVoiceStatus(voice.v2vEnabled ? 'voice-to-voice enabled' : 'voice-to-voice off');
  updateVoiceButtons();
};

btnTts.onclick = () => {
  voice.ttsEnabled = !voice.ttsEnabled;
  if (!voice.ttsEnabled) stopSpeaking();
  setVoiceStatus(voice.ttsEnabled ? 'tts enabled' : 'tts off');
  updateVoiceButtons();
};

btnPtt.onmousedown = () => {
  markUserActivity();
  voice.pttPressing = true;
  updateVoiceButtons();
  startListening('ptt');
};
btnPtt.onmouseup = () => {
  voice.pttPressing = false;
  updateVoiceButtons();
  stopListening(true);
};
btnPtt.onmouseleave = () => {
  if (!voice.pttPressing) return;
  voice.pttPressing = false;
  updateVoiceButtons();
  stopListening(true);
};
btnPtt.ontouchstart = (event) => {
  event.preventDefault();
  markUserActivity();
  voice.pttPressing = true;
  updateVoiceButtons();
  startListening('ptt');
};
btnPtt.ontouchend = (event) => {
  event.preventDefault();
  voice.pttPressing = false;
  updateVoiceButtons();
  stopListening(true);
};

btnClear.onclick = () => {
  disarmNudge();
  stopListening(true);
  stopSpeaking();
  voice.dictationEnabled = false;
  chatIn.value = '';
  logEl.innerHTML = '';
  log('Cleared conversation log', 'sys');
  setVoiceStatus('idle');
  updateVoiceButtons();
};

function extractAssistantText(message) {
  if (typeof message === 'string') return message;
  if (!message || typeof message !== 'object') return '';
  if (typeof message.text === 'string') return message.text;

  const content = Array.isArray(message.content) ? message.content : [];
  const chunks = [];
  for (const part of content) {
    if (typeof part === 'string') {
      chunks.push(part);
      continue;
    }
    if (!part || typeof part !== 'object') continue;
    if (typeof part.text === 'string') chunks.push(part.text);
    else if (part.type === 'text' && typeof part.value === 'string') chunks.push(part.value);
  }
  return chunks.join('').trim();
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connect() {
  setConnected(false, 'Connecting…');
  dot.className = 'dot';
  statusText.textContent = 'Connecting to gateway…';
  ws = new WebSocket(GW_WS);

  ws.onopen = async () => {
    log('WebSocket open — authenticating…', 'sys');
    try {
      await request('connect', {
        minProtocol: 3, maxProtocol: 3,
        client: { id: 'clawdbot-control-ui', version: '1.0', platform: 'web', mode: 'ui' },
        auth: { token: GW_TOKEN }
      });
      setConnected(true);
      log('Bridge: ready · gateway-connected', 'sys');
    } catch(e) {
      setConnected(false, 'Auth failed');
      log(`Auth error: ${e.message}`, 'err');
    }
  };

  ws.onmessage = ({ data }) => {
    let frame;
    try { frame = JSON.parse(data); } catch { return; }
    if (frame.type === 'res') {
      const p = pending[frame.id];
      if (p) { delete pending[frame.id]; frame.error ? p.reject(new Error(frame.error.message ?? 'error')) : p.resolve(frame.payload); }
    } else if (frame.type === 'event' && frame.event === 'chat') {
      const evt = frame.payload ?? frame.data;
      if (!evt || typeof evt !== 'object') return;
      if (evt.sessionKey && evt.sessionKey !== SESSION) return;

      if (evt.state === 'delta') {
        // Ignore partial stream updates to avoid logging truncated duplicate lines.
      } else if (evt.state === 'final') {
        const text = extractAssistantText(evt.message);
        if (text) {
          log(`NemoClawd: ${text}`, 'agent');
          armNudge();
          speakText(text).then(() => {
            if (voice.v2vEnabled && connected && !voice.dictationEnabled && !voice.listeningMode) {
              startListening('stt');
            }
          });
        }
      } else if (evt.state === 'aborted') {
        log('Run aborted', 'sys');
      } else if (evt.state === 'error') {
        log(`Run error: ${evt.errorMessage ?? 'unknown'}`, 'err');
      }
    } else if (frame.type === 'event' && frame.event === 'agent.error') {
      log(`Agent error: ${(frame.payload ?? frame.data)?.message ?? 'unknown'}`, 'err');
    }
  };

  ws.onerror = () => { setConnected(false, 'Connection error'); log('WebSocket error', 'err'); };
  ws.onclose = () => {
    setConnected(false, 'Disconnected — retrying in 4 s…');
    log('Connection closed — retrying…', 'sys');
    setTimeout(connect, 4000);
  };
}

connect();
if (!voice.sttSupported) setVoiceStatus('stt unsupported by this browser');
if (!voice.ttsSupported) setVoiceStatus('tts unsupported by this browser');
updateVoiceButtons();
