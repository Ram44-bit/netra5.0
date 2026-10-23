(() => {
  const $ = (id) => document.getElementById(id);
  const video = $("video");
  const uploadedImage = $("uploadedImage");
  const overlay = $("overlay");
  const ctx = overlay.getContext("2d");
  const emptyState = $("emptyState");
  const previewBadge = $("previewBadge");
  const modelStatus = $("modelStatus");
  const resultArea = $("resultArea");
  const objectCount = $("objectCount");
  const cameraBtn = $("cameraBtn");
  const uploadBtn = $("uploadBtn");
  const fileInput = $("fileInput");
  const stopBtn = $("stopBtn");
  const detectBtn = $("detectBtn");
  const speakBtn = $("speakBtn");
  const detailedScanBtn = $("detailedScanBtn");
  const micBtn = $("micBtn");
  const voiceStatus = $("voiceStatus");
  const questionInput = $("questionInput");
  const answerText = $("answerText");
  const askBtn = $("askBtn");

  let model = null;
  let stream = null;
  let latestPredictions = [];
  let detectionTimer = null;
  let recognition = null;
  let latestSpeechText = "";

  function setModelStatus(message, type = "") {
    modelStatus.className = "model-status " + type;
    modelStatus.textContent = message;
  }

  function showResultEmpty(title, message) {
    resultArea.innerHTML = `<div class="result-empty"><div class="scan-icon">⌕</div><strong>${title}</strong><p>${message}</p></div>`;
  }

  async function loadModel() {
    if (!window.cocoSsd || !window.tf) {
      setModelStatus("AI libraries unavailable", "error");
      return;
    }
    try {
      setModelStatus("Preparing fast AI model…");
      // Prefer WebGL for GPU acceleration, but keep the browser's fallback if unavailable.
      try { await tf.setBackend("webgl"); } catch (_) { /* fallback backend */ }
      await tf.ready();
      model = await cocoSsd.load({ base: "lite_mobilenet_v2" });
      // Warm up once so the first user-triggered detection is less likely to feel frozen.
      const warmup = document.createElement("canvas");
      warmup.width = 320; warmup.height = 320;
      try { await model.detect(warmup); } catch (_) { /* warmup is optional */ }
      setModelStatus("AI model ready", "ready");
      detectBtn.disabled = !(video.classList.contains("active") || uploadedImage.classList.contains("active"));
      detailedScanBtn.disabled = detectBtn.disabled;
      previewBadge.textContent = "AI model ready";
      if (video.classList.contains("active")) startLiveDetection();
    } catch (err) {
      console.error(err);
      setModelStatus("Model load failed", "error");
      showResultEmpty("AI model unavailable", "Check your internet connection, refresh the page, and try again.");
    }
  }

  function clearOverlay() {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }

  function resizeOverlayToPreview() {
    const rect = $("preview").getBoundingClientRect();
    overlay.width = Math.max(1, Math.round(rect.width * devicePixelRatio));
    overlay.height = Math.max(1, Math.round(rect.height * devicePixelRatio));
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      previewBadge.textContent = "Camera not supported";
      voiceStatus.textContent = "Camera API is unavailable. Open the HTTPS Render URL in a modern browser.";
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
      video.srcObject = stream;
      await video.play();
      video.classList.add("active");
      uploadedImage.classList.remove("active");
      emptyState.hidden = true;
      stopBtn.disabled = false;
      cameraBtn.querySelector("span").textContent = "Camera on";
      detectBtn.disabled = !model;
      detailedScanBtn.disabled = false;
      previewBadge.textContent = "Camera is active";
      resizeOverlayToPreview();
      if (model) startLiveDetection();
    } catch (err) {
      console.error(err);
      previewBadge.textContent = "Camera permission needed";
      voiceStatus.textContent = "Camera could not start. Allow camera permission in your browser and try again.";
    }
  }

  function stopCamera() {
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    video.classList.remove("active");
    stopBtn.disabled = true;
    cameraBtn.querySelector("span").textContent = "Start camera";
    if (detectionTimer) clearTimeout(detectionTimer);
    detectionTimer = null;
    clearOverlay();
    if (!uploadedImage.classList.contains("active")) {
      emptyState.hidden = false;
      detectBtn.disabled = true;
      detailedScanBtn.disabled = true;
      previewBadge.textContent = "Waiting for input";
    }
  }

  let liveDetectionBusy = false;
  async function runLiveDetectionFrame() {
    if (!video.classList.contains("active") || !stream) { detectionTimer = null; return; }
    const nextDelay = 250;
    if (!model || video.readyState < 2 || liveDetectionBusy) {
      detectionTimer = setTimeout(runLiveDetectionFrame, nextDelay);
      return;
    }
    liveDetectionBusy = true;
    try {
      // Resize frames before inference: substantially less work on mobile devices.
      const sourceW = video.videoWidth, sourceH = video.videoHeight;
      if (!sourceW || !sourceH) return;
      const scale = Math.min(1, 320 / Math.max(sourceW, sourceH));
      const small = document.createElement("canvas");
      small.width = Math.max(1, Math.round(sourceW * scale));
      small.height = Math.max(1, Math.round(sourceH * scale));
      small.getContext("2d", { alpha: false }).drawImage(video, 0, 0, small.width, small.height);
      const raw = await model.detect(small);
      const predictions = raw.map(p => ({ ...p, bbox: [p.bbox[0] / scale, p.bbox[1] / scale, p.bbox[2] / scale, p.bbox[3] / scale] }));
      latestPredictions = predictions;
      drawPredictions(predictions, sourceW, sourceH);
      renderPredictions(predictions);
      previewBadge.textContent = predictions.length ? `${predictions.length} object type(s) spotted` : "No objects detected";
    } catch (e) { console.warn("Live detection issue", e); }
    finally {
      liveDetectionBusy = false;
      if (video.classList.contains("active") && stream) detectionTimer = setTimeout(runLiveDetectionFrame, 250);
    }
  }
  function startLiveDetection() {
    if (detectionTimer) clearTimeout(detectionTimer);
    detectionTimer = setTimeout(runLiveDetectionFrame, 0);
  }

  function drawPredictions(predictions, sourceW, sourceH) {
    resizeOverlayToPreview();
    clearOverlay();
    const rect = $("preview").getBoundingClientRect();
    const isVideo = video.classList.contains("active");
    const source = isVideo ? video : uploadedImage;
    if (!sourceW || !sourceH || !source) return;
    const scale = Math.min(rect.width / sourceW, rect.height / sourceH);
    const drawW = sourceW * scale, drawH = sourceH * scale;
    const offsetX = (rect.width - drawW) / 2, offsetY = (rect.height - drawH) / 2;
    ctx.lineWidth = 2;
    ctx.font = "bold 12px system-ui";
    predictions.forEach(p => {
      const [x, y, w, h] = p.bbox;
      const dx = offsetX + x * scale, dy = offsetY + y * scale, dw = w * scale, dh = h * scale;
      ctx.strokeStyle = "#65e4e0";
      ctx.fillStyle = "#65e4e0";
      ctx.strokeRect(dx, dy, dw, dh);
      const label = `${p.class} ${Math.round(p.score * 100)}%`;
      const labelW = ctx.measureText(label).width + 12;
      ctx.fillRect(dx, Math.max(0, dy - 23), labelW, 22);
      ctx.fillStyle = "#07111f";
      ctx.fillText(label, dx + 6, Math.max(15, dy - 8));
    });
  }

  function renderPredictions(predictions) {
    latestPredictions = predictions;
    objectCount.textContent = `${predictions.length} object${predictions.length === 1 ? "" : "s"}`;
    if (!predictions.length) {
      showResultEmpty("Nothing detected yet", "Try moving the camera or using a clearer photo.");
      speakBtn.disabled = false;
      return;
    }
    resultArea.innerHTML = "";
    predictions.slice().sort((a,b) => b.score - a.score).forEach(p => {
      const pct = Math.round(p.score * 100);
      const item = document.createElement("div");
      item.className = "object-item";
      item.innerHTML = `<div style="flex:1;min-width:0"><div class="object-name"></div><div class="confidence-track"><div class="confidence-fill" style="width:${pct}%"></div></div></div><span class="object-confidence">${pct}% confidence</span>`;
      item.querySelector(".object-name").textContent = p.class;
      resultArea.appendChild(item);
    });
    speakBtn.disabled = false;
  }

  async function detectCurrentImage() {
    if (!model) {
      showResultEmpty("AI model is still loading", "Wait a moment and try again.");
      return;
    }
    const source = uploadedImage.classList.contains("active") ? uploadedImage : video;
    if (source === video && (!stream || video.readyState < 2)) return;
    detectBtn.disabled = true;
    detectBtn.textContent = "◌ Detecting…";
    try {
      const w = source.videoWidth || source.naturalWidth;
      const h = source.videoHeight || source.naturalHeight;
      const scale = Math.min(1, 640 / Math.max(w, h));
      const small = document.createElement("canvas");
      small.width = Math.max(1, Math.round(w * scale));
      small.height = Math.max(1, Math.round(h * scale));
      small.getContext("2d", { alpha: false }).drawImage(source, 0, 0, small.width, small.height);
      const raw = await model.detect(small);
      const predictions = raw.map(p => ({ ...p, bbox: [p.bbox[0] / scale, p.bbox[1] / scale, p.bbox[2] / scale, p.bbox[3] / scale] }));
      latestPredictions = predictions;
      drawPredictions(predictions, w, h);
      renderPredictions(predictions);
      previewBadge.textContent = predictions.length ? "Detection complete" : "No objects detected";
      if (predictions.length) speakResults(false);
    } catch (err) {
      console.error(err);
      showResultEmpty("Could not analyse this image", "Try another photo or refresh the page.");
    } finally {
      detectBtn.disabled = false;
      detectBtn.textContent = "✧ Detect objects";
    }
  }

  function speakText(text) {
    if (!("speechSynthesis" in window)) {
      voiceStatus.textContent = "Text-to-speech is not supported in this browser.";
      return;
    }
    window.speechSynthesis.cancel();
    // Let browsers finish loading their installed voices before choosing Hindi.
    latestSpeechText = String(text).replace(/\s+/g, " ").trim();
    const utterance = new SpeechSynthesisUtterance(latestSpeechText);
    const voices = window.speechSynthesis.getVoices();
    const hindiVoice = voices.find(v => /^hi(-|_)/i.test(v.lang)) || voices.find(v => /hindi/i.test(v.name));
    const indianEnglishVoice = voices.find(v => /^en-IN$/i.test(v.lang)) || voices.find(v => /^en(-|_)/i.test(v.lang));
    const hasDevanagari = /[\u0900-\u097F]/.test(utterance.text);
    if (hasDevanagari && hindiVoice) {
      utterance.voice = hindiVoice;
      utterance.lang = hindiVoice.lang || "hi-IN";
    } else if (indianEnglishVoice) {
      utterance.voice = indianEnglishVoice;
      utterance.lang = indianEnglishVoice.lang || "en-IN";
    } else {
      utterance.lang = hasDevanagari ? "hi-IN" : "en-IN";
    }
    utterance.rate = 0.84;
    utterance.pitch = 1;
    utterance.volume = 1;
    utterance.onstart = () => { voiceStatus.textContent = "Reading aloud in a slower, clearer voice…"; };
    utterance.onerror = () => { voiceStatus.textContent = "Audio could not play. Check your browser's installed Hindi voices and volume."; };
    window.speechSynthesis.speak(utterance);
  }

  function speakResults(updateStatus = true) {
    if (!latestPredictions.length) {
      speakText("No objects detected yet. Start the camera or upload an image.");
      return;
    }
    const names = latestPredictions.slice(0, 8).map(p => p.class);
    const labelsHi = { person: "व्यक्ति", book: "किताब या कॉपी", pen: "पेन", pencil: "पेंसिल", cup: "कप", bottle: "बोतल", cell_phone: "मोबाइल फोन", laptop: "लैपटॉप", keyboard: "कीबोर्ड", mouse: "माउस", backpack: "बैग", scissors: "कैंची", remote: "रिमोट", clock: "घड़ी", chair: "कुर्सी", tv: "टीवी" };
    const uniqueNames = [...new Set(names.map(name => labelsHi[name] || name.replaceAll("_", " ")))];
    const text = `NETRA को ये चीज़ें दिखाई दे रही हैं: ${uniqueNames.join(", ")}. यह शुरुआती पहचान है; छोटी चीज़ें छूट सकती हैं।`;
    if (updateStatus) voiceStatus.textContent = "Reading detection results aloud.";
    speakText(text);
  }

  async function detailedVisionScan() {
    const source = uploadedImage.classList.contains("active") ? uploadedImage : video;
    const isVideo = source === video;
    const w = isVideo ? video.videoWidth : source.naturalWidth;
    const h = isVideo ? video.videoHeight : source.naturalHeight;
    if (!w || !h || (isVideo && (!stream || video.readyState < 2))) {
      voiceStatus.textContent = "Pehle camera start karo ya ek clear photo upload karo.";
      return;
    }
    detailedScanBtn.disabled = true;
    const oldLabel = detailedScanBtn.textContent;
    detailedScanBtn.textContent = "◌ Analysing image…";
    voiceStatus.textContent = "Detailed vision AI image ko dhyan se check kar raha hai. Ismein kuch seconds lag sakte hain.";
    try {
      const canvas = document.createElement("canvas");
      const scale = Math.min(1, 1280 / Math.max(w, h));
      canvas.width = Math.max(1, Math.round(w * scale));
      canvas.height = Math.max(1, Math.round(h * scale));
      canvas.getContext("2d", { alpha: false }).drawImage(source, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.88));
      if (!blob) throw new Error("Could not prepare image");
      const form = new FormData();
      form.append("image", blob, "netra-capture.jpg");
      form.append("question", "Hindi mein image ka detailed visual scan karo. Chhote objects—pendrive/USB drive, pen, pencil, charger/cable, notebook/copy—ko specifically check karo. Agar koi person ka face clearly visible ho toh sirf apparent visible expression ko cautious estimate ke roop mein batao. Sabse pehle sabse clear object aur uski image mein position batao. Jo cheez confirm na ho uske baare mein uncertainty clearly bolo.");
      form.append("language", "Hindi");
      const response = await fetch(`${API_BASE}/api/describe`, { method: "POST", body: form });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `Vision service error (${response.status})`);
      const description = String(data.description || "Image ka description nahi mil saka.").trim();
      answerText.textContent = description;
      resultArea.innerHTML = "";
      const summary = document.createElement("div");
      summary.className = "result-empty";
      const strong = document.createElement("strong");
      strong.textContent = "Detailed AI scan complete";
      const paragraph = document.createElement("p");
      paragraph.textContent = description;
      summary.append(strong, paragraph);
      resultArea.appendChild(summary);
      objectCount.textContent = "Detailed scan";
      speakBtn.disabled = false;
      voiceStatus.textContent = "Detailed scan ready. Hindi audio sunne ke liye Read results aloud dabao.";
      speakText(description);
      previewBadge.textContent = "Detailed AI scan complete";
    } catch (err) {
      console.warn("Detailed vision scan failed", err);
      voiceStatus.textContent = `Detailed scan failed: ${err.message}. Backend par GEMINI_API_KEY aur FRONTEND_ORIGIN check karo.`;
      answerText.textContent = "Small-object scan ke liye backend vision API zaroori hai. Render backend environment mein GEMINI_API_KEY add karke redeploy karo.";
    } finally {
      detailedScanBtn.disabled = !(video.classList.contains("active") || uploadedImage.classList.contains("active"));
      detailedScanBtn.textContent = oldLabel;
    }
  }

  function setupSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      voiceStatus.textContent = "Voice input is not supported here. You can type a question below; read-aloud may still work.";
      micBtn.disabled = true;
      return;
    }
    recognition = new SpeechRecognition();
    recognition.lang = "hi-IN";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => {
      micBtn.classList.add("listening");
      voiceStatus.textContent = "Listening… say something now.";
    };
    recognition.onend = () => micBtn.classList.remove("listening");
    recognition.onerror = (event) => {
      micBtn.classList.remove("listening");
      voiceStatus.textContent = `Voice input issue (${event.error}). You can type instead.`;
    };
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      questionInput.value = transcript;
      answerQuestion(transcript);
    };
  }

  const API_BASE = (window.NETRA_API_BASE || "https://netra-ai-1-i8dy.onrender.com").replace(/\/$/, "");
  const chatHistory = [];
  async function answerQuestion(question) {
    const q = question.trim();
    if (!q) return;
    askBtn.disabled = true;
    micBtn.disabled = true;
    answerText.textContent = "NETRA AI is thinking…";
    voiceStatus.textContent = "Connecting to NETRA AI…";
    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q, history: chatHistory.slice(-8), detected_objects: [...new Set(latestPredictions.slice(0, 12).map(p => p.class))] })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.error || `AI service error (${response.status})`);
      const answer = String(data.answer || data.response || "I couldn't create an answer. Please try again.").trim();
      answerText.textContent = answer;
      chatHistory.push({ role: "user", content: q }, { role: "assistant", content: answer });
      if (chatHistory.length > 12) chatHistory.splice(0, chatHistory.length - 12);
      voiceStatus.textContent = "NETRA AI answer ready.";
      speakText(answer);
    } catch (err) {
      console.warn("NETRA chat request failed", err);
      answerText.textContent = `AI chat is not connected yet: ${err.message}. Check that the Python backend is deployed with OPENAI_API_KEY and that NETRA_API_BASE points to it.`;
      voiceStatus.textContent = "AI service connection failed. You can retry after checking the backend.";
    } finally {
      askBtn.disabled = false;
      micBtn.disabled = !recognition;
    }
  }

  cameraBtn.addEventListener("click", startCamera);
  stopBtn.addEventListener("click", stopCamera);
  uploadBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      voiceStatus.textContent = "Please select an image file.";
      return;
    }
    stopCamera();
    const url = URL.createObjectURL(file);
    uploadedImage.onload = () => {
      uploadedImage.classList.add("active");
      emptyState.hidden = true;
      detectBtn.disabled = !model;
      detailedScanBtn.disabled = false;
      previewBadge.textContent = "Photo ready for detection";
      clearOverlay();
      resizeOverlayToPreview();
      if (model) detectCurrentImage();
    };
    uploadedImage.src = url;
  });
  detectBtn.addEventListener("click", detectCurrentImage);
  detailedScanBtn.addEventListener("click", detailedVisionScan);
  speakBtn.addEventListener("click", () => latestSpeechText ? speakText(latestSpeechText) : speakResults(true));
  micBtn.addEventListener("click", () => {
    if (recognition) {
      try { recognition.start(); } catch (e) { voiceStatus.textContent = "Voice input is already listening."; }
    }
  });
  askBtn.addEventListener("click", () => answerQuestion(questionInput.value));
  questionInput.addEventListener("keydown", e => { if (e.key === "Enter") answerQuestion(questionInput.value); });
  window.addEventListener("resize", () => {
    if (video.classList.contains("active") && latestPredictions.length) drawPredictions(latestPredictions, video.videoWidth, video.videoHeight);
    else if (uploadedImage.classList.contains("active") && latestPredictions.length) drawPredictions(latestPredictions, uploadedImage.naturalWidth, uploadedImage.naturalHeight);
  });
  if ("speechSynthesis" in window) window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
  setupSpeechRecognition();
  loadModel();
})();

/* NETRA Map module: current location, geocoding, nearby services and directions. */
(() => {
  const mapEl = document.getElementById("netraMap");
  if (!mapEl) return;
  const statusEl = document.getElementById("locationStatus");
  const messageEl = document.getElementById("mapMessage");
  const locateBtn = document.getElementById("locateBtn");
  const searchForm = document.getElementById("mapSearchForm");
  const destinationInput = document.getElementById("destinationInput");
  const nearbyCategory = document.getElementById("nearbyCategory");
  const nearbyBtn = document.getElementById("nearbyBtn");
  const routeBtn = document.getElementById("routeBtn");

  if (typeof L === "undefined") {
    statusEl.textContent = "Map library could not load. Check your internet connection and refresh.";
    return;
  }

  // General India view only; this is not presented as the user's location.
  const map = L.map("netraMap", { zoomControl: true }).setView([22.9734, 78.6569], 5);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
  }).addTo(map);

  let userCoords = null;
  let destinationCoords = null;
  let userMarker = null;
  let destinationMarker = null;
  let nearbyMarkers = [];
  let searchRequest = 0;

  function announce(message) { messageEl.textContent = message; }
  function updateRouteButton() { routeBtn.disabled = !(userCoords && destinationCoords); }

  function setUserLocation(lat, lon) {
    userCoords = { lat, lon };
    if (userMarker) map.removeLayer(userMarker);
    userMarker = L.marker([lat, lon]).addTo(map).bindPopup("Your current location");
    map.setView([lat, lon], 16);
    statusEl.textContent = "Your location is shown on the map.";
    announce("Location found. Search for a destination or nearby service.");
    updateRouteButton();
  }
  map.on("click", (event) => {
    destinationCoords = { lat: event.latlng.lat, lon: event.latlng.lng };
    if (destinationMarker) map.removeLayer(destinationMarker);
    destinationMarker = L.marker([destinationCoords.lat, destinationCoords.lon]).addTo(map)
      .bindPopup("Selected destination").openPopup();
    announce("Destination selected from the map. If your location is set, you can open walking directions.");
    updateRouteButton();
  });

  locateBtn.addEventListener("click", () => {
    if (!navigator.geolocation) {
      announce("This browser does not support location. You can still search for a destination.");
      return;
    }
    locateBtn.disabled = true;
    announce("Requesting location permission…");
    navigator.geolocation.getCurrentPosition((position) => {
      setUserLocation(position.coords.latitude, position.coords.longitude);
      locateBtn.disabled = false;
    }, (error) => {
      const errors = {
        1: "Location permission was denied. Enable it in browser settings if you want to use this feature.",
        2: "Your location is currently unavailable. Try again or search for a place manually.",
        3: "Location request timed out. Please try again."
      };
      statusEl.textContent = errors[error.code] || "Could not get your location.";
      announce(statusEl.textContent);
      locateBtn.disabled = false;
    }, { enableHighAccuracy: true, timeout: 15000, maximumAge: 30000 });
  });

  searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = destinationInput.value.trim();
    if (!query) { announce("Enter a destination or landmark first."); destinationInput.focus(); return; }
    const requestId = ++searchRequest;
    announce("Searching for destination…");
    try {
      const url = new URL("https://nominatim.openstreetmap.org/search");
      url.searchParams.set("q", query);
      url.searchParams.set("format", "jsonv2");
      url.searchParams.set("limit", "1");
      const response = await fetch(url.toString(), { headers: { "Accept": "application/json" } });
      if (!response.ok) throw new Error("search failed");
      const results = await response.json();
      if (requestId !== searchRequest) return;
      if (!results.length) { destinationCoords = null; updateRouteButton(); announce("No result found. Try a more specific place name."); return; }
      const place = results[0];
      destinationCoords = { lat: Number(place.lat), lon: Number(place.lon) };
      if (destinationMarker) map.removeLayer(destinationMarker);
      destinationMarker = L.marker([destinationCoords.lat, destinationCoords.lon]).addTo(map)
        .bindPopup(String(place.display_name || query));
      map.setView([destinationCoords.lat, destinationCoords.lon], 16);
      destinationMarker.openPopup();
      announce("Destination found: " + (place.display_name || query));
      updateRouteButton();
    } catch (_) {
      announce("Destination search is temporarily unavailable. Check your connection and try again.");
    }
  });

  routeBtn.addEventListener("click", () => {
    if (!userCoords || !destinationCoords) return;
    const url = new URL("https://www.google.com/maps/dir/");
    url.searchParams.set("api", "1");
    url.searchParams.set("origin", `${userCoords.lat},${userCoords.lon}`);
    url.searchParams.set("destination", `${destinationCoords.lat},${destinationCoords.lon}`);
    url.searchParams.set("travelmode", "walking");
    window.open(url.toString(), "_blank", "noopener,noreferrer");
  });

  const nearbySelectors = {
    pharmacy: 'node["amenity"="pharmacy"]',
    hospital: 'node["amenity"="hospital"]',
    cafe: 'node["amenity"="cafe"]',
    restaurant: 'node["amenity"="restaurant"]',
    bus_stop: 'node["highway"="bus_stop"]',
    toilets: 'node["amenity"="toilets"]',
    atm: 'node["amenity"="atm"]',
    grocery: 'node["shop"="supermarket"]',
    railway: 'node["railway"="station"]',
    library: 'node["amenity"="library"]'
  };
  nearbyBtn.addEventListener("click", async () => {
    if (!userCoords) {
      announce("Choose My location first to search nearby services.");
      return;
    }
    const selector = nearbySelectors[nearbyCategory.value];
    if (!selector) return;
    nearbyBtn.disabled = true;
    announce("Searching nearby places within about 1.5 km…");
    const { lat, lon } = userCoords;
    const query = `[out:json][timeout:20];(${selector}(around:1500,${lat},${lon}););out center 15;`;
    const endpoints = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"];
    try {
      let data = null;
      for (const endpoint of endpoints) {
        try {
          const response = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "text/plain;charset=UTF-8" },
            body: query
          });
          if (response.ok) { data = await response.json(); break; }
        } catch (_) { /* try alternate public endpoint */ }
      }
      if (!data) throw new Error("nearby service unavailable");
      nearbyMarkers.forEach(marker => map.removeLayer(marker));
      nearbyMarkers = [];
      const places = (data.elements || []).map(item => ({
        item,
        lat: item.lat ?? item.center?.lat,
        lon: item.lon ?? item.center?.lon
      })).filter(place => Number.isFinite(place.lat) && Number.isFinite(place.lon));
      places.forEach(({ item, lat: placeLat, lon: placeLon }) => {
        const name = item.tags?.name || nearbyCategory.options[nearbyCategory.selectedIndex].text;
        const marker = L.marker([placeLat, placeLon]).addTo(map).bindPopup(String(name));
        marker.on("click", () => {
          destinationCoords = { lat: placeLat, lon: placeLon };
          if (destinationMarker) map.removeLayer(destinationMarker);
          destinationMarker = marker;
          destinationInput.value = String(name);
          updateRouteButton();
          announce(`${name} selected. Open walking directions if your current location is set.`);
        });
        nearbyMarkers.push(marker);
      });
      if (!places.length) { announce("No mapped places found nearby. Try another category or search manually."); return; }
      const points = [[lat, lon], ...places.map(place => [place.lat, place.lon])];
      map.fitBounds(L.latLngBounds(points), { padding: [25, 25], maxZoom: 16 });
      announce(`${places.length} nearby mapped place(s) found. Select a marker to read its name.`);
    } catch (_) {
      announce("Nearby search is temporarily unavailable. Public map services can be busy; try again later.");
    } finally {
      nearbyBtn.disabled = false;
    }
  });

  window.addEventListener("load", () => map.invalidateSize());
  window.addEventListener("resize", () => map.invalidateSize());
  requestAnimationFrame(() => map.invalidateSize());
})();
