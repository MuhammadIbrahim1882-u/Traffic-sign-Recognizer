(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var els = {
    viewfinder: $("viewfinder"), drop: $("drop"), preview: $("preview"), video: $("video"),
    busy: $("busy"), input: $("fileInput"),
    chooseBtn: $("chooseBtn"), cameraBtn: $("cameraBtn"),
    captureBtn: $("captureBtn"), closeCameraBtn: $("closeCameraBtn"),
    hint: $("modeHint"), decision: $("decision"), chip: $("levelChip"),
    text: $("decisionText"), name: $("signName"), meter: $("meter"),
    confValue: $("confValue"), confBar: $("confBar"),
    top: $("topList"), found: $("foundList"), empty: $("emptyText")
  };

  var HINTS = {
    sign: "Single sign: use a close-up photo where the sign fills most of the picture.",
    scene: "Road scene: the car first looks for red, blue and yellow signs in the picture, then reads each one."
  };
  var CHIPS = { stop: "Stop", caution: "Slow down", info: "Follow the sign", ok: "Clear to continue" };

  var mode = "sign";
  var currentFile = null;
  var originalUrl = null;
  var stream = null;
  var requestId = 0;

  // ---------- helpers ----------
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function make(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function showImage(url) {
    els.preview.src = url;
    els.preview.hidden = false;
    els.drop.hidden = true;
  }

  function setBusy(on) {
    els.viewfinder.classList.toggle("analyzing", on);
    els.busy.hidden = !on;
  }

  function resetResult() {
    els.meter.hidden = true;
    els.top.hidden = true;
    els.found.hidden = true;
    els.name.textContent = "";
    clear(els.top);
    clear(els.found);
  }

  function setDecision(level, text) {
    els.decision.setAttribute("data-level", level);
    els.chip.textContent = CHIPS[level] || "Result";
    els.text.textContent = text;
    els.empty.hidden = true;
  }

  function showError(message) {
    resetResult();
    els.decision.setAttribute("data-level", "error");
    els.chip.textContent = "Something went wrong";
    els.text.textContent = message;
    els.empty.hidden = true;
  }

  // ---------- rendering ----------
  function renderSign(data) {
    setDecision(data.decision.level, data.decision.text);
    els.name.textContent = "Recognized: " + data.name;
    els.confValue.textContent = data.confidence.toFixed(1) + "%";
    els.confBar.style.width = Math.max(2, data.confidence) + "%";
    els.meter.hidden = false;

    clear(els.top);
    data.top.forEach(function (item) {
      var li = make("li");
      var row = make("div", "row");
      row.appendChild(make("span", "", item.name));
      row.appendChild(make("span", "", item.confidence.toFixed(1) + "%"));
      var bar = make("div", "bar");
      var fill = make("span");
      fill.style.width = Math.max(1, item.confidence) + "%";
      bar.appendChild(fill);
      li.appendChild(row);
      li.appendChild(bar);
      els.top.appendChild(li);
    });
    els.top.hidden = false;
  }

  function renderScene(data) {
    setDecision(data.decision.level, data.decision.text);
    els.meter.hidden = true;
    els.top.hidden = true;
    clear(els.found);
    if (data.annotated) showImage(data.annotated);

    if (data.detections.length === 0) {
      els.name.textContent = "No sign passed the confidence check. If one is in the picture, try a sharper or closer image.";
      els.found.hidden = true;
      return;
    }
    els.name.textContent = data.detections.length === 1 ? "1 sign found" : data.detections.length + " signs found";
    data.detections.forEach(function (d) {
      var li = make("li");
      li.setAttribute("data-level", d.level);
      li.appendChild(make("div", "f-name", d.name + " (" + d.confidence.toFixed(0) + "%)"));
      li.appendChild(make("div", "f-meta", d.action));
      els.found.appendChild(li);
    });
    els.found.hidden = false;
  }

  // ---------- server call ----------
  function analyze() {
    if (!currentFile) return;
    var myId = ++requestId;
    var form = new FormData();
    form.append("image", currentFile, currentFile.name || "image.jpg");
    form.append("mode", mode);
    setBusy(true);

    fetch("/api/predict", { method: "POST", body: form })
      .then(function (res) {
        return res.json().catch(function () { return { ok: false, error: "The server sent an unexpected reply." }; });
      })
      .then(function (data) {
        if (myId !== requestId) return; // a newer request replaced this one
        if (!data.ok) { showError(data.error || "The image could not be analyzed."); return; }
        if (data.mode === "scene") renderScene(data); else renderSign(data);
      })
      .catch(function () {
        if (myId !== requestId) return;
        showError("Could not reach the app. Check that 'python app.py' is still running in your terminal.");
      })
      .then(function () {
        if (myId === requestId) setBusy(false);
      });
  }

  function setFile(file) {
    if (!file) return;
    if (file.type && file.type.indexOf("image/") !== 0) {
      showError("That file is not an image. Choose a PNG, JPG or BMP.");
      return;
    }
    currentFile = file;
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    originalUrl = URL.createObjectURL(file);
    showImage(originalUrl);
    resetResult();
    analyze();
  }

  // ---------- mode switch ----------
  document.querySelectorAll(".mode-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var next = btn.getAttribute("data-mode");
      if (next === mode) return;
      mode = next;
      document.querySelectorAll(".mode-btn").forEach(function (b) {
        b.setAttribute("aria-pressed", b === btn ? "true" : "false");
      });
      els.hint.textContent = HINTS[mode];
      if (currentFile) {
        showImage(originalUrl);
        resetResult();
        analyze();
      }
    });
  });

  // ---------- file input and drag/drop ----------
  els.chooseBtn.addEventListener("click", function () { els.input.click(); });
  els.input.addEventListener("change", function () {
    setFile(els.input.files[0]);
    els.input.value = "";
  });
  els.drop.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.input.click(); }
  });

  ["dragenter", "dragover"].forEach(function (type) {
    els.viewfinder.addEventListener(type, function (e) {
      e.preventDefault();
      els.viewfinder.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach(function (type) {
    els.viewfinder.addEventListener(type, function (e) {
      e.preventDefault();
      els.viewfinder.classList.remove("dragging");
    });
  });
  els.viewfinder.addEventListener("drop", function (e) {
    if (e.dataTransfer && e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });

  // ---------- camera ----------
  function openCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("The camera is not available here. Open http://127.0.0.1:5000 in a recent Chrome or Edge.");
      return;
    }
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false })
      .then(function (s) {
        stream = s;
        els.video.srcObject = s;
        return els.video.play();
      })
      .then(function () {
        els.video.hidden = false;
        els.preview.hidden = true;
        els.drop.hidden = true;
        els.captureBtn.hidden = false;
        els.closeCameraBtn.hidden = false;
        els.cameraBtn.hidden = true;
      })
      .catch(function () {
        closeCamera();
        showError("Could not open the camera. Allow camera access in the browser and try again.");
      });
  }

  function closeCamera() {
    if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
    stream = null;
    els.video.srcObject = null;
    els.video.hidden = true;
    els.captureBtn.hidden = true;
    els.closeCameraBtn.hidden = true;
    els.cameraBtn.hidden = false;
    if (originalUrl) showImage(originalUrl); else { els.preview.hidden = true; els.drop.hidden = false; }
  }

  function capture() {
    var w = els.video.videoWidth, h = els.video.videoHeight;
    if (!w || !h) { showError("The camera is not ready yet. Wait a second and try again."); return; }
    var canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    canvas.getContext("2d").drawImage(els.video, 0, 0, w, h);
    canvas.toBlob(function (blob) {
      if (!blob) { showError("The camera frame could not be captured."); return; }
      closeCamera();
      setFile(new File([blob], "camera.jpg", { type: "image/jpeg" }));
    }, "image/jpeg", 0.92);
  }

  els.cameraBtn.addEventListener("click", openCamera);
  els.closeCameraBtn.addEventListener("click", closeCamera);
  els.captureBtn.addEventListener("click", capture);
  window.addEventListener("beforeunload", function () { if (stream) closeCamera(); });
})();
