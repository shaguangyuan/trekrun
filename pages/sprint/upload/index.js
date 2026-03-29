import { uploadVideo, extractDebugLandmarks } from '../../../utils/api';
import config from '../../../utils/config';

const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [27, 29], [29, 31], [24, 26], [26, 28], [28, 30], [30, 32],
];
const IDX_LEFT_HIP = 23;
const IDX_LEFT_KNEE = 25;
const IDX_LEFT_ANKLE = 27;
const IDX_LEFT_HEEL = 29;
const IDX_LEFT_FOOT_INDEX = 31;

Page({
  data: {
    videoPath: '',
    videoName: '',
    videoSizeMB: 10,
    form: {
      athlete_id: '',
      event_group: '100',
      session_type: 'normal',
      fatigue_state: 'no',
    },
    canSubmit: false,
    uploading: false,
    uploadProgress: 0,
    debugEnabled: true,
    showLandmarks: true,
    showConnections: true,
    showIndex: true,
    showVisibility: false,
    visibilityThreshold: 0.45,
    overlayWidth: 300,
    overlayHeight: 200,
    hasLandmarks: false,
    codecHintVisible: false,
    codecHintText: '',
    videoPlayErrorText: '',
    debugLoading: false,
    debugLoadingText: '',
  },

  onReady() {
    this.initPoseLandmarker();
    this.initVideoPoseDebug();
  },

  onUnload() {
    this._debugFrames = null;
    this._overlayCanvas = null;
    this._overlayCtx = null;
    this._debugRequestId = 0;
  },

  /* ===========================================================
   * Pose Debug Layer — 4 core functions
   * =========================================================== */

  initPoseLandmarker() {
    this._debugFrames = null;
    this._debugFps = 0;
    this._videoNativeWidth = 0;
    this._videoNativeHeight = 0;
    this._debugRequestId = 0;
    this._lastHasLandmarks = false;
    console.log('[pose-debug] initPoseLandmarker: ready for backend extraction');
  },

  initVideoPoseDebug() {
    wx.createSelectorQuery()
      .in(this)
      .select('#poseOverlay')
      .fields({ node: true, size: true })
      .exec((res) => {
        if (!res || !res[0] || !res[0].node) {
          console.warn('[pose-debug] Canvas 2D node not available');
          return;
        }
        const canvas = res[0].node;
        const ctx = canvas.getContext('2d');
        this._overlayCanvas = canvas;
        this._overlayCtx = ctx;

        const sysInfo = wx.getWindowInfo ? wx.getWindowInfo() : wx.getSystemInfoSync();
        this._dpr = sysInfo.pixelRatio || 2;

        const cssW = res[0].width || 300;
        const cssH = res[0].height || 200;
        this._canvasCssWidth = cssW;
        this._canvasCssHeight = cssH;
        canvas.width = Math.floor(cssW * this._dpr);
        canvas.height = Math.floor(cssH * this._dpr);

        console.log('[pose-debug] initVideoPoseDebug: canvas', canvas.width, 'x', canvas.height, 'dpr', this._dpr);
      });
  },

  predictLoop(frameInfo = {}) {
    const currentTimeMs = (frameInfo.currentTimeSec || 0) * 1000;
    let landmarks = null;

    if (this._debugFrames && this._debugFrames.length > 0) {
      landmarks = this._findLandmarksByTime(currentTimeMs);
    }

    if (
      !landmarks &&
      typeof globalThis !== 'undefined' &&
      typeof globalThis.__POSE_DEBUG_GET_LANDMARKS__ === 'function'
    ) {
      try {
        const r = globalThis.__POSE_DEBUG_GET_LANDMARKS__(
          frameInfo.currentTimeSec || 0,
          this.data.videoPath || '',
        );
        landmarks = Array.isArray(r) ? r : null;
      } catch (e) {
        console.error('[pose-debug] external getLandmarks error', e);
      }
    }

    this.renderPoseOverlay(landmarks || []);
  },

  renderPoseOverlay(landmarks) {
    const ctx = this._overlayCtx;
    const canvas = this._overlayCanvas;
    if (!ctx || !canvas) return;

    const cw = canvas.width;
    const ch = canvas.height;
    const dpr = this._dpr || 2;
    const {
      visibilityThreshold,
      showLandmarks,
      showConnections,
      showIndex,
      showVisibility,
    } = this.data;

    const hasLm = Array.isArray(landmarks) && landmarks.length > 0;
    if (this._lastHasLandmarks !== hasLm) {
      this._lastHasLandmarks = hasLm;
      this.setData({ hasLandmarks: hasLm });
    }

    ctx.clearRect(0, 0, cw, ch);
    if (!hasLm) return;

    // object-fit: contain — compute rendered video area inside canvas
    const vw = this._videoNativeWidth || 1;
    const vh = this._videoNativeHeight || 1;
    const dispW = this._canvasCssWidth || cw / dpr;
    const dispH = this._canvasCssHeight || ch / dpr;

    const videoAR = vw / vh;
    const containerAR = dispW / dispH;
    let renderedW, renderedH, offsetX, offsetY;
    if (videoAR > containerAR) {
      renderedW = dispW;
      renderedH = dispW / videoAR;
      offsetX = 0;
      offsetY = (dispH - renderedH) / 2;
    } else {
      renderedH = dispH;
      renderedW = dispH * videoAR;
      offsetX = (dispW - renderedW) / 2;
      offsetY = 0;
    }

    const ox = offsetX * dpr;
    const oy = offsetY * dpr;
    const rw = renderedW * dpr;
    const rh = renderedH * dpr;
    const toX = (nx) => ox + nx * rw;
    const toY = (ny) => oy + ny * rh;

    if (showConnections) {
      ctx.lineWidth = 2 * dpr;
      const VIS_CONN_FLOOR = 0.12;
      POSE_CONNECTIONS.forEach(([a, b]) => {
        const p1 = landmarks[a];
        const p2 = landmarks[b];
        if (!p1 || !p2) return;
        if ((p1.visibility || 0) < VIS_CONN_FLOOR || (p2.visibility || 0) < VIS_CONN_FLOOR) return;
        const low =
          (p1.visibility || 0) < visibilityThreshold ||
          (p2.visibility || 0) < visibilityThreshold;
        ctx.strokeStyle = low ? '#ff8a00' : '#00d084';
        ctx.beginPath();
        ctx.moveTo(toX(p1.x), toY(p1.y));
        ctx.lineTo(toX(p2.x), toY(p2.y));
        ctx.stroke();
      });
    }

    if (showLandmarks) {
      const radius = 3 * dpr;
      const VIS_FLOOR = 0.12;
      landmarks.forEach((p, idx) => {
        if (!p || (p.visibility || 0) < VIS_FLOOR) return;
        const x = toX(p.x);
        const y = toY(p.y);
        const low = (p.visibility || 0) < visibilityThreshold;

        ctx.fillStyle = low ? '#ff3b30' : '#0a84ff';
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();

        if (showIndex) {
          ctx.fillStyle = '#ffffff';
          ctx.font = `bold ${10 * dpr}px sans-serif`;
          ctx.fillText(String(idx), x + 4 * dpr, y - 4 * dpr);
        }
        if (showVisibility) {
          ctx.fillStyle = '#ffd60a';
          ctx.font = `${9 * dpr}px sans-serif`;
          ctx.fillText(
            `v:${(p.visibility || 0).toFixed(2)}`,
            x + 4 * dpr,
            y + 12 * dpr,
          );
        }
      });

      const hip = landmarks[IDX_LEFT_HIP];
      const knee = landmarks[IDX_LEFT_KNEE];
      const ankle = landmarks[IDX_LEFT_ANKLE];
      const heel = landmarks[IDX_LEFT_HEEL];
      const foot = landmarks[IDX_LEFT_FOOT_INDEX];
      if (hip || knee || ankle || heel || foot) {
        console.log('[pose-debug:left-leg]', {
          left_hip: hip ? { x: hip.x, y: hip.y, v: hip.visibility } : null,
          left_knee: knee ? { x: knee.x, y: knee.y, v: knee.visibility } : null,
          left_ankle: ankle
            ? { x: ankle.x, y: ankle.y, v: ankle.visibility }
            : null,
          left_heel: heel
            ? { x: heel.x, y: heel.y, v: heel.visibility }
            : null,
          left_foot_index: foot
            ? { x: foot.x, y: foot.y, v: foot.visibility }
            : null,
        });
      }
    }
  },

  /* ===========================================================
   * Debug helpers
   * =========================================================== */

  _findLandmarksByTime(timeMs) {
    const frames = this._debugFrames;
    if (!frames || frames.length === 0) return null;
    let lo = 0;
    let hi = frames.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (frames[mid].t < timeMs) lo = mid + 1;
      else hi = mid;
    }
    if (
      lo > 0 &&
      Math.abs(frames[lo - 1].t - timeMs) < Math.abs(frames[lo].t - timeMs)
    ) {
      lo--;
    }
    const f = frames[lo];
    if (!f || !f.lm) return null;
    return f.lm.map((p) => ({
      x: p.x,
      y: p.y,
      z: p.z,
      visibility: p.v,
    }));
  },

  _fetchDebugLandmarks(videoPath, sizeMB) {
    if (!this.data.debugEnabled) return;

    this._debugRequestId = Date.now();
    const requestId = this._debugRequestId;

    this._debugFrames = null;
    this.setData({
      debugLoading: true,
      debugLoadingText: '正在提取姿态关键点…',
      hasLandmarks: false,
    });
    console.log('[pose-debug] starting landmark extraction');

    extractDebugLandmarks({
      videoPath,
      fileSizeMB: sizeMB,
      onProgress: (p) => {
        if (this._debugRequestId !== requestId) return;
        this.setData({ debugLoadingText: `上传视频中 ${Math.round(p)}%` });
      },
    })
      .then((res) => {
        if (this._debugRequestId !== requestId) return;
        this._debugFrames = res.frames || [];
        this._debugFps = res.fps || 30;
        this.setData({ debugLoading: false, debugLoadingText: '' });
        console.log(
          '[pose-debug] landmarks ready:',
          this._debugFrames.length,
          'frames with pose, fps',
          this._debugFps,
        );
      })
      .catch((err) => {
        if (this._debugRequestId !== requestId) return;
        this._debugFrames = null;
        this.setData({ debugLoading: false, debugLoadingText: '' });
        console.warn(
          '[pose-debug] extraction failed (non-blocking):',
          (err && err.message) || err,
        );
      });
  },

  _updateCanvasSize() {
    if (!this._overlayCanvas || !this._canvasCssWidth) return;
    const dpr = this._dpr || 2;
    this._overlayCanvas.width = Math.floor(this._canvasCssWidth * dpr);
    this._overlayCanvas.height = Math.floor(this._canvasCssHeight * dpr);
  },

  /* ===========================================================
   * Video debug events
   * =========================================================== */

  onDebugVideoMeta(e) {
    const d = (e && e.detail) || {};
    const w = Number(d.width) || 0;
    const h = Number(d.height) || 0;
    if (w > 0 && h > 0) {
      this._videoNativeWidth = w;
      this._videoNativeHeight = h;
      this.setData({ overlayWidth: w, overlayHeight: h });
      this._updateCanvasSize();
    }
  },

  onDebugVideoLoadedData(e) {
    console.log('[video-debug] loadeddata', (e && e.detail) || {});
    this.setData({ videoPlayErrorText: '' });
  },

  onDebugVideoPlay(e) {
    console.log('[video-debug] play', (e && e.detail) || {});
  },

  onDebugVideoError(e) {
    const detail = (e && e.detail) || {};
    const code = detail.errMsg || detail.code || 'unknown';
    const text =
      `视频可播放性异常（${code}）。常见原因是编码不兼容（如 H.265）。` +
      '建议转码为 H.264 + AAC 的 MP4 后再上传。';
    this.setData({ videoPlayErrorText: text });
    console.warn('[video-debug] play error', detail);
  },

  onDebugVideoTimeUpdate(e) {
    if (!this.data.debugEnabled) return;
    const d = (e && e.detail) || {};
    this.predictLoop({ currentTimeSec: Number(d.currentTime || 0) });
  },

  onToggleDebug(e) {
    const key = e.currentTarget.dataset.key;
    const checked = !!(e.detail && e.detail.value);
    if (!key) return;
    this.setData({ [key]: checked });
  },

  /* ===========================================================
   * Business logic — unchanged
   * =========================================================== */

  chooseVideo() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['video'],
      sourceType: ['album', 'camera'],
      maxDuration: 60,
      success: (res) => {
        const file = res.tempFiles[0];
        const parts = file.tempFilePath.split('/');
        const filename = parts[parts.length - 1] || '';
        const lower = filename.toLowerCase();
        const sizeMB = file.size ? file.size / 1024 / 1024 : 10;
        let codecHintVisible = false;
        let codecHintText = '';
        if (lower.endsWith('.mov')) {
          codecHintVisible = true;
          codecHintText =
            '检测到 MOV 文件。部分设备导出的 MOV 可能为 H.265，开发工具中可能出现"有声音无画面"。建议转码为 MP4(H.264 + AAC)。';
        } else if (sizeMB > 80) {
          codecHintVisible = true;
          codecHintText =
            '视频较大，若出现"有声音无画面"或上传超时，建议先转码为 MP4(H.264 + AAC)并适当压缩。';
        }
        this.setData({
          videoPath: file.tempFilePath,
          videoName: filename,
          videoSizeMB: Math.max(1, Math.round(sizeMB)),
          codecHintVisible,
          codecHintText,
          videoPlayErrorText: '',
        });
        this._checkCanSubmit();

        this._fetchDebugLandmarks(
          file.tempFilePath,
          Math.max(1, Math.round(sizeMB)),
        );
      },
    });
  },

  onAthleteIdInput(e) {
    this.setData({ 'form.athlete_id': e.detail.value });
    this._checkCanSubmit();
  },

  onEventGroupChange(e) {
    this.setData({ 'form.event_group': e.detail.value });
  },

  onSessionTypeChange(e) {
    this.setData({ 'form.session_type': e.detail.value });
  },

  onFatigueStateChange(e) {
    this.setData({ 'form.fatigue_state': e.detail.value });
  },

  _checkCanSubmit() {
    const { videoPath, form } = this.data;
    this.setData({ canSubmit: !!videoPath && !!form.athlete_id });
  },

  submit() {
    if (!this.data.canSubmit || this.data.uploading) return;

    const { videoPath, videoSizeMB, form } = this.data;

    this.setData({ uploading: true, uploadProgress: 0 });
    wx.showLoading({ title: '准备上传...', mask: true });

    uploadVideo({
      videoPath,
      fileSizeMB: videoSizeMB,
      onProgress: (progress) => {
        this.setData({ uploadProgress: progress });
        wx.showLoading({
          title: `上传中 ${Math.round(progress)}%`,
          mask: true,
        });
      },
      ...form,
    })
      .then((res) => {
        wx.hideLoading();
        const videoId = res && res.video_id ? String(res.video_id) : '';
        console.log('[upload] response video_id =', videoId);
        if (!videoId) {
          this.setData({ uploading: false });
          wx.showToast({ title: '上传成功但缺少video_id', icon: 'none' });
          return;
        }
        wx.navigateTo({
          url: `/pages/sprint/status/index?video_id=${encodeURIComponent(videoId)}`,
        });
      })
      .catch((err) => {
        wx.hideLoading();
        this.setData({ uploading: false, uploadProgress: 0 });
        console.error('[upload:page] failed raw error object:', err);
        console.error('[upload:page] debug fields:', {
          apiBaseUrl: config.apiBaseUrl,
          uploadPath: '/api/videos/upload',
          fullUrl: `${config.apiBaseUrl}/api/videos/upload`,
          code: err && err.code,
          message: err && err.message,
          statusCode: err && err.statusCode,
          raw: err && err.raw,
          rawBody: err && err.rawBody,
          parsedBody: err && err.parsedBody,
          timeoutMs: err && err.timeoutMs,
          elapsedMs: err && err.elapsedMs,
        });

        const isTimeout =
          err &&
          err.message &&
          (err.message.includes('timeout') || err.message.includes('超时'));

        const toastText = isTimeout
          ? '上传超时，请重试或压缩视频'
          : err && err.message
            ? String(err.message).length > 18
              ? `${String(err.message).slice(0, 18)}…`
              : String(err.message)
            : '上传失败';
        wx.showToast({ title: toastText, icon: 'none', duration: 3000 });
      });
  },
});
