import config from './config';
import { get, post, upload } from './request';

const API_BASE_URL = config.apiBaseUrl;

// ---------------------------------------------------------------------------
// API functions — pages import from here only, never from request.js directly
// ---------------------------------------------------------------------------

/**
 * Upload a sprint video with session labels.
 * POST /api/videos/upload (multipart/form-data)
 * @param {object} params
 * @param {string} params.videoPath
 * @param {string} params.athlete_id
 * @param {string} params.session_type
 * @param {string} params.fatigue_state
 * @param {string} params.event_group
 * @param {number} params.fileSizeMB  estimated file size in MB for timeout calculation
 * @param {function} params.onProgress  progress callback (0-100)
 */
export function uploadVideo({ videoPath, athlete_id, session_type, fatigue_state, event_group, fileSizeMB = 10, onProgress = null }) {
  console.log('[api] uploadVideo ->', `${API_BASE_URL}/api/videos/upload`, { fileSizeMB });
  return upload('/api/videos/upload', videoPath, {
    athlete_id,
    session_type,
    fatigue_state,
    event_group,
    _fileSizeMB: fileSizeMB,
  }, onProgress);
}

/**
 * Poll analysis job status for a given video.
 * GET /api/analysis/{video_id}/status
 */
export function getAnalysisStatus(videoId) {
  console.log('[api] getAnalysisStatus ->', `${API_BASE_URL}/api/analysis/${videoId}/status`);
  return get(`/api/analysis/${videoId}/status`);
}

/**
 * Fetch single analysis report (5 metrics + comparison with previous).
 * GET /api/reports/{video_id}
 */
export function getReport(videoId) {
  console.log('[api] getReport ->', `${API_BASE_URL}/api/reports/${videoId}`);
  return get(`/api/reports/${videoId}`);
}

/**
 * Manually save this finished report into history/trend.
 * POST /api/reports/{video_id}/save
 */
export function saveReport(videoId) {
  console.log('[api] saveReport ->', `${API_BASE_URL}/api/reports/${videoId}/save`);
  return post(`/api/reports/${videoId}/save`, {});
}

/**
 * Trend / history for an athlete (chronological records for charts + list).
 * GET /api/athletes/{athlete_id}/trend
 * (Backend also exposes GET /api/athletes/{athlete_id}/history with the same shape.)
 */
export function getHistory(athleteId) {
  console.log('[api] getHistory ->', `${API_BASE_URL}/api/athletes/${athleteId}/trend`);
  return get(`/api/athletes/${athleteId}/trend`).then((body) => {
    if (!body || !Array.isArray(body.history)) return [];
    return body.history;
  });
}

/**
 * Get AI analysis for a video (DeepSeek-powered).
 * GET /api/ai-analysis/{video_id}
 */
const AI_TIMEOUT = 180000;

export function getAIAnalysis(videoId, forceRefresh = false) {
  const url = `/api/ai-analysis/${videoId}${forceRefresh ? '?force_refresh=true' : ''}`;
  console.log('[api] getAIAnalysis ->', `${API_BASE_URL}${url}`);
  return get(url, {}, { timeout: AI_TIMEOUT });
}

/**
 * Refresh AI analysis for a video (clear cache and regenerate).
 * POST /api/ai-analysis/{video_id}/refresh
 */
export function refreshAIAnalysis(videoId) {
  console.log('[api] refreshAIAnalysis ->', `${API_BASE_URL}/api/ai-analysis/${videoId}/refresh`);
  return post(`/api/ai-analysis/${videoId}/refresh`, {}, { timeout: AI_TIMEOUT });
}

/**
 * Debug: upload video for pose landmark extraction only (no analysis job).
 * POST /api/debug/extract-landmarks (multipart/form-data)
 * Returns compact per-frame landmarks for the debug overlay canvas.
 * @param {object} params
 * @param {string} params.videoPath
 * @param {number} params.fileSizeMB
 * @param {function} params.onProgress
 */
export function extractDebugLandmarks({ videoPath, fileSizeMB = 10, onProgress = null }) {
  console.log('[api] extractDebugLandmarks ->', `${API_BASE_URL}/api/debug/extract-landmarks`);
  return upload('/api/debug/extract-landmarks', videoPath, {
    _fileSizeMB: Math.max(fileSizeMB, 80),
  }, onProgress);
}
