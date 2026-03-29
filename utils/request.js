import config from './config';

function buildRequestMeta(method, path) {
  const runtimeEnv = config.runtimeEnv || {};
  return {
    method,
    path,
    url: config.apiBaseUrl + path,
    env: runtimeEnv.client || 'unknown',
    runtime: runtimeEnv.runtime || 'unknown',
    platform: runtimeEnv.platform || 'unknown',
    envName: config.envName || 'unknown',
  };
}

function buildError({
  code,
  message,
  statusCode = null,
  errMsg = '',
  data = null,
  meta = {},
  raw = null,
}) {
  return {
    code,
    message,
    statusCode,
    errMsg,
    url: meta.url,
    method: meta.method,
    path: meta.path,
    env: meta.env,
    runtime: meta.runtime,
    platform: meta.platform,
    envName: meta.envName,
    data,
    raw,
  };
}

function validateBaseUrl(meta) {
  const baseUrl = String(config.apiBaseUrl || '');
  if (!baseUrl) {
    return buildError({
      code: -1,
      message: 'apiBaseUrl 未配置，无法发起请求',
      errMsg: 'missing apiBaseUrl',
      meta,
    });
  }
  if (!/^https?:\/\//i.test(baseUrl)) {
    return buildError({
      code: -1,
      message: `apiBaseUrl 非法：${baseUrl}`,
      errMsg: 'invalid apiBaseUrl protocol',
      meta,
    });
  }
  if (/example\.com/i.test(baseUrl)) {
    return buildError({
      code: -1,
      message: '当前 apiBaseUrl 仍是示例域名，请替换成真实后端地址后再重试',
      errMsg: 'placeholder apiBaseUrl',
      meta,
    });
  }
  if (config.isDevtools) {
    return null;
  }
  if (config.isLocalhostLike) {
    return buildError({
      code: -1,
      message: '当前为真机/体验版环境，不能请求 localhost 或 127.0.0.1；必须改为公网 HTTPS 域名',
      errMsg: 'localhost not reachable on real-device',
      meta,
    });
  }
  if (/^http:\/\//i.test(baseUrl)) {
    return buildError({
      code: -1,
      message: '当前为真机/体验版环境，必须使用公网 HTTPS 域名，不能继续使用 HTTP 地址',
      errMsg: 'http not allowed on real-device',
      meta,
    });
  }
  return null;
}

function handleResponse(statusCode, data, meta) {
  if (statusCode >= 200 && statusCode < 300) {
    return data;
  }
  const message = (data && data.detail) || `请求失败 (${statusCode})`;
  const error = buildError({
    code: statusCode,
    message,
    statusCode,
    data,
    meta,
  });
  console.error('[request:http-fail]', error);
  throw error;
}

function buildNetworkErrorMessage(err, meta) {
  const errMsg = err && err.errMsg ? String(err.errMsg) : 'wx.request:fail';
  if (config.isDevtools) {
    return `网络错误，请检查开发者工具是否真的连到后端。当前 URL: ${meta.url}；如需本地调试，请确认后端已启动，并使用本机可访问地址（127.0.0.1 或局域网 IP），必要时关闭合法域名校验。`;
  }
  return `网络错误，请检查公网 HTTPS 域名、服务器可达性与小程序合法域名配置。当前 URL: ${meta.url}；errMsg: ${errMsg}`;
}

function request(method, path, data = {}, { timeout = 60000 } = {}) {
  const meta = buildRequestMeta(method, path);
  const configError = validateBaseUrl(meta);
  if (configError) {
    console.error('[request:block]', configError);
    return Promise.reject(configError);
  }
  console.log('[request:start]', {
    method: meta.method,
    url: meta.url,
    env: meta.env,
    runtime: meta.runtime,
    platform: meta.platform,
    envName: meta.envName,
    timeout,
  });
  return new Promise((resolve, reject) => {
    wx.request({
      url: meta.url,
      method,
      data,
      timeout,
      header: { 'Content-Type': 'application/json' },
      success(res) {
        try {
          resolve(handleResponse(res.statusCode, res.data, meta));
        } catch (err) {
          reject(err);
        }
      },
      fail(err) {
        const error = buildError({
          code: -1,
          message: buildNetworkErrorMessage(err, meta),
          statusCode: null,
          errMsg: err && err.errMsg ? String(err.errMsg) : 'wx.request:fail',
          meta,
          raw: err,
        });
        console.error('[request:fail]', error);
        reject(error);
      },
    });
  });
}

/**
 * Unified GET request.
 * @param {string} path  e.g. '/api/reports/mock-001'
 * @param {object} params  query params appended to URL
 */
export function get(path, params = {}, options = {}) {
  return request('GET', path, params, options);
}

/**
 * Unified POST request.
 * @param {string} path  e.g. '/api/athletes'
 * @param {object} data  request body
 * @param {object} options  e.g. { timeout: 120000 }
 */
export function post(path, data = {}, options = {}) {
  return request('POST', path, data, options);
}

/**
 * File upload via wx.uploadFile (multipart/form-data).
 * @param {string} path      e.g. '/api/videos/upload'
 * @param {string} filePath  local temp file path from wx.chooseMedia
 * @param {object} formData  additional form fields
 * @param {function} onProgress  progress callback (progress: number) => void
 */
export function upload(path, filePath, formData = {}, onProgress = null) {
  const apiBaseUrl = config.apiBaseUrl;
  const uploadPath = path;
  const fullUrl = apiBaseUrl + path;

  // 计算预估超时时间：基础 30 秒 + 每 MB 3 秒，最大 300 秒
  const fileSizeMB = formData._fileSizeMB || 10;
  const timeoutMs = Math.min(300000, Math.max(60000, 30000 + fileSizeMB * 3000));

  console.log('[upload] start', { fullUrl, filePath, timeoutMs, fileSizeMB });

  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    const uploadTask = wx.uploadFile({
      url: fullUrl,
      filePath,
      name: 'file',
      formData,
      timeout: timeoutMs,
      success(res) {
        let data;
        try {
          data = typeof res.data === 'string' ? JSON.parse(res.data) : res.data;
        } catch (parseErr) {
          console.error('[upload:fail] JSON parse error (raw)', parseErr);
          console.error('[upload:fail] context', {
            apiBaseUrl,
            uploadPath,
            fullUrl,
            fileFieldName: 'file',
            formData,
            statusCode: res.statusCode,
            rawBody: res.data,
          });
          reject({
            code: res.statusCode || -1,
            message: '响应解析失败',
            raw: parseErr,
            rawBody: res.data,
            apiBaseUrl,
            uploadPath,
            fullUrl,
          });
          return;
        }
        try {
          resolve(handleResponse(res.statusCode, data));
        } catch (httpErr) {
          console.error('[upload:fail] HTTP / business error (raw)', httpErr);
          console.error('[upload:fail] context', {
            apiBaseUrl,
            uploadPath,
            fullUrl,
            fileFieldName: 'file',
            formData,
            statusCode: res.statusCode,
            parsedBody: data,
          });
          if (httpErr && typeof httpErr === 'object') {
            reject({
              ...httpErr,
              apiBaseUrl,
              uploadPath,
              fullUrl,
              statusCode: res.statusCode,
              parsedBody: data,
            });
          } else {
            reject(httpErr);
          }
        }
      },
      fail(err) {
        const elapsed = Date.now() - startTime;
        console.error('[upload:fail] wx.uploadFile fail (raw)', err);
        console.error('[upload:fail] context', {
          apiBaseUrl,
          uploadPath,
          fullUrl,
          fileFieldName: 'file',
          formData,
          wxErrMsg: err && err.errMsg,
          wxErrno: err && err.errno,
          timeoutMs,
          elapsedMs: elapsed,
        });
        reject({
          code: err && typeof err.errno === 'number' ? err.errno : -1,
          message: (err && err.errMsg) || 'wx.uploadFile 调用失败',
          raw: err,
          apiBaseUrl,
          uploadPath,
          fullUrl,
          timeoutMs,
          elapsedMs: elapsed,
        });
      },
    });

    // 进度监听
    if (uploadTask && uploadTask.onProgressUpdate && onProgress) {
      uploadTask.onProgressUpdate((res) => {
        if (res && typeof res.progress === 'number') {
          onProgress(res.progress);
        }
      });
    }
  });
}
