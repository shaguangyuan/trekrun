const ENV_CONFIG = {
  // Mock mode: all API calls return local stub data. Backend not required.
  dev: {
    // DevTools temporary debugging: use a backend address reachable from this machine.
    devtoolsApiBaseUrl: 'http://127.0.0.1:8000',
    // Real device / preview / release must use a public HTTPS domain.
    deviceApiBaseUrl: 'https://api.example.com',
    useMock: true,
  },
  // Local real-backend mode: points to local FastAPI server, mock off.
  // Prerequisite in WeChat DevTools: 详情 → 本地设置 → 勾选"不校验合法域名"
  // 必须先启动后端，否则 upload 会 ECONNREFUSED：
  //   Windows: 双击 backend\start-server.bat
  //   或终端: cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  local: {
    // Prefer 127.0.0.1 over localhost to avoid environment ambiguity in DevTools.
    devtoolsApiBaseUrl: 'http://127.0.0.1:8000',
    // Replace with your real public HTTPS API domain before testing on real devices.
    deviceApiBaseUrl: 'https://api.example.com',
    useMock: false,
  },
  test: {
    devtoolsApiBaseUrl: 'https://sprint-apii-239166-8-1414555897.sh.run.tcloudbase.com',
    deviceApiBaseUrl: 'https://sprint-apii-239166-8-1414555897.sh.run.tcloudbase.com',
    useMock: false,
  },
  prod: {
    devtoolsApiBaseUrl: 'https://sprint-apii-239166-8-1414555897.sh.run.tcloudbase.com',
    deviceApiBaseUrl: 'https://sprint-apii-239166-8-1414555897.sh.run.tcloudbase.com',
    useMock: false,
  },
};

// Switch to 'local' to use the local real backend configuration.
const CURRENT_ENV = 'local';

function getRuntimeEnv() {
  try {
    const sysInfo = wx.getSystemInfoSync ? wx.getSystemInfoSync() : {};
    const platform = sysInfo && sysInfo.platform ? String(sysInfo.platform) : 'unknown';
    return {
      runtime: 'mp',
      client: platform === 'devtools' ? 'devtools' : 'real-device',
      platform,
    };
  } catch (err) {
    return {
      runtime: 'mp',
      client: 'unknown',
      platform: 'unknown',
    };
  }
}

function isLocalhostLike(url) {
  return /^https?:\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?(?=\/|$)/i.test(String(url || ''));
}

const selectedEnv = ENV_CONFIG[CURRENT_ENV] || ENV_CONFIG.local;
const runtimeEnv = getRuntimeEnv();
const apiBaseUrl =
  runtimeEnv.client === 'devtools'
    ? selectedEnv.devtoolsApiBaseUrl
    : selectedEnv.deviceApiBaseUrl;

export default {
  ...selectedEnv,
  envName: CURRENT_ENV,
  apiBaseUrl,
  runtimeEnv,
  isDevtools: runtimeEnv.client === 'devtools',
  isLocalhostLike: isLocalhostLike(apiBaseUrl),
};
