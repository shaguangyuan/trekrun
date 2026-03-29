import { getAnalysisStatus } from '../../../utils/api';

const POLL_INTERVAL_MS = 2000;
// 长视频 + MediaPipe 逐帧可能超过 1–2 分钟；超时太短会误显示「分析失败」
const MAX_POLLS = 120; // ~4 minutes at 2s interval

const STATUS_MAP = {
  queued: {
    icon: '\u23f3',
    label: '\u6392\u961f\u4e2d',
    desc: '\u89c6\u9891\u5df2\u4e0a\u4f20\uff0c\u7b49\u5f85\u5904\u7406',
  },
  processing: {
    icon: '\u2699\ufe0f',
    label: '\u5206\u6790\u4e2d',
    desc: '\u6b63\u5728\u63d0\u53d6\u5173\u952e\u70b9\u5e76\u8ba1\u7b97\u6307\u6807\uff0c\u8bf7\u7a0d\u5019...',
  },
  done: {
    icon: '\u2705',
    label: '\u5206\u6790\u5b8c\u6210',
    desc: '5 \u9879\u8dd1\u59ff\u6307\u6807\u5df2\u751f\u6210',
  },
  failed: {
    icon: '\u274c',
    label: '\u5206\u6790\u5931\u8d25',
    desc: '\u89c6\u9891\u4e0d\u7b26\u5408\u5206\u6790\u8981\u6c42\uff0c\u8bf7\u68c0\u67e5\u540e\u91cd\u65b0\u4e0a\u4f20',
  },
};

Page({
  data: {
    videoId: '',
    status: 'queued',
    statusIcon: STATUS_MAP.queued.icon,
    statusLabel: STATUS_MAP.queued.label,
    statusDesc: STATUS_MAP.queued.desc,
    statusError: '',
    stageTag: '排队中',
    stepItems: [],
  },

  _pollTimer: null,
  _pollCount: 0,

  onLoad(options) {
    const videoId = (options.video_id || '').trim();
    if (!videoId) {
      this.setData({
        status: 'failed',
        statusIcon: STATUS_MAP.failed.icon,
        statusLabel: STATUS_MAP.failed.label,
        statusDesc: '缺少video_id，无法查询分析状态',
        statusError: 'missing video_id',
      });
      return;
    }
    this.setData({ videoId });
    this._applyStatus('queued');
    this._schedulePoll();
  },

  onUnload() {
    if (this._pollTimer) clearTimeout(this._pollTimer);
  },

  _schedulePoll() {
    if (this._pollCount >= MAX_POLLS) {
      const secs = (MAX_POLLS * POLL_INTERVAL_MS) / 1000;
      this._applyStatus(
        'failed',
        `等待超时（已轮询约 ${secs} 秒）。长视频分析可能仍在后台进行，可稍后在报告页重试或查看后端 uploads 目录下该任务的 .job.json。`,
      );
      console.error('[status] poll timeout', this.data.videoId, 'maxPolls=', MAX_POLLS);
      return;
    }
    this._pollCount += 1;
    getAnalysisStatus(this.data.videoId)
      .then((res) => {
        const status = res && res.status ? String(res.status) : 'failed';
        console.log('[status] poll', this.data.videoId, '=>', status);
        if (status === 'failed') {
          // 后端用纯 JSON dict 返回；仍多 key 兜底（hint 最短，部分环境会丢 error）
          const raw =
            (res && res.hint) ||
            (res && res.failure_reason) ||
            (res && res.error) ||
            '';
          const reason =
            raw !== '' && raw != null
              ? String(raw)
              : '(未拿到失败说明，请查后端 uploads/<video_id>.job.json)';
          console.error('[status] FAILED reason:', reason);
          console.error('[status] FAILED raw response:', res);
          this._applyStatus(status, reason);
        } else {
          const hint = (res && res.hint) || (res && res.failure_reason) || (res && res.error) || '';
          this._applyStatus(status, hint ? String(hint) : '');
        }
        if (status !== 'done' && status !== 'failed') {
          this._pollTimer = setTimeout(() => this._schedulePoll(), POLL_INTERVAL_MS);
        }
      })
      .catch((err) => {
        console.error('[status] poll failed:', err);
        this._applyStatus('failed', err && err.message ? err.message : '状态接口请求失败');
      });
  },

  _applyStatus(status, error = '') {
    const s = STATUS_MAP[status] || STATUS_MAP.failed;
    const stepItems = this._buildSteps(status);
    const stageTag =
      status === 'done' ? '已完成' : (status === 'failed' ? '失败' : (status === 'processing' ? '处理中' : '排队中'));
    this.setData({
      status,
      statusIcon: s.icon,
      statusLabel: s.label,
      statusDesc: status === 'failed' && error ? error : s.desc,
      statusError: status === 'failed' ? error : '',
      stageTag,
      stepItems,
    });
  },

  _buildSteps(status) {
    const isDone = status === 'done';
    const processing = status === 'processing';
    const failed = status === 'failed';
    return [
      { key: 'uploaded', title: '已上传视频', desc: '文件已进入分析队列', state: (isDone || processing || failed) ? 'done' : 'current' },
      { key: 'pose', title: '提取姿态关键点', desc: '识别人体骨架与核心关节', state: (isDone || processing || failed) ? 'done' : 'pending' },
      { key: 'segment', title: '筛选有效片段', desc: '合并短缺口并选择连续片段', state: (isDone || processing || failed) ? 'done' : 'pending' },
      { key: 'metrics', title: '计算跑姿指标', desc: '逐项计算并生成报告内容', state: isDone ? 'done' : (processing ? 'current' : (failed ? 'failed' : 'pending')) },
      { key: 'report', title: '生成分析报告', desc: failed ? '本次未成功生成报告' : '可进入报告页查看详情', state: isDone ? 'done' : (failed ? 'failed' : 'pending') },
    ];
  },

  goReport() {
    wx.navigateTo({
      url: `/pages/sprint/report/index?video_id=${this.data.videoId}`,
    });
  },

  goBack() {
    wx.navigateBack();
  },
});
