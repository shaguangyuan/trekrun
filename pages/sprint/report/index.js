import { getAIAnalysis, getReport, saveReport } from '../../../utils/api';

const METRIC_KEYS = [
  'step_rate',
  'trunk_lean_mean',
  'arm_swing_variability',
  'left_right_timing_diff',
];

const METRIC_META = {
  step_rate: { label: '步频', unit: '步/秒', decimals: 2 },
  trunk_lean_mean: { label: '躯干前倾均值', unit: '°', decimals: 1 },
  arm_swing_variability: { label: '摆臂波动', unit: '(归一化)', decimals: 2 },
  left_right_timing_diff: { label: '左右节律差', unit: '%', decimals: 1 },
};

const FEATURE_GROUP_META = {
  pose_geometry: '关节角指标',
  temporal: '节律指标',
  symmetry: '对称性指标',
  qc: '质量控制指标',
};

function numOrNull(v) {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function fmt(v, decimals) {
  if (v === null || v === undefined || Number.isNaN(v)) return '--';
  return Number(v).toFixed(decimals);
}

function fmtDelta(v, decimals) {
  if (v === null || v === undefined || Number.isNaN(v)) return '--';
  const n = Number(v);
  const s = n >= 0 ? '+' : '';
  return s + n.toFixed(decimals);
}

/** Normalize API payload; keeps extra fields (video_info, coach_summary, warnings). */
function normalizeReport(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const m = raw.metrics || {};
  const c = raw.comparison || {};
  const metrics = {};
  const comparison = {};
  const allKeys = [...METRIC_KEYS, 'tech_stability_score'];
  allKeys.forEach((k) => {
    metrics[k] = numOrNull(m[k]);
    const d = numOrNull(c[k]);
    comparison[k] = d === null ? 0 : d;
  });
  return {
    ...raw,
    athlete_id: raw.athlete_id != null ? String(raw.athlete_id) : '--',
    created_at: raw.created_at != null ? String(raw.created_at) : '--',
    metrics,
    comparison,
  };
}

Page({
  data: {
    report: null,
    metricList: [],
    scoreDelta: 0,
    scorePositive: true,
    loading: true,
    loadError: false,
    loadErrorText: '报告未就绪',
    saving: false,
    savedToHistory: false,
    nl: null,
    nlProcessSteps: [],
    nlWarnings: [],
    hasNlWarnings: false,
    metricExplainList: [],
    showMoreIndicators: false,
    featureGroupList: [],
    aiLoading: false,
    ai: null,
    aiIsFallback: false,
    aiEvidenceList: [],
    aiTechFindingsList: [],
    aiDataReliabilityList: [],
    aiOtherRiskFlagsList: [],
    aiSuggestionsList: [],
    showAIAnalysis: true,
  },

  onLoad(options) {
    const videoId = (options.video_id || '').trim();
    if (!videoId) {
      this.setData({ loading: false, loadError: true });
      return;
    }
    console.log('[report] request video_id =', videoId);
    getReport(videoId)
      .then((raw) => {
        console.log('[report] response keys =', raw ? Object.keys(raw) : []);
        const report = normalizeReport(raw);
        if (!report) {
          this.setData({ loading: false, loadError: true, loadErrorText: '报告数据为空', report: null });
          return;
        }
        const metricList = METRIC_KEYS.map((key) => {
          const meta = METRIC_META[key];
          const value = report.metrics[key];
          const delta = report.comparison[key];
          return {
            key,
            label: meta.label,
            unit: meta.unit,
            value: fmt(value, meta.decimals),
            delta: fmtDelta(delta, meta.decimals),
            deltaPositive: delta >= 0,
          };
        });
        const scoreDelta = report.comparison.tech_stability_score;
        const ai = report.ai_analysis || null;
        this.setData({
          loading: false,
          loadError: false,
          report,
          metricList,
          scoreDelta,
          scorePositive: scoreDelta >= 0,
          savedToHistory: !!report.saved_to_history,
          nl: report.natural_language || null,
          nlProcessSteps:
            report &&
            report.natural_language &&
            Array.isArray(report.natural_language.process_steps)
              ? report.natural_language.process_steps
              : [],
          nlWarnings:
            report &&
            report.natural_language &&
            Array.isArray(report.natural_language.warnings)
              ? report.natural_language.warnings
              : [],
          hasNlWarnings:
            !!(
              report &&
              report.natural_language &&
              Array.isArray(report.natural_language.warnings) &&
              report.natural_language.warnings.length
            ),
          metricExplainList: Array.isArray(report.metrics_detail) ? report.metrics_detail : [],
          featureGroupList: this._buildFeatureGroups(report.feature_groups || {}),
          ...this._buildAIData(ai),
        });
        if (ai && ai.is_fallback) {
          console.log('[report] report loaded with fallback AI; user can click to generate real AI');
        }
      })
      .catch((err) => {
        console.error('[report] failed:', err);
        const message = err && err.message ? String(err.message) : '';
        const notReady = message.includes('not complete') || message.includes('not ready') || message.includes('Unknown video_id') || message.includes('Metrics not found');
        const text = notReady ? '报告未就绪' : (message || '报告加载失败');
        wx.showToast({ title: text, icon: 'none' });
        this.setData({ loading: false, loadError: true, loadErrorText: text, report: null });
      });
  },

  goHistory() {
    const athleteId = this.data.report && this.data.report.athlete_id;
    if (!athleteId || athleteId === '--') {
      wx.showToast({ title: '缺少运动员信息', icon: 'none' });
      return;
    }
    wx.navigateTo({
      url: `/pages/sprint/history/index?athlete_id=${encodeURIComponent(athleteId)}`,
    });
  },

  onSaveToHistory() {
    if (this.data.saving || this.data.savedToHistory) return;
    const report = this.data.report;
    const videoId = report && report.video_id;
    if (!videoId) {
      wx.showToast({ title: '缺少 video_id', icon: 'none' });
      return;
    }
    this.setData({ saving: true });
    saveReport(videoId)
      .then((res) => {
        const ok = !!(res && res.saved_to_history);
        this.setData({
          saving: false,
          savedToHistory: ok,
          report: {
            ...(this.data.report || {}),
            saved_to_history: ok,
            saved_at: (res && res.saved_at) || (this.data.report && this.data.report.saved_at) || null,
          },
        });
        wx.showToast({ title: ok ? '已保存到历史' : '已处理', icon: 'none' });
      })
      .catch((err) => {
        this.setData({ saving: false });
        wx.showToast({ title: err && err.message ? err.message : '保存失败', icon: 'none' });
      });
  },

  toggleMoreIndicators() {
    this.setData({ showMoreIndicators: !this.data.showMoreIndicators });
  },

  toggleAIAnalysis() {
    this.setData({ showAIAnalysis: !this.data.showAIAnalysis });
  },

  generateAIAnalysis() {
    const videoId = this.data.report && this.data.report.video_id;
    if (!videoId || this.data.aiLoading) return;
    this.setData({ aiLoading: true });
    wx.showToast({ title: 'AI 分析生成中，请稍候...', icon: 'loading', duration: 180000 });
    getAIAnalysis(videoId, true)
      .then((ai) => {
        wx.hideToast();
        this.setData({ aiLoading: false, ...this._buildAIData(ai) });
        wx.showToast({ title: 'AI 分析已生成', icon: 'success' });
      })
      .catch((err) => {
        wx.hideToast();
        console.error('[report] AI generation failed:', err);
        this.setData({ aiLoading: false });
        wx.showToast({ title: 'AI 分析生成失败，请重试', icon: 'none' });
      });
  },

  refreshAIAnalysis() {
    this.generateAIAnalysis();
  },

  _buildAIData(ai) {
    const isFallback = !!(ai && ai.is_fallback);
    const findings = ai && Array.isArray(ai.key_findings) ? ai.key_findings : [];
    const riskFlags = ai && Array.isArray(ai.risk_flags) ? ai.risk_flags : [];
    const suggestions = ai && Array.isArray(ai.suggestions) ? ai.suggestions : [];

    const technicalFindings = [];
    const dataReliability = [];

    findings.forEach((item) => {
      if (this._isDataQualityFinding(item && item.title, item && item.description)) {
        dataReliability.push(item);
      } else {
        technicalFindings.push(item);
      }
    });

    riskFlags.forEach((item) => {
      const isDataType = item && item.type === 'data_quality';
      const text = item && item.message ? String(item.message) : '';
      if (isDataType || this._isDataQualityFinding(text, text)) {
        dataReliability.push({
          title: '数据可靠性提示',
          description: text,
          confidence: 'medium',
          related_metrics: [],
        });
      }
    });

    const trainingSuggestions = [];
    const qualitySuggestions = [];
    suggestions.forEach((s) => {
      const text = String(s || '');
      if (this._isShootingSuggestion(text)) {
        qualitySuggestions.push(text);
      } else {
        trainingSuggestions.push(text);
      }
    });
    const orderedSuggestions = [...trainingSuggestions, ...qualitySuggestions.slice(0, 1)];

    return {
      ai: ai || null,
      aiIsFallback: isFallback,
      aiEvidenceList: ai && Array.isArray(ai.evidence_trace) ? ai.evidence_trace : [],
      aiTechFindingsList: technicalFindings,
      aiDataReliabilityList: dataReliability.slice(0, 1),
      aiOtherRiskFlagsList: riskFlags.filter((item) => item && item.type !== 'data_quality'),
      aiSuggestionsList: orderedSuggestions,
    };
  },

  _isDataQualityFinding(title, description) {
    const text = `${title || ''} ${description || ''}`;
    const keywords = ['抖动', '拍摄', '数据质量', '污染', '指标矛盾', '算法', '复核', '可靠性'];
    return keywords.some((k) => text.includes(k));
  },

  _isShootingSuggestion(text) {
    const keywords = ['拍摄', '机位', '三脚架', '稳定器', '光线', '视频', '复拍'];
    return keywords.some((k) => text.includes(k));
  },

  _buildFeatureGroups(groups) {
    const out = [];
    Object.keys(FEATURE_GROUP_META).forEach((key) => {
      const raw = groups[key];
      if (!raw || typeof raw !== 'object') return;
      const items = [];
      Object.keys(raw).forEach((k) => {
        const v = raw[k];
        if (v !== null && typeof v !== 'object') {
          items.push({
            key: k,
            label: k,
            value: typeof v === 'number' ? Number(v).toFixed(3) : String(v),
            unit: '',
            confidence: '--',
            explanation: '由关键点时序与几何关系派生得到。',
          });
        }
      });
      out.push({ key, title: FEATURE_GROUP_META[key], items });
    });
    return out;
  },
});
