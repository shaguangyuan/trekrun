import { getHistory } from '../../../utils/api';

const BAR_MAX_HEIGHT = 140;

Page({
  data: {
    athleteId: '',
    history: [],
    savedCount: 0,
    loading: true,
    loadError: false,
  },

  onLoad(options) {
    const athleteId = (options.athlete_id || '').trim() || 'A001';
    this.setData({ athleteId, loading: true, loadError: false });
    getHistory(athleteId)
      .then((list) => {
        const safe = Array.isArray(list) ? list : [];
        const scores = safe
          .map((r) => (typeof r.tech_stability_score === 'number' ? r.tech_stability_score : null))
          .filter((n) => n != null);
        const maxScore = scores.length ? Math.max(...scores, 1) : 1;
        const history = safe.map((r) => {
          const sc = typeof r.tech_stability_score === 'number' ? r.tech_stability_score : null;
          return {
            video_id: r.video_id != null ? String(r.video_id) : '',
            created_at: r.created_at != null ? String(r.created_at) : '--',
            tech_stability_score: sc,
            barHeight:
              sc == null ? 8 : Math.round((sc / maxScore) * BAR_MAX_HEIGHT),
            shortDate:
              r.created_at && String(r.created_at).length >= 10
                ? String(r.created_at).slice(5, 10)
                : '--',
          };
        });
        this.setData({ loading: false, loadError: false, history, savedCount: history.length });
      })
      .catch(() => {
        wx.showToast({ title: '加载失败', icon: 'none' });
        this.setData({ loading: false, loadError: true, history: [] });
      });
  },

  goReport(e) {
    const videoId = e.currentTarget.dataset.id;
    if (!videoId) return;
    wx.navigateTo({
      url: `/pages/sprint/report/index?video_id=${encodeURIComponent(videoId)}`,
    });
  },

  goUpload() {
    wx.navigateTo({ url: '/pages/sprint/upload/index' });
  },
});
