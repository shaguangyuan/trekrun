Page({
  goUpload() {
    wx.navigateTo({ url: '/pages/sprint/upload/index' });
  },
  goHistory() {
    wx.navigateTo({ url: '/pages/sprint/history/index?athlete_id=A001' });
  },
});
