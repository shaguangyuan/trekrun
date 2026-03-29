# 核心指标定义

## 1. 步频 step_rate
定义：有效分析区间内的步数 / 时间
输出：steps per second

## 2. 躯干前倾均值 trunk_lean_mean
定义：肩-髋连线相对垂线的夹角均值
输出：degree

## 3. 摆臂波动 arm_swing_variability
定义：肩-肘-腕轨迹振幅和周期波动的组合指标
输出：normalized score

## 4. 左右节律差 left_right_timing_diff
定义：左右步时或关键相位间隔差异
输出：percentage

## 5. 技术稳定性分数 tech_stability_score
定义：基于步频变化、躯干波动、摆臂波动、左右差异综合加权的分数
输出：0-100

# 注意
- 第一版只做训练反馈，不输出医疗或伤病结论
- 第一版优先做个体前后比较，不做绝对“标准跑姿”判断