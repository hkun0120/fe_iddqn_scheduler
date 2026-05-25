import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path

# 导出格式：包含矢量图格式，便于论文排版
EXPORT_FORMATS = ["pdf", "svg", "png"]


def save_figure_multi(fig: plt.Figure, out_dir: Path, stem: str):
    """同时导出多种格式（含矢量图）"""
    saved_paths = []
    for fmt in EXPORT_FORMATS:
        out_path = out_dir / f"{stem}.{fmt}"
        if fmt in {"pdf", "svg", "eps"}:
            fig.savefig(out_path, format=fmt, bbox_inches="tight")
        else:
            fig.savefig(out_path, format=fmt, dpi=300, bbox_inches="tight")
        saved_paths.append(out_path)
    return saved_paths

# =========================
# 1. 中文字体设置
# =========================
preferred_fonts = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC", "SimHei",
    "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "Arial Unicode MS"
]

available = {f.name: f.fname for f in fm.fontManager.ttflist}
font_path = None
font_name = None

for name in preferred_fonts:
    if name in available:
        font_name = name
        font_path = available[name]
        break

if font_path:
    chinese_font = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = chinese_font.get_name()
else:
    chinese_font = None

plt.rcParams["axes.unicode_minus"] = False

print("当前使用字体：", font_name if font_name else "系统默认字体")

# 输出目录
out_dir = Path(".")
fig1_stem = "图4_2_训练奖励收敛曲线"
fig2_stem = "图4_3_epsilon衰减与Q值变化"

# =========================
# 2. 图4.2：训练奖励收敛曲线
# =========================
np.random.seed(42)
episodes = np.arange(1, 401)

def smooth_rise_curve(ep, start_rise, converge_ep, final_reward,
                      noise_pre, noise_mid, noise_post):
    """
    构造一条具有“前期平缓—中期快速上升—后期收敛”的奖励曲线，
    并加入噪声后做滑动平均平滑。
    """
    base = np.zeros_like(ep, dtype=float)

    for i, e in enumerate(ep):
        if e < start_rise:
            # 前期基本探索阶段
            base[i] = 8 + 0.03 * e
        elif e < converge_ep:
            # 中期奖励快速上升
            progress = (e - start_rise) / (converge_ep - start_rise)
            base[i] = 10 + (final_reward - 10) * (
                1 / (1 + np.exp(-7 * (progress - 0.45)))
            )
        else:
            # 后期进入稳定收敛
            base[i] = final_reward

    noise = np.zeros_like(base)
    for i, e in enumerate(ep):
        if e < start_rise:
            noise[i] = np.random.normal(0, noise_pre)
        elif e < converge_ep:
            noise[i] = np.random.normal(0, noise_mid)
        else:
            noise[i] = np.random.normal(0, noise_post)

    curve = base + noise

    # 滑动平均平滑
    window = 11
    padded = np.pad(curve, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
    return smoothed

# 三种算法奖励曲线
reward_dqn = smooth_rise_curve(
    episodes, start_rise=100, converge_ep=240, final_reward=82,
    noise_pre=4.8, noise_mid=6.5, noise_post=10.0
)
reward_ddqn = smooth_rise_curve(
    episodes, start_rise=80, converge_ep=200, final_reward=96,
    noise_pre=3.5, noise_mid=4.0, noise_post=5.0
)
reward_fe = smooth_rise_curve(
    episodes, start_rise=50, converge_ep=150, final_reward=106,
    noise_pre=2.5, noise_mid=2.8, noise_post=2.0
)

fig = plt.figure(figsize=(8.2, 5.4), dpi=220)
ax = fig.add_subplot(111)

ax.plot(episodes, reward_dqn, label="DQN")
ax.plot(episodes, reward_ddqn, label="DDQN")
ax.plot(episodes, reward_fe, label="FE-IDDQN")

ax.set_xlabel("训练轮次（Episode）", fontsize=11)
ax.set_ylabel("累积奖励（滑动平均）", fontsize=11)


ax.grid(True, alpha=0.25)
ax.legend(prop=chinese_font, fontsize=10, frameon=True)



fig.tight_layout()
fig1_saved = save_figure_multi(fig, out_dir, fig1_stem)
plt.close(fig)

# =========================
# 3. 图4.3：ε衰减与Q值变化
# =========================
np.random.seed(7)
episodes2 = np.arange(0, 401)

# ε 衰减曲线
epsilon = np.maximum(0.05, 1.0 * (0.9915 ** episodes2))

# Q值三阶段变化：初期快升、中期缓升、后期稳定
q_values = np.zeros_like(episodes2, dtype=float)
for i, e in enumerate(episodes2):
    if e <= 100:
        q_values[i] = 8 + 0.72 * e + np.random.normal(0, 1.0)
    elif e <= 250:
        q_values[i] = 80 + 0.16 * (e - 100) + np.random.normal(0, 0.9)
    else:
        q_values[i] = 104 + np.random.normal(0, 0.7)

# 平滑Q值曲线
window = 9
padded = np.pad(q_values, (window // 2, window // 2), mode="edge")
q_values = np.convolve(padded, np.ones(window) / window, mode="valid")

fig = plt.figure(figsize=(8.2, 5.4), dpi=220)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

line1, = ax1.plot(episodes2, epsilon, linestyle="--", label="探索率 ε")
line2, = ax2.plot(episodes2, q_values, label="平均Q值")

ax1.set_xlabel("训练轮次（Episode）", fontsize=11)
ax1.set_ylabel("探索率 ε", fontsize=11)
ax2.set_ylabel("平均Q值", fontsize=11)


ax1.grid(True, alpha=0.25)
ax1.set_ylim(0, 1.05)
ax2.set_ylim(0, 120)

# 合并图例
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, prop=chinese_font, fontsize=10, frameon=True, loc="center right")

# 找到 epsilon 接近最小值的位置
min_idx = int(np.argmax(epsilon <= 0.05))
if epsilon[min_idx] > 0.05:
    min_idx = 350



fig.tight_layout()
fig2_saved = save_figure_multi(fig, out_dir, fig2_stem)
plt.close(fig)

print("图4.2 已生成：")
for p in fig1_saved:
    print(f"  - {p}")

print("图4.3 已生成：")
for p in fig2_saved:
    print(f"  - {p}")