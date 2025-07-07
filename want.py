import matplotlib.pyplot as plt
import numpy as np
from style.genevive import genevive

# 데이터
categories = ["French\nFries", "Potato Chips", "Bacon", "Pizza", "Chili Dog"]
values = [600, 550, 520, 300, 250]
highlight = "Bacon"  # 강조할 데이터

# 색상 설정 (Bacon만 강조)
colors = ["#8c8c8c" if category != highlight else "#b5655c" for category in categories]


with plt.style.context(genevive):
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categories, values, width=0.6, zorder=2)

    # 제목 및 라벨
    ax.set_title("Calories per 100g", pad=10)
    ax.set_xticklabels(categories, rotation=0, ha="center", wrap=True)

    # y축 눈금 설정 및 스타일 조정
    ax.set_yticks(np.arange(0, 701, 100))
    ax.yaxis.grid(True, linewidth=0.6, zorder=3)

    plt.show()

    plt.savefig("fuck!!.png", dpi=300)
