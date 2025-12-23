def draw_pie_chart(danger, suspicious, total, figsize=(1, 1), fontsize=4):
    import matplotlib.pyplot as plt
    labels = ['위험', '의심', '정상']
    sizes = [danger, suspicious, max(total - danger - suspicious, 0)]
    colors = ['#e74c3c', '#f1c40f', '#2ecc71']
    fig, ax = plt.subplots(figsize=figsize)

    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        colors=colors,
        textprops={'fontsize': fontsize}  # 글자 크기 지정
    )
    ax.set_title("위험도 분포", fontsize=fontsize + 2)
    return fig

def draw_histogram(scores, figsize=(1, 1), fontsize=4):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(scores, bins=10, color='#3498db')
    ax.set_title("Hybrid 점수 히스토그램", fontsize=fontsize + 2)
    ax.set_xlabel("Hybrid 점수", fontsize=fontsize)
    ax.set_ylabel("빈도", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize - 1)  # 눈금 글자 크기 조절
    return fig
