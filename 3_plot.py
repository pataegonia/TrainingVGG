import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_three_results(csv_a, csv_b, csv_c, save_path=None):
    """
    Plots training loss, validation loss, and validation accuracy
    from exactly three CSV files.

    CSV는 각 파일에 'epoch', 'train_loss', 'val_loss', 'val_accuracy' 열을 포함해야 함.
    x축은 epoch 대신 모든 러닝을 0~1 구간으로 정규화해서 표시함.
    """
    csv_files = [csv_a, csv_b, csv_c]
    labels = [os.path.basename(p).rsplit(".", 1)[0] for p in csv_files]

    # 파일 존재 확인 및 로드
    dataframes = []
    for p in csv_files:
        if not os.path.exists(p):
            print(f"Error: '{p}' 파일을 찾을 수 없습니다.")
            return
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Error: '{p}' 읽기 실패 – {e}")
            return

        required = {"epoch", "train_loss", "val_loss", "val_accuracy"}
        if not required.issubset(df.columns):
            print(f"Error: '{p}'에 필요한 열 {required} 이(가) 모두 존재해야 합니다.")
            return
        dataframes.append(df)

    # 3개 서브플롯
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle("Comparison of Training Metrics (3 runs, normalized x-axis)", fontsize=20)

    # 공통 그리기 함수
    def plot_metric(ax, col, title, ylabel):
        for label, df in zip(labels, dataframes):
            x = np.linspace(0, 1, len(df[col]))  # 0~1 정규화
            ax.plot(x, df[col], label=label)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Training Progress (0~1)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True)
        ax.legend()

    # Plot들
    plot_metric(axes[0], "train_loss", "Training Loss over Progress", "Loss")
    plot_metric(axes[1], "val_loss", "Validation Loss over Progress", "Loss")
    plot_metric(axes[2], "val_accuracy", "Validation Accuracy over Progress", "Accuracy (%)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path)
        print(f"Combined plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # 예시: 세 파일을 지정하여 저장
    plot_three_results(
        "SENet_16.csv",
        "CSV/SENet_8.csv",
        "SENet.csv",
        save_path="VSVGG.png"
    )
