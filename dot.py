import pandas as pd
import matplotlib.pyplot as plt

# CSV 불러오기 (경로는 적절히 수정)
df = pd.read_csv("model_analysis_results.csv")

# accuracy 컬럼 자동 탐색
acc_col = [c for c in df.columns if "acc" in c.lower()][0]

# --- Accuracy vs Mult-Adds ---
plt.figure()
for sz, g in df.groupby("image_size"):
    plt.scatter(g["total_mult_adds_M"], g[acc_col], label=str(sz), marker='o')
plt.xscale("log")
plt.xlabel("Million Mult-Adds")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Mult-Adds (colored by input resolution)")
plt.legend(title="Input Resolution")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("accuracy_vs_multadds.png", dpi=300)  # 저장
plt.close()

# --- Accuracy vs Million Parameters ---
plt.figure()
for sz, g in df.groupby("image_size"):
    plt.scatter(g["total_params_M"], g[acc_col], label=str(sz), marker='o')
plt.xlabel("Million Parameters")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Million Parameters (colored by input resolution)")
plt.legend(title="Input Resolution")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("accuracy_vs_params.png", dpi=300)  # 저장
plt.close()

print("✅ 그래프가 PNG 파일로 저장되었습니다.")
