import matplotlib.pyplot as plt
import json
import os


# グラフの最初のn個のデータを無視する
GRAPH_IGNORE_FIRST_N = 8

BASE_DIR = "/workspace/Checkpoint_files/original/SGD_test_model2/checkpoint-1450"
LOG_FILE_NAME = os.path.join(BASE_DIR, "trainer_state.json")
OUTPUT_FILE_NAME = os.path.join(BASE_DIR, "loss_graph.png")
# ファイルの読み込み
with open(LOG_FILE_NAME, 'r') as f:
    log_history = json.load(f)["log_history"]


# training lossのx軸とy軸のデータを取得
train_steps = []
train_losses = []
for log in log_history[GRAPH_IGNORE_FIRST_N:]:
    if 'loss' in log:
        train_steps.append(log['step'])
        train_losses.append(log['loss'])

# evaluation lossのx軸とy軸のデータを取得
eval_steps = []
eval_losses = []
for log in log_history[GRAPH_IGNORE_FIRST_N:]:
    if 'eval_loss' in log:
        eval_steps.append(log['step'])
        eval_losses.append(log['eval_loss'])

# グラフを作成
plt.plot(train_steps, train_losses, label='Training Loss')
plt.plot(eval_steps, eval_losses, label='Evaluation Loss')

# グラフのタイトルと軸ラベルを設定
plt.title("Training and Evaluation Loss")
plt.xlabel("Step")
plt.ylabel("Loss")

# 凡例を表示
plt.legend()

# グラフを表示
plt.show()

# グラフを画像として保存
plt.savefig(OUTPUT_FILE_NAME)
