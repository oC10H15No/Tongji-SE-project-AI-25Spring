# 基于 DQN 的贪吃蛇游戏

本项目利用深度强化学习中的 DQN 算法训练智能体，使其能够掌握并优化贪吃蛇游戏策略。

项目采用模块化设计，除了核心的训练脚本外，还提供了直观的可视化工具。用户可以进行训练，通过图形界面体验游戏等操作。

<img src="https://cloud.githubusercontent.com/assets/2750531/24808769/cc825424-1bc5-11e7-816f-7320f7bda2cf.gif" width="300px">


## 安装

1.  进入项目根目录。
2.  创建您的 Python 虚拟环境（可选，但推荐）：
    ```bash
    conda create -n snakeai
    conda activate snakeai
    ```
3.  安装所有 Python 依赖：
    ```bash
    make deps
    ```
    或者，如果您修改了 `requirements.txt`，可以手动执行：
    ```bash
    python3 -m pip install --upgrade -r requirements.txt
    ```

## 使用说明

### 1. 训练 DQN 代理

*   **使用 Makefile 默认配置训练**:
    ```bash
    make train
    ```
    这将使用 `snakeai/levels/10x10-blank.json` 关卡，默认训练 21000 轮。训练过程中，模型检查点 (checkpoints) 会定期保存为 `.h5` 文件（例如 `dqn-00003000.h5`）。训练完成后，最终模型将保存为 `dqn-final.h5`。
    训练日志（每轮的统计数据）会保存在项目根目录下的 `snake-env-YYYYMMDD-HHMMSS.csv` 文件中。

*   **自定义训练**:
    您可以直接运行 `train.py` 脚本并指定参数 (使用命令 ``train.py -h`` 查看帮助)：
    ```bash
    python train.py --level <关卡文件路径> --num-episodes <训练轮数> [--load-model <预训练模型路径.h5>] [--initial-episode <起始轮数>]
    ```
    参数说明:
    *   `--level`: (必需) 指定关卡定义的 JSON 文件路径，例如 `snakeai/levels/10x10-obstacles.json`。
    *   `--num-episodes`: (必需) 训练的总轮数。
    *   `--load-model`: (可选) 如果您想从一个已有的模型继续训练，请指定该模型的 `.h5` 文件路径。
    *   `--initial-episode`: (可选, 通常与 `--load-model` 配合使用) 指定当前训练从那一轮开始计数。例如，如果加载了一个训练了10000轮的模型，可以将此参数设为 `10000`。

    例如，从零开始训练一个使用障碍物关卡的模型，共 30000 轮：
    ```bash
    python train.py --level snakeai/levels/10x10-obstacles.json --num-episodes 30000
    ```

### 2. 使用训练好的模型游戏

*   **GUI 图形界面模式**:
    实时显示游戏过程。
    ```bash
    make play-gui
    ```
    默认使用 `dqn-final.h5` 模型，在 `10x10-blank.json` 关卡上运行 10 轮。

    自定义运行：
    ```bash
    python play.py --interface gui --agent dqn --model <模型路径.h5> --level <关卡文件路径> --num-episodes <回放轮数>
    ```

### 3. 可视化训练数据

项目提供了一个脚本 `plot_training_curves.py` 用于将训练过程中生成的 CSV 日志文件可视化。
1.  确保您的训练已经产生了一些 `snake-env-*.csv` 文件在项目根目录。
2.  运行绘图脚本：
    ```bash
    python plot_training_curves.py
    ```
3.  生成的图表（例如总奖励、吃掉的果子数、存活步数等随训练轮数的变化）会保存在 `training_plots` 文件夹中。

脚本会自动查找并合并所有 `snake-env*.csv` 文件的数据进行绘图。

## 项目结构

```bash
.
├── README.md                     # 项目介绍、安装、使用说明等
├── snake/                        # 核心代码目录：包含贪吃蛇游戏及DQN模型
│   ├── train.py                  # 脚本：训练DQN智能体
│   ├── play.py                   # 脚本：加载已训练模型并进行游戏
│   ├── dqn-final.h5              # 默认训练完成的模型文件
│   ├── plot_training_curves.py   # 脚本：可视化训练数据
│   ├── Makefile                  # 用于简化编译、训练和运行等任务
│   ├── requirements.txt          # Python依赖包列表
│   ├── snakeai/                  # 核心代码
│   │   ├── agent/                # DQN智能体实现
│   │   ├── gameplay/             # 游戏逻辑和环境
│   │   ├── levels/               # 游戏关卡配置文件 (*.json)
│   │   └── ...
│   ├── saves/                    # 目录：存放训练保存的模型检查点 (.h5)
│   ├── training_plots/           # 目录：存放生成的训练曲线图
│   └── snake-env-*.csv           # 文件：训练日志数据 (CSV格式, 位于 snake/ 目录下)
└── hw3_2351050_杨瑞晨.pdf         # 演示文稿
```

## 预训练模型

本次项目中训练过程中生成的模型文件（`.h5`）存放于 `saves/` 目录下。
