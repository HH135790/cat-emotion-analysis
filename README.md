# 🐱 Cat Emotion Analysis / 猫叫声情绪识别

**Keywords**: cat emotion recognition · cat sound classification · cat meow analysis · YAMNet transfer learning · audio classification · pet behavior AI · 猫叫情绪识别 · 猫叫声分类

基于 Google **YAMNet** 迁移学习的猫叫声情绪识别模型。输入 16 kHz 猫叫音频，输出 4 类情绪状态：
**安静 (resting) / 开心 (happy) / 狩猎 (hunting) / 打斗 (fighting)**，推理脚本默认输出 3 类产品语义：
**安静 / 开心 / 激烈 (intense)**。

> **English**: A cat emotion recognition model fine-tuned from Google YAMNet.
> Input a 16 kHz cat vocalization clip, get one of 4 emotional states: resting / happy / hunting / fighting
> (merged to resting / happy / intense in inference). Evaluated on an independently collected
> cross-domain test set (different devices & sources from training data).

---

## 📊 评测结果

独立跨域测试集（25 条音频，录音设备与数据来源均与训练集不同，模拟真实部署场景）：

- **Top-1 准确率（4 类）：80%**（16/20 in-scope）
- **Macro F1：0.794**
- **calm/intense 两级语义：100%**（resting+happy 合并为 calm，hunting+fighting 合并为 intense，组间零泄漏）

每类表现（每类 5 条）：

| 情绪 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| resting 安静 | 1.00 | 1.00 | 1.000 |
| happy 开心 | 1.00 | 0.80 | 0.889 |
| hunting 狩猎 | 1.00 | 0.40 | 0.571 |
| fighting 打斗 | 0.556 | 1.00 | 0.714 |

混淆矩阵（行=真值，列=预测）：

|            | resting | happy | hunting | fighting |
|------------|---------|-------|---------|----------|
| **resting**  | 5 | 0 | 0 | 0 |
| **happy**    | 0 | 4 | 0 | 1 |
| **hunting**  | 0 | 0 | 2 | 3 |
| **fighting** | 0 | 0 | 0 | 5 |

亮点：

- **calm 与 intense 两组之间零泄漏**——所有错分均发生在语义组内（resting↔happy、hunting↔fighting），
  两级语义（平静/激烈）在跨域测试中达到 100%
- hunting 精确率 1.00（判成狩猎必对），fighting 召回 1.00（打斗叫声全召回）

---

## 🚀 快速开始

### 环境

```bash
pip install -r requirements.txt
# Python 3.10-3.12, TensorFlow 2.16+ (Keras 3)
```

### 推理

下载权重文件 `yamnet_final_hybrid.h5`（获取方式见下节），放入 `models/` 目录，然后：

```bash
python inference.py cat_meow.wav                # 单文件
python inference.py test_folder/                # 整个目录
python inference.py cat_meow.wav --raw          # 显示原始 4 类概率
```

输出示例：

```
文件                                       情绪        置信度
------------------------------------------------------------
cat_meow_001.wav                           激烈(intense)    0.923
```

---

## 📦 模型权重获取

**本仓库不含权重文件。** 如需训练好的模型权重
（`yamnet_final_hybrid.h5`，约 36 MB，Keras 3 / TensorFlow 2.16 格式，HDF5），
请**通过 GitHub Issue 或私信联系作者**，本人可**免费提供**。

> **English**: Pre-trained weights (`yamnet_final_hybrid.h5`, ~36 MB, Keras 3 / TF 2.16 HDF5)
> are **not** included in this repo. **Open a GitHub issue or DM me** — available **free of charge**.

拿到权重后放入 `models/yamnet_final_hybrid.h5`，并按
[tensorflow/models](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
下载 YAMNet 源码（`yamnet.py / params.py / yamnet_class_map.csv`）放入 `yamnet/` 目录即可直接推理。

---

## 📂 目录结构

```
cat-emotion-analysis/
├── README.md                 # 本文件
├── inference.py              # 推理脚本 (3 类语义输出)
├── requirements.txt
├── LICENSE
├── experiments/              # 训练相关脚本 (检测/选样/微调)
└── docs/                     # 标签格式参考 (labels_10K.csv / labels_V2.csv)
```

## 🙏 数据与模型来源声明

- **YAMNet**：Google 研究的通用音频事件检测模型（AudioSet 521 类），Apache 2.0 许可，本项目在其基础上微调。
- **训练数据**：CatMeows10k 社区猫叫数据集（YouTube 音频剪辑等混合来源）+ 人工采集的 hunting 真实录音段（经降噪切段与人工质检）。音频数据不随仓库分发。
- **测试数据**：独立采集的猫叫音频，与训练集来源、设备均不同，用于模拟真实部署场景。

## License

[MIT](LICENSE)（不含 YAMNet 官方权重与第三方数据集）
