# 大语言模型部署实验

本项目是 *人工智能导论* 课程的作业，旨在通过魔搭（ModelScope）平台，部署并测试当前主流的开源大语言模型，并对它们的表现进行横向对比分析。

*   姓名：杨瑞晨
*   学号：2351050
*   学院：计算机科学与技术学院
*   专业：软件工程

## 项目概述

本项目主要完成了以下工作：
1.  在魔搭（ModelScope）平台的DSW (Data Science Workshop)环境中配置了必要的Python依赖。
2.  从魔搭平台获取了选定的开源大语言模型（如通义千问Qwen-7B-Chat、智谱ChatGLM3-6B）的代码和权重文件到DSW环境的本地。
3.  在DSW的Jupyter Notebook中成功加载并部署了这些模型。
4.  针对一系列预设问题（包括复杂语境理解、逻辑推理、知识问答、代码生成、多语言翻译等），对部署的模型进行了测试。
5.  记录了模型的回答，并基于测试结果从多个维度（如逻辑推理能力、歧义理解、知识广度、文本生成能力等）对模型进行了横向对比分析。
6.  完成了详细的实验报告。

## 文件结构

```
.
├── hw4_2351050_杨瑞晨.pdf        # 实验报告最终版
├── run.ipynb                    # 主要的Jupyter Notebook文件，包含模型部署和测试代码
├── fig/                         # 存放报告中引用的图片
│   ├── qwen.png
│   ├── chatglm.png
│   └── ...
└── README.md                    # 本说明文件
```
*   **`hw4_2351050_杨瑞晨.pdf`**: 详细的实验报告，包含了实验目的、环境配置、部署过程截图、问答测试结果截图以及详细的横向对比分析和总结。
*   **`run.ipynb`**: Jupyter Notebook 文件，其中包含了：
    *   环境依赖的安装命令（注释形式或实际执行）。
    *   加载和部署所选大语言模型的Python代码。
    *   用于测试模型的各个问题及其对应的模型调用代码和输出结果。


## 使用模型

本项目主要测试和对比了以下模型：
*   **通义千问 Qwen-7B-Chat**
*   **智谱 ChatGLM3-6B**

## 主要测试维度

在横向对比分析中，我们主要关注了模型在以下方面的表现：
*   逻辑推理与歧义理解能力
*   知识广度与事实准确性
*   创造性与文本生成能力（包括代码生成）
*   多语言能力

## 如何查看

1.  **实验报告**: 请直接打开 `hw4_2351050_杨瑞晨.pdf` 文件阅读详细的实验过程、结果和分析。
2.  **Jupyter Notebook**:
    *   可以下载 `run.ipynb` 文件。
    *   建议在支持Jupyter Notebook的环境中打开（如本地Jupyter Lab/Notebook, VS Code, Google Colab等）。
    *   Notebook中包含了实际运行的代码和输出，可以直接查看。

## 项目公开链接 

*   [https://www.modelscope.cn/models/C10H15N/C10H15N/summary](https://www.modelscope.cn/models/C10H15N/C10H15N/summary)

*   [https://github.com/oC10H15No/Tongji-SE-project-AI-25Spring](https://www.modelscope.cn/models/THUDM/chatglm3-6b)
    