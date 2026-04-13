# Predicting Customer Ratings of Beauty Products Based on Amazon Reviews
# 基于亚马逊评论预测美妆产品客户评分

**Author / 作者:** Yanran Qiu (yanranq@uchicago.edu)  
**Program / 项目:** MACSS, University of Chicago  
**Course / 课程:** MACS 30100

---

## Table of Contents / 目录

- [English](#english)
- [中文](#中文)

---

## English

### Project Overview

This project applies supervised and unsupervised machine learning to Amazon customer reviews of beauty products (2023) to: (1) predict whether a review corresponds to a high or low product rating, and (2) discover the latent dimensions customers emphasize when writing reviews.

A **[video presentation](https://drive.google.com/file/d/13UVpUAcbJJElUE4-2sxq4pm5DOtbvrPA/view?usp=sharing)** and a **[Google Colab notebook](https://colab.research.google.com/drive/1GZKTPo9s0nPxvpuaIPkwR6ObgGZcexTO?authuser=1)** are available for this project.

### Research Questions

**RQ1 (Supervised):** Can customer review text predict whether a review corresponds to a high or low product rating?

**RQ2 (Unsupervised):** What are the distinct dimensions that customers emphasize when reviewing beauty products?

**RQ3 (Applied):** How are individual products rated on each dimension based on the predicted rating and discovered topics?

### Repository Structure

```
├── data/
│   ├── Product Review_beauty2023_raw.xlsx        # Raw extracted dataset (14,405 reviews)
│   ├── Product Review_beauty2023_cleaned.xlsx    # Cleaned & preprocessed dataset
│   ├── Product Review_beauty2023_balanced.xlsx   # Balanced dataset after undersampling
│   └── Product Review_beauty2023_topic.xlsx      # Dataset with BERTopic topic assignments
├── amz_rating_prediction_qyr_final.ipynb         # Main analysis notebook
├── presentation_slides.pdf                       # Presentation slides (PDF)
├── presentation_slides.pptx                      # Presentation slides (PPTX)
└── README.md 
```

### Data

**Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (UC San Diego), `All_Beauty` category.

| Stage | Description | Size |
|---|---|---|
| Raw | Reviews posted in 2023, merged with product metadata | 14,405 samples, 7 features |
| Cleaned | Removed missing values, duplicates, unverified purchases | 12,647 samples |
| Balanced | Random undersampling to equalize class distribution | 8,048 samples |

**Raw features:**

`rating` (1–5): numerical, the corresponding rating of the product from the review (standard deviation: 1.562, mean: 3.872, median: 5)

`title`: textual, title of the review

`text`: textual, the review text

`product_title`: textual, the name of the product, 5397 unique products

`parent_asin`: textual, the primary identifier code for the product, representing the core product listing

`verified_purchase`: categorical, whether the purchase of the review is verified

`timestamp`: the date and time at which the review was posted, from Jan 1, 2023 to September 9, 2023

**Target variable:** `cat_rating` — binarized from `rating` (1–3 → `low`; 4–5 → `high`)

### Methodology

#### Supervised Learning — Rating Classification

Text preprocessing steps:
- Expand contractions, remove HTML tags, URLs, punctuation, non-alphabetic characters, and repeated letters
- Merge review title and body
- TF-IDF vectorization (4,101 features)

Models trained with 70/30 train-test split and GridSearchCV tuning:

| Model | F1-Score | AUC | Recall (Low) | Recall (High) |
|---|---|---|---|---|
| **Logistic Regression** | **0.87** | **0.93** | 0.89 | 0.85 |
| Random Forest | 0.86 | 0.93 | 0.89 | 0.83 |
| Decision Tree | 0.78 | 0.78 | 0.80 | 0.75 |

**Best model:** Logistic Regression (L2 penalty, C=1, lbfgs solver). Top predictive features: *great, love, amazing, perfect, best*.

#### Unsupervised Learning — Topic Modeling with BERTopic

BERTopic was chosen because the dataset consists of short, informal product reviews for which contextual sentence embeddings are better suited.

Configuration: `CountVectorizer` + `UMAP` (n_neighbors=15) + `HDBSCAN` (min_cluster_size=30), reduced to 15 topics.

**Result:** 14 coherent topics (coherence score: 0.47), of which 5 represent product review:

    0: "product_skin_eye_face",
    1: "dimension_size_quality_value",
    2: "product_hair_care_wigs",
    3: "dimension_fragrance_scent",
    4: "dimension_packaging_bottle_pump",
    5: "product_nail_products",
    6: "product_hair_tools_clips_shavers",
    7: "dimension_color_shade",
    8: "product_brushes_bristles",
    9: "product_glue_tape",
    10: "product_dental_tools",
    11: "dimension_durability",
    12: "product_waxing_hair_removal",
    13: "product_dryer_diffuser_fit"

#### Product-Level Dimension Ratings (RQ3)

By combining topic assignments with predicted ratings, individual products can be scored on each review dimension (scale: 1–5 pt):

| Dimension | Grucioso Gel Nail Polish Set | Sekaler Heated Eyelash Curler |
|---|---|---|
| Size / Quality / Value | 2.59 | 2.67 |
| Packaging | 4.39 | — |
| Color | 4.27 | — |
| Durability | — | 1.11 |
| **Predicted Rating (overall)** | **3.76** | **3.30** |

### Key Findings

- Review text is a strong predictor of high/low ratings; Logistic Regression with TF-IDF achieves F1=0.87 and AUC=0.93.
- Regularized linear models (LR) and ensemble methods (RF) both outperform single decision trees for this task.
- A key error pattern: classifiers struggle with reviews that mix positive language with overall negative sentiment, as word-level features cannot capture contextual nuance.
- BERTopic identifies 5 customer evaluation dimensions in beauty reviews: size/quality/value, scent, packaging, color, and durability.
- Combining predicted ratings with topic assignments enables granular, dimension-level product performance profiling.

### Dependencies

```
numpy, pandas, scikit-learn, matplotlib, seaborn,
beautifulsoup4, tqdm, bertopic, umap-learn, hdbscan
```

---

## 中文

### 项目简介

本项目将监督学习与无监督学习方法应用于亚马逊美妆产品的用户评论数据（2023年），主要实现两个目标：（1）预测某条评论对应的产品评分是高还是低；（2）挖掘顾客撰写评论时所关注的潜在评价维度。

本项目提供 **[视频演示](https://drive.google.com/file/d/13UVpUAcbJJElUE4-2sxq4pm5DOtbvrPA/view?usp=sharing)** 及 **[Google Colab 笔记本](https://colab.research.google.com/drive/1GZKTPo9s0nPxvpuaIPkwR6ObgGZcexTO?authuser=1)** 供参考。

### 研究问题

**RQ1（监督学习）：** 顾客评论文本能否预测该评论对应的产品评分高低？

**RQ2（无监督学习）：** 顾客在评价美妆产品时，有哪些典型的评价维度？

**RQ3（应用）：** 基于预测评分和话题分配，如何对单个产品在各维度上进行评分？

### Repo结构

```
├── data/
│   ├── Product Review_beauty2023_raw.xlsx        # 原始提取数据集（14,405条评论）
│   ├── Product Review_beauty2023_cleaned.xlsx    # 清洗及预处理后的数据集
│   ├── Product Review_beauty2023_balanced.xlsx   # 下采样后的平衡数据集
│   └── Product Review_beauty2023_topic.xlsx      # 包含BERTopic主题分配的数据集
├── amz_rating_prediction_qyr_final.ipynb         # 主要分析笔记本
├── presentation_slides.pdf                       # 演示幻灯片（PDF格式）
├── presentation_slides.pptx                      # 演示幻灯片（PPTX格式）
└── README.md                                     # 项目说明文档
```

### 数据说明

**数据来源：** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)（加州大学圣地亚哥分校），使用 `All_Beauty` 类别子集。

| 处理阶段 | 说明 | 规模 |
|---|---|---|
| 原始数据 | 提取2023年评论，与产品元数据合并 | 14,405条，7个特征 |
| 清洗后 | 删除缺失值、重复项、未验证购买 | 12,647条 |
| 平衡后 | 随机欠采样，使两类别数量一致 | 8,048条（高评分4,024 / 低评分4,024） |

**特征变量：** 

原始特征：
`rating`（1-5分）：数值型，评论对应的产品评分（标准差：1.562，均值：3.872，中位数：5）

`title`：文本型，评论标题

`text`：文本型，评论正文

`product_title`：文本型，产品名称，共5397个独特产品

`parent_asin`：文本型，产品的主要标识代码，代表核心产品列表

`verified_purchase`：分类变量，评论对应的购买是否经过验证

`timestamp`：评论发布的日期和时间，范围为2023年1月1日至2023年9月9日

**目标变量：** `cat_rating`——将 `rating` 二值化（1–3 → `low` 低评分；4–5 → `high` 高评分）

### 方法

#### 监督学习——评分分类

文本预处理步骤：
- 展开缩写词、去除HTML标签、URL、标点、非字母字符及重复字母
- 合并评论标题与正文为统一 `review` 字段
- TF-IDF向量化（共4,101个特征，`ngram_range=(1,1)`）

三个模型均采用70/30训练测试划分，并通过GridSearchCV进行超参数调优：

| 模型 | F1分数 | AUC | 召回率（低评分） | 召回率（高评分） |
|---|---|---|---|---|
| **逻辑回归** | **0.87** | **0.93** | 0.89 | 0.85 |
| 随机森林 | 0.86 | 0.93 | 0.89 | 0.83 |
| 决策树 | 0.78 | 0.78 | 0.80 | 0.75 |

**最优模型：** 逻辑回归（L2正则化，C=1，lbfgs求解器）。最重要预测词汇：*great、love、amazing、perfect、best*。

#### 无监督学习——BERTopic主题建模

相较于传统LDA模型，BERTopic更适合本项目中短文本、非正式的产品评论场景，因为它利用上下文句子嵌入来捕捉语义相似性。

模型配置：`CountVectorizer` + `UMAP`（n_neighbors=15）+ `HDBSCAN`（min_cluster_size=30），目标话题数设为15。

**结果：** 识别出14个有意义的话题（话题连贯性得分：0.47），含5个产品评价核心维度：

    0: "product_skin_eye_face"（产品_皮肤_眼部_面部）,
    1: "dimension_size_quality_value"（维度_尺寸_质量_价值）,
    2: "product_hair_care_wigs"（产品_护发_假发）,
    3: "dimension_fragrance_scent"（维度_香气_气味）,
    4: "dimension_packaging_bottle_pump"（维度_包装_瓶身_泵头）,
    5: "product_nail_products"（产品_美甲产品）,
    6: "product_hair_tools_clips_shavers"（产品_美发工具_发夹_剃须刀）,
    7: "dimension_color_shade"（维度_颜色_色号）,
    8: "product_brushes_bristles"（产品_刷子_刷毛）,
    9: "product_glue_tape"（产品_胶水_胶带）,
    10: "product_dental_tools"（产品_牙科工具）,
    11: "dimension_durability"（维度_耐用性）,
    12: "product_waxing_hair_removal"（产品_脱毛蜡_脱毛）,
    13: "product_dryer_diffuser_fit"（产品_吹风机_扩散器_适配性）

#### 产品维度评分（RQ3）

将话题分配与预测评分结合，即可对单个产品在各评价维度上进行精细化评分（满分5分）：

| 评价维度 | Grucioso 光疗甲油套装 | Sekaler 加热睫毛夹 |
|---|---|---|
| 尺寸/质量/性价比 | 2.59 | 2.67 |
| 包装 | 4.39 | — |
| 颜色 | 4.27 | — |
| 耐用性 | — | 1.11 |
| **综合预测评分** | **3.76** | **3.30** |

### 主要结论

- 评论文本对评分高低具有较强预测力，逻辑回归（TF-IDF）达到F1=0.87、AUC=0.93。
- 对于情感分类任务，正则化线性模型（逻辑回归）和集成方法（随机森林）均优于单棵决策树。
- 主要误分类模式：当评论同时包含正面词汇和总体负面倾向时，基于词袋特征的分类器容易产生误判，因为它无法捕捉上下文语义。
- BERTopic在美妆评论中识别出5个顾客关注的核心评价维度：质量/性价比、气味、包装、颜色、耐用性。
- 将预测评分与话题分配结合，可实现对产品在各维度上的细粒度分析。

### 依赖库

```
numpy, pandas, scikit-learn, matplotlib, seaborn,
beautifulsoup4, tqdm, bertopic, umap-learn, hdbscan
```