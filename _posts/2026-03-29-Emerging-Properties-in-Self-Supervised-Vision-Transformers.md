---
layout: post
title: "Emerging Properties in Self Supervised Vision Transformers"
date: 2026-03-29 21:24:00 +0900
tags: [Article, ML/DL]
---

## [논문 리뷰] Emerging Properties in Self-Supervised Vision Transformers (DINO)

본 글에서는 2021년 ICCV에서 발표된 "Emerging Properties in Self-Supervised Vision Transformers"를 읽고 이해한 내용을 정리한다. 해당 연구는 Self-Supervised Learning 방법론을 Vision Transformer(ViT)에 적용했을 때 발현되는 고유한 특성을 분석하며, Knowledge Distillation 개념을 차용한 단순하고 효과적인 프레임워크인 DINO를 제안한다.

최근 기후 데이터 분석, 특히 극단적 강수량의 주요 발생 지역(hotspot)과 같은 복잡한 공간적 특징을 탐지하기 위해 DINO와 ViT를 활용한 Self-Supervised Learning 파이프라인을 설계하고 있다. 정답 레이블이 없는 대규모 관측 데이터에서 모델이 데이터 내부의 구조와 경계를 스스로 학습하는 원리를 파악하는 것은 연구의 논리적 완결성을 위해 필수적이다. 이에 논문의 핵심 방법론과 주요 실험 데이터를 체계적으로 정리하였다.

---

### 1. Research Motivation

기존 Vision Transformer(ViT)는 시각 인식 분야에서 뛰어난 성과를 거두었음에도 불구하고, Convolution Network 기반 모델 대비 방대한 학습 데이터와 높은 연산량을 요구한다는 한계가 있었다. 또한, 학습된 feature들이 기존 모델과 차별화되는 ViT만의 고유한 이점을 명확히 보여주지 못했다.

연구진은 자연어 처리(NLP) 분야의 성공 요인이 BERT나 GPT와 같은 Self-Supervised pretraining에 있음에 주목했다. 이미지 분류의 Supervised Learning이 시각적 정보를 단일 레이블로 축소시키는 것과 달리, 레이블 없이 이미지 자체의 정보를 활용하는 방식은 ViT에서 새로운 창발적 특성을 이끌어낼 것으로 기대되었다.

---

### 2. DINO Framework Methodology

DINO(Self-**di**stillation with **no** labels)는 레이블이 없는 환경에서 수행되는 Self-Distillation 구조를 취한다. Student 네트워크 $g_{\theta_{s}}$가 Teacher 네트워크 $g_{\theta_{t}}$의 출력을 예측하도록 학습한다.


**Figure 2 설명:** DINO 프레임워크의 Self-distillation 구조. 입력 이미지 $x$로부터 생성된 서로 다른 증강 뷰 $x_1, x_2$를 두 네트워크에 입력한다. Teacher 출력은 배치 평균으로 Centering 처리되며, 두 네트워크의 출력 분포 간 유사도를 Cross-entropy loss로 최소화하여 Student를 업데이트한다. Teacher는 역전파 없이 Student 가중치의 지수 이동 평균(EMA)으로 갱신되는 Momentum Encoder 역할을 수행한다.

#### 2.1. Teacher Network의 동역학 (Figure 6)
Teacher 네트워크는 Student보다 항상 우수한 성능을 유지하며 학습의 타겟을 제공한다.


**Figure 6 설명:** (Left) 학습 과정 중 Student와 Teacher의 성능 추이. Teacher가 시종일관 Student를 상회하며 학습을 견인하는 모습을 보인다. (Right) Teacher 구축 전략 비교. Momentum 방식이 가장 우수하며, 단순 Student 복사(Student copy) 방식은 수렴하지 못하고 붕괴한다.

#### 2.2. Avoiding Collapse (Figure 7)
Self-Supervised Learning의 고질적인 문제인 모델 붕괴(Collapse)를 방지하기 위해 DINO는 Centering과 Sharpening의 균형을 활용한다.


**Figure 7 설명:** (Left) Teacher의 Target Entropy 변화. (Right) Teacher와 Student 간의 KL divergence 변화. Centering만 수행하면 출력이 균등 분포로 붕괴하고, Sharpening만 수행하면 한 차원이 지배하는 붕괴가 일어난다. 두 연산이 동시에 적용될 때만 안정적인 학습이 가능하다.

#### 2.3. Multi-crop 전략
DINO는 이미지의 지엽적인 부분(local)을 통해 전체적인 문맥(global)을 유추하는 "local-to-global" 대응 관계를 학습하기 위해 Multi-crop을 적극 활용한다.

---

### 3. Key Findings & Emerging Properties

#### 3.1. 명시적인 Semantic Segmentation 능력 (Figure 1, 3, 4)
DINO로 학습된 ViT는 객체의 경계에 대한 명시적인 정보를 스스로 학습한다.


**Figure 1 설명:** 지도 학습 없이 DINO로 학습된 ViT의 Self-attention 맵. 마지막 레이어의 [CLS] 토큰 attention이 객체의 레이아웃과 경계를 정확히 포착하고 있다.


**Figure 3 설명:** 다양한 Attention Head 시각화. 각 head가 이미지 내의 서로 다른 의미론적 영역이나 객체의 세부 부위에 독립적으로 집중하고 있음을 보여준다.


**Figure 4 설명:** Supervised ViT와 DINO ViT의 Segmentation 비교. Supervised 모델은 clutter가 있는 상황에서 객체에 집중하지 못하는 반면, DINO는 명확한 마스크를 형성한다. 하단 테이블의 Jaccard similarity 지표에서 DINO(45.9)가 Supervised(27.3)를 크게 앞선다.

#### 3.2. 성능 지표 및 패치 크기의 영향 (Table 2, Figure 5)
DINO는 특히 단순한 k-NN 분류기만으로도 매우 강력한 성능을 발휘한다.


**Table 2 설명:** ImageNet에서의 선형 및 k-NN 분류 성능 비교. DINO ViT-S/8은 k-NN만으로 78.3%의 정확도를 기록하여 기존 방법론들을 압도한다. 이는 feature space가 의미론적으로 매우 잘 정렬되어 있음을 뜻한다.

**Figure 5 설명:** 패치 크기(Patch size)에 따른 k-NN 성능과 처리량(throughput)의 관계. 패치 크기를 16에서 8, 5로 줄일수록 정확도는 비약적으로 향상되지만, 연산 비용 증가로 인해 처리량은 감소한다.

---

### 4. Ablation Study (Table 7)

DINO 프레임워크의 각 구성 요소가 성능에 미치는 영향을 분석하였다.

**Table 7 설명:** 주요 컴포넌트의 중요도 분석. Momentum Encoder가 없을 경우 학습 자체가 불가능하며(row 2), Multi-crop(MC)과 Cross-entropy(CE) loss가 성능 확보에 결정적인 역할을 수행함을 보여준다.

---

### 5. Conclusion

DINO 프레임워크는 유연성이 뛰어나 아키텍처 수정 없이 Convolution Network와 ViT 모두에 범용적으로 적용 가능하다. 특히 정답 레이블이 없는 조건에서도 ViT가 스스로 영상의 의미론적 구조를 이해하고 객체의 경계를 파악하는 창발적 특성(Emerging Properties)을 증명해 낸 것은 컴퓨터 비전 분야의 중대한 진전이다.

이러한 발견은 기상 관측 데이터와 같이 레이블링이 어려운 대규모 비정형 데이터에서 유의미한 공간적 hotspot을 추출하고 정밀하게 분할해야 하는 연구에 핵심적인 방법론적 근거를 제시한다. 데이터의 국지적 특성과 전체적 맥락을 연결하는 DINO의 원리는 향후 연구 모델링의 견고한 기준점이 될 것이다.
