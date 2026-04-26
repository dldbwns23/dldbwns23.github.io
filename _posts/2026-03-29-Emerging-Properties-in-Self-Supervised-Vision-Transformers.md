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

DINO(Self-**di**stillation with **no** labels) 프레임워크의 핵심 작동 원리는 정답 레이블이 배제된 상태에서 네트워크 스스로 자신의 이전 상태(Teacher)를 증류(Distillation)하여 학습하는 Self-Distillation 구조에 있다. 이 프레임워크는 동일한 구조를 공유하는 Student 네트워크($g_{\theta_{s}}$)와 Teacher 네트워크($g_{\theta_{t}}$) 간의 상호작용을 기반으로 최적화를 수행한다.

{% raw %}
<img src="/images/Emerging_Properties_in_Self_Supervised_Vision_Transformers/fig_2.png" alt="Figure 2: Self-distillation with no labels">
{% endraw %}
**Figure 2:** DINO 프레임워크의 Self-distillation 구조. 동일한 입력 이미지 $x$로부터 무작위 증강(augmentation)을 통해 생성된 서로 다른 뷰(view) $x_1, x_2$를 두 네트워크에 각각 통과시킨다. 두 네트워크는 동일한 아키텍처를 공유하지만, 파라미터는 상이하게 유지된다. Teacher의 출력은 배치(batch) 평균을 통해 Centering 처리되며, 최종적으로 Student와 Teacher의 출력 확률 분포 간 유사도를 측정하여 Cross-entropy loss를 최소화하는 방향으로 Student의 파라미터만 업데이트된다. Teacher는 역전파(backpropagation) 과정 없이 Student 가중치의 지수 이동 평균(EMA)으로 갱신된다.

#### 2.1. Teacher-Student 최적화 및 Momentum Encoder의 동역학
DINO의 구조적 특징은 Teacher 네트워크가 사전에 훈련되어 고정된 모델이 아니라, Student 네트워크의 과거 상태를 기반으로 동적으로 갱신된다는 점이다. 두 네트워크는 주어진 이미지 뷰에 대해 K-차원의 확률 분포 $P_{s}$와 $P_{t}$를 출력하며, Student는 Teacher의 분포 $P_{t}$와 일치하도록 아래의 Cross-entropy loss 식을 통해 학습된다.

$$\min_{\theta_{s}}H(P_{t}(x), P_{s}(x))$$

여기서 Teacher 네트워크 파라미터 $\theta_{t}$는 경사하강법으로 직접 갱신되지 않고, Student 파라미터 $\theta_{s}$의 지수 이동 평균(Exponential Moving Average, EMA)을 적용한 Momentum Encoder 방식으로 업데이트된다. 업데이트 규칙은 다음과 같다.

$$\theta_{t} \leftarrow \lambda\theta_{t} + (1-\lambda)\theta_{s}$$

{% raw %}
<img src="/images/Emerging_Properties_in_Self_Supervised_Vision_Transformers/fig_6.png" alt="Figure 6: Top-1 accuracy on ImageNet validation with k-NN classifier">
{% endraw %}
**Figure 6:** (Left) 학습 과정 중 Student와 Teacher의 성능 추이. Teacher가 시종일관 Student를 상회하며 학습을 견인하는 모습을 보인다. (Right) Teacher 구축 전략 비교. Momentum 방식이 가장 우수하며, 단순 Student 복사(Student copy) 방식은 수렴하지 못하고 붕괴한다.

연구진은 이 메커니즘이 Polyak-Ruppert averaging에 기반한 모델 앙상블 효과를 창출한다고 논리적으로 해석한다. 학습 전반에 걸쳐 지속적인 가중치 평균화를 통해 구축된 Teacher 모델은 단일 반복(iteration) 단위로 갱신되는 Student보다 수학적으로 더 안정적이고 우수한 품질의 타겟 특성(feature)을 생성해 내며, 결과적으로 Student가 안정적으로 수렴할 수 있도록 학습을 견인한다.

#### 2.2. Avoiding Collapse: Centering과 Sharpening의 상호 보완성
자기지도학습 모델이 직면하는 치명적인 한계는 네트워크가 모든 입력 이미지에 대해 동일하고 획일화된 벡터를 출력해버리는 모델 붕괴(Collapse) 현상이다. DINO는 복잡한 제약 조건이나 별도의 Predictor 구조를 추가하는 대신, Teacher 출력 분포에 Centering과 Sharpening이라는 두 가지 연산을 동시 적용하여 붕괴를 정교하게 제어한다.

* **Centering:** Teacher의 출력에 배치(batch) 단위로 계산된 이동 평균 편향(bias) $c$를 더하는 연산이다. 이는 네트워크의 특정 차원(dimension)이 활성화 값을 과도하게 지배하여 발생하는 붕괴 현상을 차단한다. 그러나 이 연산 단독으로는 네트워크의 출력을 균등 분포(Uniform distribution) 형태로 평탄화시켜 모든 정보량이 소실되는 또 다른 형태의 붕괴를 유발할 위험이 있다.
* **Sharpening:** Teacher의 Softmax 온도(temperature) 파라미터 $\tau_{t}$를 극단적으로 낮게 설정하여, 출력 분포의 분산을 줄이고 확률값을 특정 차원에 뾰족하게 집중시키는 기법이다. 이는 Centering이 유발할 수 있는 균등 분포화 현상을 수학적으로 상쇄한다.

{% raw %}
<img src="/images/Emerging_Properties_in_Self_Supervised_Vision_Transformers/fig_7.png" alt="Figure 7: Collapse study">
{% endraw %}
**Figure 7:** (Left) Teacher의 Target Entropy 변화. (Right) Teacher와 Student 간의 KL divergence 변화. Centering만 수행하면 출력이 균등 분포로 붕괴하고, Sharpening만 수행하면 한 차원이 지배하는 붕괴가 일어난다. 두 연산이 상호 보완적으로 결합될 때만 엔트로피가 적정 수준을 유지하며 안정적인 최적화가 이루어진다.

#### 2.3. Multi-crop 전략과 Local-to-Global 대응 학습
Vision Transformer 구조의 공간적 특성 이해도를 극대화하기 위해 DINO는 Multi-crop Augmentation 기법을 프레임워크의 핵심 축으로 채택한다. 
* **Global views:** 원본 이미지 면적의 50% 이상을 포함하는 고해상도($224^{2}$) 뷰 2개.
* **Local views:** 원본 이미지 면적의 50% 미만 영역만을 포함하는 저해상도($96^{2}$) 뷰 여러 개.

학습 시 Student 네트워크에는 Global 뷰와 Local 뷰가 모두 입력되지만, Teacher 네트워크에는 오직 Global 뷰만이 입력된다. 이를 통해 Student 모델은 이미지의 극히 지엽적인 일부(Local)만 관찰하고도, Teacher가 넓은 영역(Global)을 보고 파악한 전체적인 문맥과 대응하도록 손실 함수가 구성된다. 이러한 "Local-to-Global" 구조적 대응 학습은 네트워크가 객체의 형태와 경계를 명시적으로 분리하고 이해하는 비지도 객체 분할(Unsupervised Object Segmentation) 역량을 확보하는 결정적 기반이 된다.

---

### 3. Key Findings & Emerging Properties

#### 3.1. 명시적인 Semantic Segmentation 능력 (Figure 1, 3, 4)
DINO로 학습된 ViT는 객체의 경계에 대한 명시적인 정보를 스스로 학습한다.

{% raw %}
<img src="/images/Emerging_Properties_in_Self_Supervised_Vision_Transformers/fig_1.png" alt="Figure 1: Self-attention from a Vision Transformer with 8 x 8 patches trained with no supervision">
{% endraw %}
**Figure 1:** 지도 학습 없이 DINO로 학습된 ViT의 Self-attention 맵. 마지막 레이어의 [CLS] 토큰 attention이 객체의 레이아웃과 경계를 정확히 포착하고 있다.

{% raw %}
<img src="/images/Emerging_Properties_in_Self_Supervised_Vision_Transformers/fig_3.png" alt="Figure 3: Attention maps from multiple heads">
{% endraw %}
**Figure 3:** 다양한 Attention Head 시각화. 각 head가 이미지 내의 서로 다른 의미론적 영역이나 객체의 세부 부위에 독립적으로 집중하고 있음을 보여준다.

{% raw %}
<img src="/images/Emerging_Properties_in_Self_Supervised_Vision_Transformers/fig_4.png" alt="Figure 4: Segmentations from supervised versus DINO">
{% endraw %}
**Figure 4:** Supervised ViT와 DINO ViT의 Segmentation 비교. Supervised 모델은 clutter가 있는 상황에서 객체에 집중하지 못하는 반면, DINO는 명확한 마스크를 형성한다. 하단 테이블의 Jaccard similarity 지표에서 DINO(45.9)가 Supervised(27.3)를 크게 앞선다.

#### 3.2. 성능 지표 및 패치 크기의 영향 (Table 2, Figure 5)
DINO는 특히 단순한 k-NN 분류기만으로도 매우 강력한 성능을 발휘한다.

{% raw %}
<img src="/images/Emerging_Properties_in_Self_Supervised_Vision_Transformers/table_2.png" alt="Table 2: Linear and k-NN classificationi on ImageNet">
{% endraw %}
**Table 2:** ImageNet에서의 선형 및 k-NN 분류 성능 비교. DINO ViT-S/8은 k-NN만으로 78.3%의 정확도를 기록하여 기존 방법론들을 압도한다. 이는 feature space가 의미론적으로 매우 잘 정렬되어 있음을 뜻한다.

{% raw %}
<img src="/images/Emerging_Properties_in_Self_Supervised_Vision_Transformers/fig_5.png" alt="Figure 5: Effect of Patch Size">
{% endraw %}
**Figure 5:** 패치 크기(Patch size)에 따른 k-NN 성능과 처리량(throughput)의 관계. 패치 크기를 16에서 8, 5로 줄일수록 정확도는 비약적으로 향상되지만, 연산 비용 증가로 인해 처리량은 감소한다.

---

### 4. Ablation Study (Table 7)

DINO 프레임워크의 각 구성 요소가 성능에 미치는 영향을 분석하였다.

{% raw %}
<img src="/images/Emerging_Properties_in_Self_Supervised_Vision_Transformers/table_7.png" alt="Table 7: Important component for self-supervised ViT pretraining">
{% endraw %}
**Table 7:** 주요 컴포넌트의 중요도 분석. Momentum Encoder가 없을 경우 학습 자체가 불가능하며(row 2), Multi-crop(MC)과 Cross-entropy(CE) loss가 성능 확보에 결정적인 역할을 수행함을 보여준다.

---

### 5. Conclusion

DINO 프레임워크는 유연성이 뛰어나 아키텍처 수정 없이 Convolution Network와 ViT 모두에 범용적으로 적용 가능하다. 특히 정답 레이블이 없는 조건에서도 ViT가 스스로 영상의 의미론적 구조를 이해하고 객체의 경계를 파악하는 창발적 특성(Emerging Properties)을 증명해 낸 것은 컴퓨터 비전 분야의 중대한 진전이다.

이러한 발견은 기상 관측 데이터와 같이 레이블링이 어려운 대규모 비정형 데이터에서 유의미한 공간적 hotspot을 추출하고 정밀하게 분할해야 하는 연구에 핵심적인 방법론적 근거를 제시한다. 데이터의 국지적 특성과 전체적 맥락을 연결하는 DINO의 원리는 향후 연구 모델링의 견고한 기준점이 될 것이다.
