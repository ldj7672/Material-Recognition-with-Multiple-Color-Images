> ## [Material Type Recognition of Indoor Scenes via Surface Reflectance Estimation](https://ieeexplore.ieee.org/document/9658490, "paper link")
>   > Seokyeong Lee, Dongjin Lee, Hyun-Cheol Kim, Seungkyu Lee, *IEEE Access 2021*

## Introduction

위 논문에서 제안한 material type reocgnition network에 대한 설명 및 pytorch 코드 입니다.

- Multiple color images(Multi-view or multi-illumination condition)를 input으로 사용하여 material type recognition 성능을 향상시키는 network
- Color texture와 surface reflectance를 two stream deep neural network 구조로 추출
- Weight sharing backbone network(CNN)에 multiple image들을 주입하고 출력된 feature들을 순차적으로 LSTM에 주입하여 multi-view correlation을 추출

## Proposed Network details

![figure11](/figures/custom_network.png)

- **Color Feature Network** : multi-view/illumination color texture 정보 추출
- **Brightness Feature Network** : multi-view/illumination 환경에서 표면에서 반사된 빛을 관측하는 각도차이에 의한 brightness 차이를 추출
- **Attention Module** : Color & Brightness variation feature의 중요도에 self-attention을 주는 구조 (channel-wise attention, fc-relu-fc-sigmoid)
- **LSTM** : View image 별 임베딩된 feature들을 순차적으로 LSTM에 주입하여 multi-view correlation을 추출하는 구조. LSTM의 output은 fc에 연결되어 material type을 분류.

  - *OneStream_multiView_Net* : Color Feature Network + Attention Module + LSTM
Color multi view 이미지들을 LSTM을 이용하여 multi-view color texture를 추출하여 재질을 분류하는 구조

  - *TwoStream_multiView_Net* : (Color Feature Network + Brightness Variation Network) + Attention Module + LSTM
Color texture와 surface reflectance를 모두 추출하여 재질을 분류하는 two-stream 구조의 network 
Color multi view 이미지들로 differential image를 생성하고, color image들은 Color Feature Network의 input으로 differential image들은 Brightness Feature Network의 input으로 사용. 2개의 network의 output을 concat하고 attention을 거친 뒤 LSTM으로 multi-view correlation을 추출. 

  - *Differential Image* : 밝기 순으로 정렬된 multiple color 이미지 패치에서 연속된 패치끼리의 subtraction 연산을 수행한 이미지. Two-stream network 에서 이미지 패치 간의 밝기 차이를 강조하기 위해 사용.

  

## Experimental Results
- multi-view, two-stream network, attention module, differential images 의 효과를 각각 검증하기 위해 5개의 경우로 나누어 실험
<img src="/figures/results.png" width="761" height="205">
