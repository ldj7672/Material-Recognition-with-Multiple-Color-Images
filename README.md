>[# **Material Type Recognition of Indoor Scenes via Surface Reflectance Estimation**](https://ieeexplore.ieee.org/document/9658490, "paper link")
>   > Seokyeong Lee, Dongjin Lee, Hyun-Cheol Kim, Seungkyu Lee, *IEEE Access 2021*

# Introduction

위 논문에서 제안한 material type reocgnition network에 대한 설명입니다.

- Multi-view 또는 multi-illumination 환경에서 color texture와 surface reflectance를 encoding하여 patch-wise material claasification을 수행하는 네트워크
- weight sharing하는 backbone network(f)에 multi-view patch들(p_1,p_2,...,p_n)을 주입하고 임베딩된 f(p_1),f(p_2),...,f(p_n)들은 순차적으로 LSTM에 주입하여 multi-view correlation을 추출하는 network 

## Basic Cumtom Network

![figure11](https://user-images.githubusercontent.com/96943196/147853235-e024583d-d4f8-4a8a-a174-2bb23d980a61.png)

- **Color Feature Network** : multi-view color texture 정보 추출
- **Brightness Feature Network** : multi-view 환경에서 표면에서 반사된 빛을 관측하는 각도차이에 의한 brightness 차이를 추출
- **Attention Module** : 2개의 backbone network의 output feature에 channel-wise attention을 가해주는 구조(fc-relu-fc-sigmoid)
- **LSTM** : view별 임베딩된 feature들을 순차적으로 LSTM에 주입하여 multi-view correlation을 추출하는 구조. LSTM의 output은 fc에 연결되어 material type을 분류.

  - *OneStream_multiView_Net* : Color Feature Network + Attention Module + LSTM
Color multi view 이미지들을 LSTM을 이용하여 multi-view color texture를 encoding하여 재질을 분류하는 구조

  - *TwoStream_multiView_Net* : (Color Feature Network + Brightness Variation Network) + Attention Module + LSTM
Color texture와 surface reflectance를 encoding하여 재질을 분류하는 two-stream 구조의 network 
Color multi view 이미지들로 differential image를 생성하고, color image들은 Color Feature Network의 input으로 differential image들은 Brightness Feature Network의 input으로 사용. 2개의 network의 output을 concat하고 attention을 거친 뒤 LSTM으로 multi-view correlation을 추출. 
  


## Main Network
TBD.. 
