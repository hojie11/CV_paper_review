# [REVIEW] 3DGS : 3D Gaussian Splatting for real time radiance field rendering (작성중)
> SIGGRAPH 2023 </br>
> Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis</br>
> Inria | Université Côte d'Azur | MPI Informatik

오늘 리뷰하는 논문은 NeRF와 같이 Novel View Synthesis 분야에 엄청난 인기를 끌었던 3D Gaussian Splatting에 대한 논문입니다. NeRF 논문이 발표되고 해당 방법을 개선하는 내용의 논문과 응용한 논문이 엄청나게 쏟아졌었는데, 현재 3DGS를 활용한 논문도 계속해서 발표되고 있어 해당 내용을 정리하고자 진행하게 되었습니다.

3DGS는 NeRF의 task와 동일한, image set과 camera pose값이 주어지면 다양한 시점에 대해서 Rendering을 수행하여 해당 scene을 3D로 표현합니다. 해당 논문에서 제안한 방법의 결과는 고해상도(1080p) Rendering quality 부문 SOTA를 기록한 Mip-NeRF 360 보다 뛰어난 결과를 보였으며, Training time 부문 SOTA를 기록한 InstantNGP 보다 빠르게 학습이 가능합니다.

지금부터 어떻게 이런 결과를 이룰 수 있었는 지에 대해서 이해한 내용을 설명해보도록 하겠습니다.

## Overview
<p align=center>
    <img src="./image/overview.png">
</p>

해당 논문의 전체 흐름은 위의 그림과 같은 순서로 구성되어 있습니다.
4개의 핵심 Block으로 구성되어 있어 어떻게 구성되어 있는지 쉽게 파악할 수 있었습니다.

1. Initialization : 3D Gaussian의 초기값을 설정하는 구간입니다. COLMAP과 같은 SfM(Structure from Motion) 알고리즘을 이용하여 연속된 이미지를 통해 카메라 파라미터와 Point Cloud를 추출하여 초기 3D Gaussian의 값으로 할당해줍니다.
2. Projection : 3D Gaussian을 2D Image plane으로 투영시켜 2D Gaussian으로 변환하는 구간입니다. Image plane으로 투영시켜 Ground Truth 이미지와 비교하여 학습과정에서 모델의 파라미터를 업데이트하기 위한 구간입니다.
3. Differentiable Tile Rasterizer : 해당 블럭의 이름에서 알 수 있듯이, 미분 가능한 Tile들을 Rasterization하여 이미지를 생성하는 구간입니다. 해당 논문의 빠른 Rendering이 가능하도록 한 핵심 아이디어 부분이라고 생각합니다.
4. Adaptive Density Control : 역전파 과정에서 Gradient를 통해 Gaussian의 형태를 업데이트 하는 구간입니다. 해당 논문에서는 Under Reconstruction과 Over Reconstruction의 경우에 대해 Optimization 과정을 거쳐 수행한다고 합니다.