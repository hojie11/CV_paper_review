# [REVIEW] 3DGS : 3D Gaussian Splatting for real time radiance field rendering (작성중)
> SIGGRAPH 2023 </br>
> Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis</br>
> Inria | Université Côte d'Azur | MPI Informatik

오늘 리뷰하는 논문은 NeRF와 같이 Novel View Synthesis 분야에 엄청난 인기를 끌었던 3D Gaussian Splatting에 대한 논문입니다. NeRF 논문이 발표되고 해당 방법을 개선하는 내용의 논문과 응용한 논문이 엄청나게 쏟아졌었는데, 현재 3DGS를 활용한 논문도 계속해서 발표되고 있어 해당 내용을 정리하고자 진행하게 되었습니다.

3DGS는 NeRF의 task와 동일한, image set과 camera pose값이 주어지면 다양한 시점에 대해서 Rendering을 수행하여 해당 scene을 3D로 표현합니다. 해당 논문에서 제안한 방법의 결과는 고해상도(1080p) Rendering quality 부문 SOTA를 기록한 Mip-NeRF 360 보다 뛰어난 결과를 보였으며, Training time 부문 SOTA를 기록한 InstantNGP 보다 빠르게 학습이 가능합니다.

지금부터 어떻게 이런 결과를 이룰 수 있었는 지에 대해서 이해한 내용을 설명해보도록 하겠습니다.