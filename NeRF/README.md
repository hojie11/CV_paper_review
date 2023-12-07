# NeRF
[REVIEW] Neural Radiance Fields for View Synthesis 

[Project page](https://www.matthewtancik.com/nerf)

## INTRO
기술이 발전하면서 3D 게임을 넘어 AR/VR 등의 많은 컨텐츠 등이 생기고 있으며, 이를 위해 실제 사물에 대한 3D 오브젝트를 생성하여 가상의 공간에서 활용할 수 있도록 하는 방법이 많이 쓰이고 있음

![VR CHAT](./image/vrchat.jpg "VR CHAT")
출처 : https://hello.vrchat.com

기존 3D 오브젝트 생성을 위한 별도의 장비나 Photogrammtrey 방법보다 더욱 효과적으로 수행하는 방법인 NeRF가 2020 ECCV에서 발표됨

NeRF는 실제로 3D 오브젝트를 생성하는 것이 아니라 새로운 View point에서 바라보는 이미지를 생성하여 연속적인 2D 이미지를 통해 3D 같은 효과를 제공함

이러한 방법은 기존 Point Cloud, Voxel, Mesh와 같이 Computing resource가 많이 필요하지 않음

![NeRF Method](./image/NeRF%20method.png)
출처: paper "Neural Radiance Fields for View Synthesis", fig. 1

## Overview : Neural Radiance Fields
![Oerview:NeRF](./image/overview_nerf.png)
출처: paper "Neural Radiance Fields for View Synthesis", fig. 2

(a) NeRF는 3차원 좌표 Position($x, y, z$)와 카메라 파라미터($\theta, \phi$)가 결합된 5차원 데이터를 입력으로 사용함

(b) MLP($F_\theta$)를 통해 입력에 대한 결과로 RGB와 $\sigma$(Density)를 출력함 

(c) 출력된 RGB와 $\sigma$는 Volume Rendering 기법을 통해 해당 각도에서 발사한 Ray가 지나는 점들을 압축하여 2차원 이미지로 변환됨

(d) 실제 이미지와 비교하여 생성된 이미지와의 차이를 최소화하며 학습을 진행함

## (a) 5D Input Position + Direction
