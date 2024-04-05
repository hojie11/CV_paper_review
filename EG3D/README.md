# [REVIEW] EG3D : Efficient Geometry-aware 3D Generative Adversarial Networks(작성중)
> CVPR 2022 </br>
> Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J. Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, Gordon Wetzstein </br>
> Stanford Univ | NVIDIA

오늘 리뷰하려는 논문은 3D GAN 모델 중 하나인 `EG3D` 입니다. `3D GAN`은 `NeRF`와 `GAN`을 합쳐 **3D-aware**한 이미지 생성을 목표로 하는 생성 모델의 한 종류입니다. `EG3D`에서는 빠르고 선명한 이미지 생성을 위해 `1. Hybrid Representation` 방법을 제안하였고, 불필요한 View-inconsistent를 정규화하기 위한 `2. Dual Discriminator`를 제안하였습니다. 자세한 내용은 아래에서 설명하겠습니다.

현재까지 발표된 `GAN` 모델은 **1. 고해상도 이미지**나 **2. Photorealistic한 이미지** 등을 생성이 가능하여 놀라운 결과를 보여주고 있습니다. 하지만, 2차원에서만 동작 가능하다는 단점이 있어 생성하는 데이터가 이미지에 한정됩니다. </br>
`3D GAN`은 단어에서 유추할 수 있듯이, 3차원 정보 출력이 가능한 `GAN`(이하 `2D GAN`) 모델입니다. **3D Position**이나 **Multi view** 데이터 없이 다양한 각도의 이미지 생성이 가능하다는 것이 큰 장점입니다. 한편, `2D GAN`의 이미지 생성 결과에 비해서는 **1. 이미지의 퀄리티가 낮고**, **2. 3D shape 마저 정교하지 못하다**는 평이 있었습니다. </br>
`3D GAN`은 모델의 Generator가 갖는 3차원 구조에 대한 Inductive bias와 View-consistent한 이미지를 생성하기 위한 Neural Rendering의 조합으로 결과를 생성합니다. 기존에 사용하는 3차원 좌표와 같이 **Implicit**한 방법이나 Voxel과 같은 **Explicit**한 방법은 Single scene에 대해서 괜찮은 결과를 보여주었지만, **1. 메모리 사용이 비효율적**이고 **2. 너무 느리다**는 단점이 있습니다.

`EG3D`에서는 기존의 **3D GAN과 Data Representation 방법의 문제점**을 개선하기 위한 방법들을 제시하였는데, 어떤 내용이 있는지 이해한 내용을 바탕으로 설명해보겠습니다.

<p align=center>
    <img src="./image/framework.jpg">
</p>

위 사진은 `EG3D`의 전체 구조입니다. `Mapping Network`에 **Latent vector와 카메라 파라미터**를 바탕으로 **새로운 Feature vector**를 출력합니다. 카메라 파라미터를 사용한 이유는 **카메라 포즈와 다른 속성에 대한 Decoupling을 유도**하기 위함입니다. 이렇게 하면, `Pose Conditioning`을 통해 데이터셋에 포함된 `Pose-dependent bias`를 표현할 수 있다고 합니다. </br>
이후, 이미지 생성을 위한 `StyleGAN2의 Generator`를 이용해서 **96채널의 Feature**를 추출하고 32채널씩 Reshaep하여 제안한 방법인 `Tri-plane`을 생성하였습니다. `Tri-plane`이 `EG3D`에서 제안한 Hybrid한 방법의 **Key point**입니다. 코드를 보면 더 쉽게 이해하실 수 있습니다.

```python
def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]
```

위의 `generate_planes`함수가 Tri-plane을 생성하는 코드인데 생각보다 간단합니다. 각 면이 이루는 정보를 벡터로 표시하여 나타내었고, `project_onto_planes`함수로 각 면에 투영시킵니다. 여기서, `coordinates`변수는 카메라 파라미터로 생성한 Ray를 나타냅니다. 다음에 `StyleGAN2 Generator`로 생성한 **Feature**를 투영된 Ray의 좌표에 따라 Sampling 후 Summation을 통해 최종 **Feature**를 생성합니다. </br>
`Tri-plane`은 Implicit한 방법과 같이 3차원 좌표 정보를 활용하였고, 해당 정보를 각 축이 이루는 면에 투영함으로써 Explicit한 방법을 동시에 사용하였습니다. 그 결과, 각 방법의 장점을 갖는 Hybrid한 방법을 구현하였습니다.(자세한 내용은 [EG3D 논문]("https://nvlabs.github.io/eg3d/media/eg3d.pdf")의 3번 항목에서 확인하실 수 있습니다.) </br>

논문에서 제시한 Single Scene Overfitting(SSO) 실험에서도 `Tri-plane` Representation 방법이 기존보다 우수하다는 것을 증명하였습니다.

<p align=center>
    <img src="./image/sso.jpg">
</p>

$x, y, z$에 해당하는 3차원 좌표를 각 축이 이루는 평면 $F_{xy}, F_{xz}, F_{yz}$에 투영시킴으로써, 채널을 줄이는 효과를 볼 수 있다고 합니다. 위의 실험 결과에서 볼 수 있는 것 처럼, Implict한 방법을 기준으로 약 **7.8배가 빠르고, 메모리 효율도 우수**한 것을 볼 수 있습니다.