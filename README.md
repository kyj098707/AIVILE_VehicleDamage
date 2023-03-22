# 차량공유업체의 차량 파손여부 분류 프로젝트
## 개요
본 프로젝트는 KT AIVLE SCHOOL에서 진행한 차량 파손 여부를 판별하는 모델을 만드는 프로젝트입니다.
## 데이터셋
**데이터의 개수가 매우 작고, 차량의 파손 정도가 커 ai hub의 차량 파손 데이터 1200장을 테스트 데이터셋에 추가하여 추가적인 실험을 진행하였습니다. 자세한 데이터는 정책상 공개가 불가능합니다.**

KT Aivle에서 DALL-E를 통해 직접 제작한 데이터셋, ai hub 데이터셋을 사용하였습니다.

<br> **일반 차량 데이터셋: 302, 파손된 차량 : 303 + 1200**
<br> train data : 401(원본만 포함), test data : 204 + 1200

## 사용 기술 스택
<p align="center">
  <img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=TensorFlow&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/scikit-learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/-matplotlib-blue"/></a>&nbsp
  <img src="https://img.shields.io/badge/Wandb-black"/></a>&nbsp
</p>

## 최종 프로젝트 결과
### 초기 데이터셋
ACC : 100%
### 파손 데이터 추가 데이터셋
ACC : 91%

## 모델의 크기에 따른 정확도 실험
train data는 401개로 매우 작은 상태였습니다. 모델이 클수록 데이터는 이미지의 더욱 디테일한 특성을 학습하게 되며 이는 데이터셋이 많으면 다양한 이미지의 특성들을 학습할 수 있는 반면
데이터셋이 작은 상황에서는 일반적인 데이터의 특성을 많이 잃게 되어 성능이 나빠지는 것을 볼 수 있었습니다. resnet34와 resnet과 비슷한 모델인 convnext를 통해 이를 실험해봤습니다.
<p align=center>
<img width=600 src="https://user-images.githubusercontent.com/54027397/226883540-c80d6aec-d147-4bd6-9371-a3df47bf38eb.png"/>
<p/>

## Crop류의 augmentation 적용 여부, Mixup 적용 여부
train에서 학습되는 파손된 차량의 경우 매우 큰 파손 정도를 가지고 있는 반면 ai hub에서 받은 데이터셋은 경미한 파손 차량도 존재하였습니다. Crop을 하게 되면 파손된 차량 정보가 날라가 성능이 나빠질지와 오히려 작은 파손을 모델이 학습하게 되어 성능이 좋아질지에 대한 실험을 진행하였습니다. 실험결과 crop을 한 이미지가 성능이 좋은 것을 알 수 있었습니다.
<br> 또한 mixup이라는 augmentation을 같은 label끼리 진행하였습니다. 이때 성능은 오히려 나빠지게 되었는데 저희 조는 해당 이유를 파손되지 않은 차량끼리의 mixup에서 노이즈가 생겨버려 오히려 모델에게 혼동을 줬다고 판단을 하였습니다.
<p align=center>
<img width=600 src="https://user-images.githubusercontent.com/54027397/226886911-20814814-16d1-4ee6-b87f-dbba91872733.png"/>
<p/>

## Transformer(VIT)의 성능
일반적으로 VIT는 원 논문에서 언급하였듯이 inductive bias가 없어 적은 데이터셋에 대해서 좋지않은 성능을 가지고 있습니다. 이를 확인하고자 VIT모델 또한 적용시켰으며 모든 CNN 보다 낮은 성능(MAX ACC:0.25)을 볼 수 있었습니다. 
<p align=center>
<img width=600 src="https://user-images.githubusercontent.com/54027397/226887457-92e0e29d-e2ff-4c87-b0b7-c7033636f591.png"/>
<p/>

## CutMix 적용
Crop류의 augmentation에 좋은 성능을 보여주었기 때문에 CutMix에도 좋은 성능이 나올것이라고 생각을 하여 해당 실험을 진행하였습니다. 이때 파손되지 않은 데이터셋끼리 cutmix를 진행하여도 해당 사진은 파손된 데이터로 보는 것이 맞다고 판단하여 모든 cutmix를 진행한 데이터는 파손된 데이터로 라벨링을 진행하였습니다. 기존 cutmix를 적용하기 이전보다 성능이 오르는 것을 볼 수 있었습니다. 
<p align=center>
<img width=600 src="https://user-images.githubusercontent.com/54027397/226889127-3ddad10a-1a51-40b5-83a5-41b5d1040b39.png"/>
<p/>

## Focal Loss 적용
Cutmix를 통해서 파손된 차량의 데이터셋이 증가하였고, test에도 일방적으로 파손된 차량의 데이터셋을 늘렸기 때문에 클래스불균형이 매우 심한 상태였습니다. 이를 극복하고자 Focal Loss를 적용하였으며 성능이 매우 향상되는 것을 알 수 있습니다(0.91). 사실 이는 Focal loss가 틀리기 쉬운 클래스에 대해 더 가중치를 주기 때문에 모델이 학습을 잘했다기 보다는 압도적으로 많은 파손 데이터셋에 대해서 대부분의 예측을 파손되었다고 예측한것으로 판단됩니다.
<p align=center>
<img width=600 src="https://user-images.githubusercontent.com/54027397/226890192-f5ee9858-c15f-4cd7-8cd4-4bfeeb52a3ba.png"/>
<p/>

## 11반 42조 맴버
김윤종 : https://github.com/kyj098707
<br>박경덕 : https://github.com/Ramdatech
<br>황소정 : https://github.com/sora0319
<br>김채원 : https://github.com/chaewon0824
<br>강선후 : https://github.com/rkdwhdrjf
<br>정정해 : https://github.com/JeongJeonghae
<br>오승권 : https://github.com/loveand30



