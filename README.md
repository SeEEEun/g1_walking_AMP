# G1 Humanoid Walking with AMP (12-DOF)
NVIDIA Isaac Gym을 활용한 Unitree G1 로봇의 AMP(Adversarial Motion Prior) 기반 보행 모방 학습 프로젝트입니다.
우선, 걷기를 수행합니다.

## 🏗️ 프로젝트 구조

    isaacgymenvs/: AMP 알고리즘 및 G1 전용 학습 환경 태스크 코드

    scripts/: Motive(모션캡처) CSV 데이터를 학습용 .npy로 변환하는 전처리 스크립트

    assets/: G1 URDF 모델 및 학습용 모션 궤적 데이터 (.npy)

    cfg/: 태스크 설정 및 PPO/AMP 알고리즘 하이퍼파라미터 (.yaml)

## 🚀 시작하기 (Server Setup)
### 1. 환경 설치
레포지토리를 클론한 후, 해당 폴더에서 패키지를 설치합니다.
```bash

git clone https://github.com/SeEEEun/g1_walking_AMP.git
cd g1_walking_AMP
pip install -e .
```

### 2. 데이터 전처리 (Motion Preprocessing)

팀원들이 획득한 모션 캡처 데이터를 학습에 사용하려면 아래 스크립트를 실행하세요.
```bash
cd scripts
python preprocess_motive.py --input your_motion.csv --output ../assets/amp/g1_motion.npy
```
### 3. 학습 실행 (Training)

```bash
#루트 폴더에서 실행
python train.py task=HumanoidAMP
```

## 🛠️ 주요 설정 사항
    Robot: Unitree G1 (12-DOF 관절 구성 완료)
    Algorithm: Adversarial Motion Prior (AMP)
