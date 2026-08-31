# AquaStereo + OAK-D W 사용 가이드

OAK-D W 카메라로 AquaStereo를 실행하는 방법입니다. 이 환경(RTX 5090 / Python 3.12 / torch 2.9)에
맞춰 이미 설정과 검증이 끝나 있습니다.

## 빠른 시작

카메라를 USB3 포트에 연결한 뒤:

```bash
cd ~/Desktop/New/AquaStereo
python oak_aquastereo.py
```

창이 뜨면 왼쪽에 정류된(rectified) 영상, 오른쪽에 시차(disparity) 맵이 나옵니다.
마우스를 올리면 그 지점의 시차와 거리(m)가 아래에 표시됩니다.

| 키 | 동작 |
|---|---|
| `q` / `ESC` | 종료 |
| `space` | 일시정지 |
| `d` | 시차 ↔ 깊이 표시 전환 |
| `s` | 현재 프레임 저장 (png + npy) |
| `p` | 포인트 클라우드 저장 (ply) |
| `[` `]` | 정제 반복 횟수 조절 (품질 ↔ 속도) |

## 카메라 없이 먼저 테스트

이미지 두 장으로 동작을 확인할 수 있습니다. **정류된(rectified) 쌍**이어야 합니다.

```bash
python oak_aquastereo.py --source images \
  --left left.png --right right.png \
  --fx 500 --baseline 0.075
```

## 주요 옵션

```bash
# 더 큰 모델 (조금 느리지만 정확도 향상)
python oak_aquastereo.py --restore_ckpt checkpoints/AquaStereo_vitb_best.pth

# 속도 우선: fp16 + 반복 횟수 축소
python oak_aquastereo.py --mixed_precision --precision_dtype float16 --iters 8

# 정확도 우선: 원본 해상도
python oak_aquastereo.py --width 1280 --height 800 --iters 32

# 수중 하우징(평면 포트)에서 사용할 때 — 아래 "수중 촬영" 참고
python oak_aquastereo.py --water_n 1.333
```

전체 옵션은 `python oak_aquastereo.py --help`로 볼 수 있습니다.

## 성능 (RTX 5090에서 실측, iters=16)

| 모델 | 해상도 | 속도 | VRAM |
|---|---|---|---|
| vits | 640×400 | 86 ms (11.6 fps) | 1.1 GB |
| vits | 640×400 (fp16) | 75 ms (13.4 fps) | 1.0 GB |
| vits | 1280×800 | 271 ms (3.7 fps) | 3.7 GB |
| vitb | 640×400 | 94 ms (10.6 fps) | 1.6 GB |
| vitb | 1280×800 | 286 ms (3.5 fps) | 4.7 GB |

fp16은 시차 오차가 평균 0.01 px 수준이라 사실상 무손실입니다.
`--mixed_precision`만 켜면 효과가 없고, `--precision_dtype float16`을 같이 줘야 빨라집니다.

## 깊이 계산

```
깊이(m) = fx × baseline / 시차(px)
```

`fx`(픽셀 단위 초점거리)와 `baseline`(m)은 카메라 캘리브레이션에서 자동으로 읽어옵니다.
직접 지정하려면 `--fx`, `--baseline`을 쓰세요.

DepthAI는 좌우 영상을 **오른쪽 카메라의 내부 파라미터로** 정류하므로, 스크립트는 CAM_C의
`fx`를 사용합니다. `--alpha`를 지정해 정류 스케일을 바꾸면 실제 `fx`가 달라지는데,
이 경우 스크립트가 프레임에 실린 값과 캘리브레이션 값을 비교해 자동으로 보정합니다.

### 수중 촬영

**평면 포트(flat port)** 하우징에서는 물–유리–공기 경계에서 광선이 굴절합니다.

```bash
python oak_aquastereo.py --water_n 1.333
```

굴절량은 광축에서 멀어질수록 커지므로, 흔히 쓰는 "초점거리에 1.333을 곱한다"는 근축
근사는 **화면 중앙에서만** 맞습니다. OAK-D W 같은 127도 광각에서는 모서리에서 오차가
수십 퍼센트에 달합니다 (검증 결과: 화면 중앙 0.12%, 모서리 79.6% 차이).

그래서 이 스크립트는 초점거리를 스칼라로 늘리는 대신, 매칭된 좌우 픽셀을 각각
**물속에서의 광선 방향으로 되돌린 뒤 삼각측량**합니다. 얇은 포트에 대해 기하학적으로
정확하고, `--water_n 1.0`이면 원래의 핀홀 공식과 완전히 일치합니다.

**돔 포트(dome port)**를 광학 중심에 정확히 맞춘 경우에는 굴절 보정이 필요 없으므로
기본값 1.0을 그대로 두세요.

정밀 계측이 목적이라면 물속에서 체커보드로 직접 캘리브레이션한 값을 `--fx`, `--fy`,
`--baseline`으로 넣는 것이 가장 정확합니다. 포트와 렌즈 사이 거리(standoff)가 크면
비중심(non-central) 효과가 남는데, 그건 이 모델이 다루지 않습니다.

## 알아둘 점

- **입력은 흑백입니다.** OAK-D W의 스테레오 쌍(OV9282)은 흑백 센서이고, DepthAI의
  정류 출력도 흑백입니다. 스크립트가 3채널로 복제해 모델에 넣습니다. AquaStereo는
  수중 컬러 감쇠에 대응하도록 학습됐지만, 스테레오 매칭 자체는 흑백에서도 정상 동작합니다
  (합성 데이터 검증에서 컬러/흑백 모두 정답 24 px에 대해 오차 0.03 px 이내).
- **광각 렌즈 정류.** OAK-D W는 화각이 넓어 3×3 호모그래피로는 부족한데, DepthAI가 85도
  이상에서 자동으로 메시 기반 정류를 씁니다. 스크립트에서 명시적으로 켜둡니다.
- **모델 가중치.** 추론에는 `checkpoints/`의 체크포인트만 있으면 됩니다. README에 나오는
  X3D·DINOv2 사전학습 파일은 학습할 때만 필요하고, 추론용 백본은 체크포인트 안에
  이미 들어 있습니다.

## 설치된 것들

이 환경에 다음 패키지를 추가로 설치해 두었습니다 (AquaStereo가 요구하지만 빠져 있던 것들):

```
timm  scipy  imageio  scikit-image  fvcore  pytorchvideo
```

원본 README는 Python 3.8 / torch 2.0.1을 권하지만, RTX 5090(sm_120)은 그 버전으로
동작하지 않습니다. 현재 설치된 Python 3.12 / torch 2.9.1+cu128 조합에서 코드 수정 없이
그대로 동작하는 것을 확인했습니다.

## 문제가 생기면

**카메라를 못 찾을 때**

```bash
lsusb | grep 03e7          # OAK가 보이는지 확인
python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

USB3 포트에 꽂고, 데이터 전송이 되는 케이블인지 확인하세요 (충전 전용 케이블은 인식되지 않습니다).
udev 규칙(`/etc/udev/rules.d/80-movidius.rules`)은 이미 설치되어 있습니다.

**시차 맵이 엉망일 때**

- 장면에 질감이 거의 없으면(민무늬 벽, 탁한 물) 매칭이 어렵습니다. 조명을 더 주세요.
- `--iters`를 32까지 올려보세요.
- 너무 가까운 물체는 시차가 커서 잘립니다. 조금 떨어뜨려 보세요.
