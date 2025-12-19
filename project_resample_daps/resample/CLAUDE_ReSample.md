# ReSample 코드 구조 및 실행 흐름

## 개요

ReSample (ICLR 2024) - Latent Diffusion Model을 사용해 inverse problem을 푸는 알고리즘.
Hard data consistency를 통해 measurement-consistent한 샘플을 생성.

---

## 디렉토리 구조

```
resample/
├── sample_condition.py      # 메인 실행 파일 (inference entry point)
├── model_loader.py          # 모델 로딩 유틸
├── environment.yaml         # Conda 환경 설정
│
├── ldm/                     # Latent Diffusion Model 코어
│   ├── models/
│   │   ├── diffusion/       # DDIM, DDPM, DPM-Solver 등 샘플러
│   │   └── autoencoder.py   # VQ-VAE (이미지↔latent 변환)
│   └── modules/             # UNet, Attention, Encoder 등
│
├── ldm_inverse/             # Inverse Problem 핵심 모듈
│   ├── condition_methods.py # PosteriorSampling (ReSample 핵심)
│   └── measurements.py      # 연산자들 (super_res, deblur, inpainting 등)
│
├── configs/
│   ├── tasks/               # 태스크별 설정 (SR, deblur, inpainting 등)
│   └── latent-diffusion/    # 모델 설정 (FFHQ, CelebA, ImageNet 등)
│
├── data/                    # 데이터로더 (FFHQ, CelebA)
├── scripts/                 # 유틸 스크립트
├── util/                    # img_utils, metrics 등
└── src/                     # 외부 라이브러리 (CLIP, taming-transformers)
```

---

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `sample_condition.py` | 메인 inference 스크립트 |
| `ldm/models/diffusion/ddim.py` | DDIM 샘플러 + ReSample 통합 |
| `ldm_inverse/measurements.py` | Forward operator 정의 (SR, blur, inpainting 등) |
| `ldm_inverse/condition_methods.py` | Conditioning 방법 (PosteriorSampling) |
| `configs/tasks/*.yaml` | 태스크별 파라미터 설정 |

---

## 태스크 설정 파일

| 태스크 | Config 파일 |
|--------|------------|
| Super Resolution | `configs/tasks/super_resolution_config.yaml` |
| Gaussian Deblur | `configs/tasks/gaussian_deblur_config.yaml` |
| Motion Deblur | `configs/tasks/motion_deblur_config.yaml` |
| Inpainting | `configs/tasks/inpainting_config.yaml` |
| Nonlinear Deblur | `configs/tasks/nonlinear_deblur_config.yaml` |

---

## 실행 방법

### 기본 실행
```bash
python3 sample_condition.py
```

### 커스텀 실행
```bash
python3 sample_condition.py \
    --ldm_config configs/latent-diffusion/ffhq-ldm-vq-4.yaml \
    --diffusion_config models/ldm/model.ckpt \
    --task_config configs/tasks/super_resolution_config.yaml \
    --ddim_steps 500 \
    --save_dir ./results
```

### 기본값
- `--ldm_config`: `configs/latent-diffusion/ffhq-ldm-vq-4.yaml`
- `--diffusion_config`: `models/ldm/model.ckpt`
- `--task_config`: `configs/tasks/gaussian_deblur_config.yaml`
- `--ddim_steps`: 500
- `--save_dir`: `./results`

---

## `sample_condition.py` 실행 흐름

### 1. 인자 파싱 (line 26-38)
```python
parser.add_argument('--ldm_config', default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml")
parser.add_argument('--diffusion_config', default="models/ldm/model.ckpt")
parser.add_argument('--task_config', default="configs/tasks/gaussian_deblur_config.yaml")
parser.add_argument('--ddim_steps', default=500)
parser.add_argument('--save_dir', default='./results')
```

### 2. Task Config 로드 (line 42)
```python
task_config = load_yaml(args.task_config)
```
→ operator, noise, conditioning 설정 읽음

### 3. 모델 로드 (line 50-51)
```python
model = get_model(args)        # LDM 모델 (UNet + VQ-VAE)
sampler = DDIMSampler(model)   # DDIM 샘플러 초기화
```
→ ffhq-ldm-vq-4.yaml 기반으로 모델 구성, model.ckpt 가중치 로드

### 4. Operator & Noise 준비 (line 54-57)
```python
operator = get_operator(...)   # e.g., GaussianBlur, SuperResolution 등
noiser = get_noise(...)        # e.g., Gaussian noise (sigma=0.01)
```
→ Forward model `y = Ax + n` 구성

### 5. Conditioning Method 준비 (line 60-63)
```python
cond_method = get_conditioning_method('ps', ...)  # PosteriorSampling
measurement_cond_fn = cond_method.conditioning
```
→ ReSample의 hard data consistency 적용할 함수

### 6. Sampler 함수 구성 (line 66-76)
```python
sample_fn = partial(sampler.posterior_sampler, ...)
```
→ DDIM + ReSample 통합 샘플링 함수 준비
→ latent shape: `[3, 64, 64]` (256×256 이미지의 4배 압축)

### 7. 출력 디렉토리 생성 (line 79-82)
```
./results/
├── input/    # 원본 이미지
├── label/    # degraded 이미지 (y)
├── recon/    # 복원된 이미지
└── progress/ # (중간 과정)
```

### 8. 데이터로더 준비 (line 85-89)
```python
dataset = get_dataset(name='ffhq', ...)
loader = get_dataloader(dataset, batch_size=1)
```

### 9. 이미지별 Inference 루프 (line 96-135)

각 이미지마다:

```
┌─────────────────────────────────────────────────────┐
│  ref_img (원본 256×256)                              │
│       ↓                                             │
│  y = operator.forward(ref_img)  # degradation       │
│       ↓                                             │
│  y_n = noiser(y)                # + noise 추가      │
│       ↓                                             │
│  samples_ddim = sample_fn(measurement=y_n)          │
│       │                                             │
│       └─→ DDIM reverse + ReSample (500 steps)       │
│           latent space에서 posterior sampling       │
│       ↓                                             │
│  x_recon = model.decode_first_stage(samples_ddim)   │
│       │                                             │
│       └─→ latent (64×64) → image (256×256)          │
│       ↓                                             │
│  저장: input/, label/, recon/                        │
│  출력: PSNR 계산                                     │
└─────────────────────────────────────────────────────┘
```

---

## 실행 흐름 요약

| 순서 | 단계 | 내용 |
|:---:|------|------|
| 1 | Config | 인자 파싱 + task config 로드 |
| 2 | Model | LDM 모델 + DDIM 샘플러 로드 |
| 3 | Setup | Operator, Noise, Conditioning 준비 |
| 4 | Data | 데이터셋 + 디렉토리 준비 |
| 5 | Loop | 이미지마다: degrade → sample → decode → save |

---

## 등록 패턴

코드베이스는 decorator 기반 등록 패턴 사용:

- `@register_operator()` → `measurements.py`에서 연산자 등록
- `@register_dataset()` → `dataloader.py`에서 데이터셋 등록
- `@register_conditioning_method()` → `condition_methods.py`에서 컨디셔닝 등록

---

## Optimization 종료 조건

ReSample은 두 가지 optimization 함수를 사용하며, 각각 **`eps`** (tolerance error) 기반으로 종료됩니다.

**파일 위치**: `ldm/models/diffusion/ddim.py`

### 1. `pixel_optimization` (line 337-370)

```python
def pixel_optimization(self, measurement, x_prime, operator_fn, eps=1e-3, max_iters=2000):
```

**종료 조건** (line 367-368):
```python
if measurement_loss < eps**2:  # eps² = 1e-6
    break
```

### 2. `latent_optimization` (line 373-432)

```python
def latent_optimization(self, measurement, z_init, operator_fn, eps=1e-3, max_iters=500, lr=None):
```

**종료 조건 2가지**:

1. **Loss plateau 감지 (Sliding Window 방식)** (line 419-426):
   - 처음 200 iteration까지는 loss 기록 (window 채우기)
   - 200 iteration 이후, **200 step 전 loss**와 **현재 loss** 비교
   - 현재 loss가 더 크면 → 수렴 실패로 판단 → 종료
   - 현재 loss가 더 작으면 → window를 한 칸 밀고 계속 진행
   ```python
   if itr < 200:
       losses.append(cur_loss)
   else:
       losses.append(cur_loss)
       if losses[0] < cur_loss:  # 200 step 전보다 loss 증가 → 종료
           break
       else:
           losses.pop(0)  # sliding window: 가장 오래된 loss 제거
   ```

2. **Threshold 도달** (line 428-429):
   ```python
   if cur_loss < eps**2:  # eps² = 1e-6
       break
   ```

### 요약 테이블

| 함수 | eps | threshold (eps²) | max_iters |
|------|-----|------------------|-----------|
| `pixel_optimization` | 1e-3 | **1e-6** | 2000 |
| `latent_optimization` | 1e-3 | **1e-6** | 500 |

> 코드 주석: "needs tuning according to noise level for early stopping" - noise level에 따라 조정 필요할 수 있음.

---

## 모델 다운로드 (사전 준비)

```bash
# LDM 모델
mkdir -p models/ldm
wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P ./models/ldm
unzip models/ldm/ffhq.zip -d ./models/ldm

# VQ-VAE autoencoder
mkdir -p models/first_stage_models/vq-f4
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip -P ./models/first_stage_models/vq-f4
unzip models/first_stage_models/vq-f4/vq-f4.zip -d ./models/first_stage_models/vq-f4
```
