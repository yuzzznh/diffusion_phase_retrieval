# DAPS 코드 분석 문서

## 프로젝트 개요

**DAPS** (Decoupled Annealing Posterior Sampling)는 CVPR 2025 Oral 논문 프로젝트로, 확산 모델(Diffusion Model)을 이용한 역문제(Inverse Problem) 해결 방법이다.

---

## 디렉토리 구조

```
DAPS/
├── posterior_sample.py     # 메인 실행 스크립트 (진입점)
├── sampler.py              # DAPS / LatentDAPS 샘플러 구현
├── eval.py                 # 평가 메트릭 (PSNR, SSIM, LPIPS)
├── data.py                 # 데이터셋 관리
│
├── configs/                # Hydra 설정 파일들
│   ├── data/               # 데이터셋 설정
│   ├── model/              # 모델 설정 (DDPM, LDM, SD)
│   ├── sampler/            # 샘플러 설정
│   └── task/               # 역문제 태스크 설정 (8가지)
│
├── cores/                  # 핵심 알고리즘
│   ├── scheduler.py        # 확산 스케줄러 (EDM, VP)
│   ├── mcmc.py             # MCMC 샘플러 (Langevin, HMC, MH)
│   └── trajectory.py       # 샘플링 궤적 기록
│
├── model/                  # 확산 모델 구현
│   ├── ddpm/               # DDPM 모델
│   ├── edm/                # EDM 모델
│   └── ldm/                # Latent Diffusion Model (VAE 포함)
│
├── forward_operator/       # 역문제 연산자 (8가지)
│   ├── resizer.py          # Super-resolution
│   ├── bkse/               # 비선형 블러
│   └── motionblur/         # 모션 블러
│
├── commands/               # 실행 스크립트
│   ├── ldm_ffhq.sh
│   └── ldm_imagenet.sh
│
└── dataset/                # 샘플 데이터 (demo/test)
```

### 지원 태스크 (역문제)
- `phase_retrieval`, `down_sampling` (SR x4), `gaussian_blur`, `motion_blur`
- `nonlinear_blur`, `inpainting`, `inpainting_rand`, `hdr`

### 지원 모델
- **DDPM** (픽셀 공간)
- **LDM** (잠재 공간, VAE 기반)
- **Stable Diffusion** (프리트레인)

---

## LDM FFHQ Phase Retrieval 실행 흐름 (시간 순서)

### 명령어
```bash
python posterior_sample.py \
    +data=test-ffhq \
    +model=ffhq256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm \
    save_dir=results/ldm/ffhq \
    num_runs=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    batch_size=10 \
    name=phase_retrieval \
    gpu=0
```

---

### 1단계: 초기화 (`posterior_sample.py:163-194`)

```
┌─────────────────────────────────────────────────────────────
│ 1. Hydra 설정 로딩                                           
│    - default.yaml + test-ffhq + ffhq256ldm + phase_retrieval
│    - latent_edm_daps sampler 설정                            
├─────────────────────────────────────────────────────────────
│ 2. Random Seed 고정 (재현성)                                  
├─────────────────────────────────────────────────────────────
│ 3. 데이터셋 로딩: dataset/test-ffhq (100장, 256x256)          
│    -> images: [100, 3, 256, 256] 범위 [-1, 1]                
├─────────────────────────────────────────────────────────────
│ 4. Forward Operator 생성: PhaseRetrieval(oversample=2.0)     
│    - pad = (2.0/8.0)*256 = 64 픽셀 패딩                       
│    - 이미지 -> FFT -> 진폭(amplitude)만 추출 (위상 정보 손실)   
├─────────────────────────────────────────────────────────────
│ 5. 측정값 생성: y = |FFT(x)| + noise (sigma=0.05)            
│    -> y: [100, 3, 384, 384] (패딩 포함된 FFT 진폭)            
├─────────────────────────────────────────────────────────────
│ 6. LatentDAPS Sampler 초기화                                 
│    - annealing_scheduler: 50 steps, sigma: 10 -> 0.1         
│    - diffusion_scheduler: 2 steps (ODE)                      
│    - MCMC sampler: HMC, 65 steps, lr=2.1e-5                  
├─────────────────────────────────────────────────────────────
│ 7. LDM 모델 로딩 (ffhq256ldm)                                
│    - VAE Encoder/Decoder + UNet Denoiser                     
│    - 체크포인트 다운로드 (처음 실행시)                          
├─────────────────────────────────────────────────────────────
│ 8. Evaluator 설정 (PSNR, SSIM, LPIPS)                        
└─────────────────────────────────────────────────────────────
```

---

### 2단계: 메인 샘플링 루프 (4 runs x 10 batches)

```
for run in range(4):  # num_runs=4
    │
    ├── x_start 생성: z ~ N(0, sigma_max^2 * I)  # 잠재 공간에서 랜덤 시작점
    │   -> z_start: [100, 4, 32, 32] (LDM 잠재 공간)
    │
    └── sample_in_batch():
        │
        for batch in range(10):  # 100장 / batch_size=10
            │
            ├── cur_z_start: [10, 4, 32, 32]
            ├── cur_y: [10, 3, 384, 384]
            │
            └── sampler.sample() 호출
```

---

### 3단계: LatentDAPS.sample() 내부 (핵심 알고리즘)

```python
# sampler.py:144-207 (LatentDAPS.sample)

for step in range(49):  # annealing 50-1 steps
    │
    │  sigma = scheduler.sigma_steps[step]  # 10.0 -> ... -> 0.1
    │
    ├── [1] Reverse Diffusion (노이즈 제거)
    │   ┌────────────────────────────────────────────────
    │   │ diffusion_scheduler = EDM(sigma_max=sigma, 2 steps)
    │   │ sampler = DiffusionPFODE(model, scheduler)
    │   │
    │   │ z0hat = sampler.sample(zt)  # ODE 2스텝 적분
    │   │   - zt (노이즈 낀 잠재) -> z0hat (깨끗한 잠재 추정)
    │   │
    │   │ x0hat = model.decode(z0hat)  # VAE 디코딩
    │   │   - [10,4,32,32] -> [10,3,256,256]
    │   └────────────────────────────────────────────────
    │
    ├── [2] MCMC Update (측정값에 맞추기)
    │   ┌────────────────────────────────────────────────
    │   │ LatentWrapper로 연산자 래핑:
    │   │   wrapped_op(z) = PhaseRetrieval(decode(z))
    │   │
    │   │ for mcmc_step in range(65):  # HMC 65 steps
    │   │   │
    │   │   │  # Score function 계산:
    │   │   │  score = data_term + xt_term + prior_term
    │   │   │
    │   │   │  data_term = -grad_z ||A(decode(z)) - y||^2 / tau^2
    │   │   │  xt_term = (zt - z) / sigma^2
    │   │   │  prior_term = (z0hat - zt) / sigma^2
    │   │   │
    │   │   │  # HMC update:
    │   │   │  velocity = 0.41*v + sqrt(lr)*score + noise
    │   │   │  z = z + velocity * sqrt(lr)
    │   │   │
    │   │
    │   │ z0y = 최종 z (측정값에 맞춰진 잠재)
    │   │ x0y = model.decode(z0y)
    │   └────────────────────────────────────────────────
    │
    ├── [3] Forward Diffusion (다음 스텝을 위해 노이즈 추가)
    │   ┌────────────────────────────────────────────────
    │   │ if step < 48:
    │   │   sigma_next = sigma_steps[step + 1]
    │   │   zt = z0y + randn() * sigma_next  # 노이즈 추가
    │   │ else:
    │   │   zt = z0y  # 마지막 스텝은 노이즈 없이
    │   │
    │   │ xt = model.decode(zt)  # 시각화용 디코딩
    │   └────────────────────────────────────────────────
    │
    └── [4] Evaluation & Recording
        ┌────────────────────────────────────────────────
        │ PSNR, SSIM, LPIPS 계산 (GT vs x0hat, x0y)
        │ Progress bar 업데이트:
        │   "49/49 [15:45] x0hat_psnr: 25.32 x0y_psnr: 26.18"
        │ Trajectory 저장 (if record=True)
        └────────────────────────────────────────────────
```

---

### Sigma 스케줄 (Annealing)

```
step:    0     10     20     30     40     49
sigma: 10.0 -> 3.2 -> 1.0 -> 0.32 -> 0.13 -> 0.1
       |      |      |      |      |      |
       v      v      v      v      v      v
      [매우 노이즈] ────────────────────-> [거의 깨끗]
```
- `poly-7` 타임스텝: 초반에 빠르게 감소, 후반에 천천히

---

### 4단계: 결과 저장

```
┌─────────────────────────────────────────────────────────────
│ 1. 샘플 이미지 저장
│    results/ldm/ffhq/phase_retrieval/samples/
│    -> 00000_run0000.png ~ 00099_run0003.png
├─────────────────────────────────────────────────────────────
│ 2. 평가 결과 계산 & 저장
│    - evaluator.report(): 각 run별 PSNR/SSIM/LPIPS 통계
│    - eval.md, metrics.json
├─────────────────────────────────────────────────────────────
│ 3. Grid 이미지 저장
│    - grid_results.png: GT, 측정값, 4runs 결과 비교
├─────────────────────────────────────────────────────────────
│ 4. FID 계산 (if eval_fid=True)
│    - 4 runs 중 best PSNR 샘플 선택
│    - calculate_fid(real, fake)
└─────────────────────────────────────────────────────────────
```

---

## 시간 분석

| 구간 | 소요 시간 (A100) |
|------|-----------------|
| 1 batch (10장) x 49 annealing steps | ~15분 45초 |
| 1 Run (10 batches) | ~2시간 37분 |
| 4 Runs (phase_retrieval) | **~10.6시간** |

**병목 지점**: MCMC 65 steps x VAE encode/decode 반복
- 각 annealing step마다 MCMC가 65번 gradient 계산
- LatentWrapper가 매번 `decode(z)` 호출 -> FFT -> loss 계산

---

## 핵심 파일 참조

| 파일 | 역할 | 주요 라인 |
|------|------|----------|
| `posterior_sample.py` | 메인 진입점 | `main()`: 163-283 |
| `sampler.py` | DAPS/LatentDAPS 샘플러 | `LatentDAPS.sample()`: 144-207 |
| `cores/mcmc.py` | MCMC 샘플러 (HMC/Langevin) | `MCMCSampler.sample()`: 123-164 |
| `cores/scheduler.py` | 노이즈 스케줄러 | `get_diffusion_scheduler()` |
| `forward_operator/__init__.py` | 역문제 연산자 | `PhaseRetrieval`: 300-314 |
| `configs/task/phase_retrieval.yaml` | 태스크별 MCMC 설정 | LDM: HMC 65 steps |

---

## 컴포넌트 관계도

```
posterior_sample.py (메인)
    │
    ├──> sampler.py (DAPS 샘플러)
    │   ├──> cores/scheduler.py (노이즈 스케줄)
    │   └──> cores/mcmc.py (MCMC 업데이트)
    │       ├──> forward_operator (측정 연산자)
    │       └──> model (확산 모델)
    │
    ├──> model/__init__.py (모델 로드)
    │   ├──> model/ddpm/ (DDPM)
    │   ├──> model/edm/ (EDM)
    │   └──> model/ldm/ (Latent DM)
    │
    ├──> forward_operator/__init__.py (연산자 로드)
    │   └──> 8가지 역문제 연산자
    │
    ├──> data.py (데이터셋 로드)
    │   └──> dataset/ (샘플 데이터)
    │
    └──> eval.py (평가)
        └──> PSNR, SSIM, LPIPS, FID
```

---

## HMC 구현 분석 (LatentDAPS)

### 분석 목표
논문 Eq. 11의 $p^{(j+1/2)}$ (half-step momentum)이 실제로 구현되어 있는지 확인.

### 핵심 코드 스니펫

**1. 메인 샘플링 루프** (`cores/mcmc.py:151-155`):
```python
for _ in pbar:
    cur_score, fitting_loss = self.score_fn(x, x0hat, model, xt, operator, measurement, sigma)
    epsilon = torch.randn_like(x)
    x = self.mc_update(x, cur_score, lr, epsilon)
```

**2. HMC 업데이트 함수** (`cores/mcmc.py:85-95`):
```python
def mc_update(self, x, cur_score, lr, epsilon):
    if self.mc_algo == 'langevin':
        x_new = x + lr * cur_score + np.sqrt(2 * lr) * epsilon
    elif self.mc_algo == 'hmc':  # (damping) hamiltonian monte carlo
        step_size = np.sqrt(lr)
        self.velocity = self.momentum * self.velocity + step_size * cur_score + np.sqrt(2 * (1 - self.momentum)) * epsilon
        x_new = x + self.velocity * step_size
    else:
        raise NotImplementedError
    return x_new
```

### 결론: **Case B (Simplified/Euler)** 에 해당

| 비교 항목 | Standard Leapfrog (Case A) | DAPS 구현 (Case B) |
|----------|---------------------------|-------------------|
| Momentum 1st update | `p += (η/2) * ∇log p(x)` | `v = γv + η·score + noise` |
| Position update | `x += η * p` | `x += v * η` |
| Momentum 2nd update | `p += (η/2) * ∇log p(x_new)` | **없음** |

**DAPS의 실제 업데이트 순서 (j번째 iteration):**
```
1. score 계산: ∇log p(x^j | x_t, y)

2. momentum 전체 업데이트 (half-step 아님):
   v^{j+1} = γ·v^j + η·score + √(2(1-γ))·ε

3. position 업데이트:
   x^{j+1} = x^j + v^{j+1}·η
```

### 논문 Eq. 10, 11과의 대응

| 논문 | 코드 | 설명 |
|-----|------|-----|
| Eq. 10: $p^{(j+1/2)} = \gamma p^{(j)} + \eta \nabla \log p(...) + \sqrt{2(1-\gamma)}\epsilon$ | `self.velocity = self.momentum * self.velocity + step_size * cur_score + np.sqrt(2 * (1 - self.momentum)) * epsilon` | momentum 전체 업데이트 |
| Eq. 11: $z_0^{(j+1)} = z_0^{(j)} + \eta p^{(j+1/2)}$ | `x_new = x + self.velocity * step_size` | position 업데이트 |

### 핵심 발견

1. **$p^{(j+1/2)}$ 표기는 오해의 소지가 있음**: 논문에서 "half-step"을 암시하는 표기를 사용했지만, 실제 구현은 **full-step momentum update**임.

2. **Standard Leapfrog가 아님**: Leapfrog integrator의 특징인 "momentum half → position full → momentum half" 순서가 구현되어 있지 않음.

3. **실제 알고리즘**: **Underdamped Langevin Dynamics** (= Damped HMC)
   - 순수 HMC (γ=1)와 Langevin (γ=0)의 중간
   - LDM phase_retrieval 기준 γ=0.41 사용

### Damping Factor (momentum) 역할

```python
self.velocity = self.momentum * self.velocity + step_size * cur_score + np.sqrt(2 * (1 - self.momentum)) * epsilon
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#               감쇠된 이전 속도 (γ=0.41)           score 방향으로 가속      thermostating noise (온도 유지)
```

| γ 값 | 동작 |
|------|------|
| γ = 1 | 순수 HMC (노이즈 없음, 이전 속도 완전 보존) |
| γ = 0 | Overdamped Langevin (이전 속도 무시) |
| 0 < γ < 1 | Underdamped Langevin (둘의 중간) |

### LDM Phase Retrieval 설정 (`configs/task/phase_retrieval.yaml`)

```yaml
ldm:
  mcmc_sampler_config:
    num_steps: 65        # MCMC 반복 횟수
    lr: 2.1e-5           # 학습률
    momentum: 0.41       # damping factor (γ)
    mc_algo: hmc         # HMC 사용
    prior_solver: gaussian
```

---

## Damping Factor (Momentum) 상세 분석

### Q1: Timestep에 따라 상수 vs 변수?

**상수!** 코드에서 `self.momentum`은 초기화 시 한 번 설정되고, MCMC 샘플링 전체에서 동일하게 사용됨:

```python
# cores/mcmc.py:35
self.momentum = momentum  # 초기화 시 고정

# cores/mcmc.py:91 - 매 step 동일한 값 사용
self.velocity = self.momentum * self.velocity + ...
```

참고: `lr`은 annealing step에 따라 감소하지만 (`get_lr(ratio)`), momentum은 상수.

### Q2: LDM 설정 - 태스크별 momentum 값

**태스크마다 다름!** 하이퍼파라미터 튜닝으로 최적화된 값:

| Task | momentum (γ) | num_steps | lr |
|------|-------------|-----------|-----|
| **phase_retrieval** | **0.41** | 65 | 2.1e-5 |
| down_sampling | 0.86 | 24 | 1.35e-4 |
| gaussian_blur | 0.95 | 35 | 2.70e-6 |
| motion_blur | 0.45 | - | - |
| nonlinear_blur | 0.91 | - | - |
| inpainting | 0.74 | 15 | 9.02e-6 |
| inpainting_rand | 0.92 | - | - |
| hdr | 0.58 | - | - |

### Q3: 요약

| 질문 | 답변 |
|-----|------|
| 현재 설정 (phase_retrieval, LDM) | **γ = 0.41** |
| timestep에 따라 변함? | **아니오, 상수** |
| LDM 내에서 태스크별로 동일? | **아니오, 태스크마다 다름** (0.41 ~ 0.95) |

### 물리적 해석

```
γ = 0.41 (phase_retrieval)  -> 이전 속도 41% 유지, 59%는 새 정보
γ = 0.95 (gaussian_blur)    -> 이전 속도 95% 유지, 관성이 큼
γ = 0.003 (inpainting, pixel_hmc) -> 거의 Langevin과 동일
```

- **γ 높음** (0.9+): 관성이 커서 momentum이 오래 유지됨 -> 수렴 느림, 탐색 넓음
- **γ 낮음** (0.4-): score에 빠르게 반응 -> 수렴 빠름, 탐색 좁음

phase_retrieval은 비선형 문제라 γ=0.41로 비교적 낮게 설정하여 빠른 적응을 유도한 것으로 보임.
