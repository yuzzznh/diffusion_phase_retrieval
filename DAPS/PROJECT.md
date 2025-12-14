# My Project: LatentDAPS로 Langevin Dynamics sampling + TDP-style 탐색으로 0° 180° 찾기 + 맨 마지막 hard data consistency 적용

## 실험별 명령어

``` bash
# ============================================================
# GPU (CUDA) 명령어 - commands_gpu/ 폴더
# ============================================================
# 실험 0
bash commands_gpu/exp0_baseline.sh --1           # 1 image sanity check -> A10에서 150/150W 12292MiB 100% 15min. 하지만 로그엔 Peak Memory: 10116.54 MB 라고 기록됨.
bash commands_gpu/exp0_baseline.sh --10          # 10 images -> A10에서 2.5시간 걸릴 듯.
bash commands_gpu/exp0_baseline.sh --90          # 90 images (10~99) -> --10과 합쳐서 100개. A10에서 150/150W 12536MB 100% 15min per image 뜸. 다 돌리면 22.5시간 예상.
bash commands_gpu/exp0_baseline.sh --1 --10      # 1 + 10 images 순차 실행

# 실험 1~4
bash commands_gpu/exp1_repulsion.sh --1 --10 --90
bash commands_gpu/exp2_pruning.sh --1 --10 --90
bash commands_gpu/exp3_2particle.sh --10 --90     # (1 image 없음)
bash commands_gpu/exp4_optimization.sh --1 --10 --90

# 실험 5
bash commands_gpu/exp5_final.sh --imagenet        # ImageNet 100
bash commands_gpu/exp5_final.sh --ffhq            # FFHQ 100
bash commands_gpu/exp5_final.sh --imagenet --ffhq # 둘 다

# ============================================================
# 인자 없이 실행하면 사용법 출력:
# ============================================================
$ bash commands_gpu/exp0_baseline.sh
# 사용법: bash exp0_baseline.sh [--1] [--10] [--90]
# --1   : 1 image sanity check (이미지 0)
# --10  : 10 images main experiment (이미지 0~9)
# --90  : 90 images final eval (이미지 10~99, --10과 합쳐서 100개)

## 실험 진행 및 구현 과정 설계

### [데이터] imagenet 10장으로 method 비교, 마지막 eval은 ffhq imagenet 100장씩으로 하는걸 목표로, 여건 안되면 ffhq는 버리기 / 시드 고정 (이미 DAPS에서는 42)

### [실험 0] LatentDAPS 논문에 eval 데이터는 100 image에만 나와있으니까 비교를 위해 LatentDAPS(with Langevin Dynamic)의 imagenet first 10 image에 대한 phase retrieval 성능 측정.
- ~~단, 이때 image별로 전부 돌아간 뒤 다음 run이 실행되는 구조로 4 run이 구현돼있는데, 이후 실험들과의 원활한 비교를 위해 eval 명령어를 4 batch = 4 run 구조로 변경해야 함.~~
- ~~time logging: diffusion timestep T를 구간개수로 하여 **timestep별 소요 시간**을 측정. 이후 실험에서 pruning/optimization 시점 전후 시간 비교에 활용. sanity check 차원에서 1 image 4 sample 명령어로 먼저 테스트할 것.~~ → **완료**: `sampler.py`의 `LatentDAPS.sample()`에서 step별 시간 측정 (`self.timing_info`에 저장), `posterior_sample.py`에서 이미지별 timing 집계 후 `metrics.json`에 저장.
- ~~GPU VRAM logging: 실험 0에서는 phase 구분 없이 **전체 구간의 peak VRAM**만 측정. `torch.cuda.max_memory_allocated()` 활용.~~ → **완료**: `posterior_sample.py`에서 `torch.cuda.reset_peak_memory_stats()` 후 `torch.cuda.max_memory_allocated()` 측정, `metrics.json`의 `metadata.gpu.peak_vram_mb`에 저장. (phase별 구간 분리는 실험 2, 4에서 pruning/optimization 추가 시 구현)
- ~~명령어 자동기록 메커니즘이 이미 있는걸로 아는데, 어떤 메커니즘인지 파악하고, 우리 실험 0~5의 각종 argument 세팅이 잘 기록되도록 코드를 수정할 것.~~ → **Hydra 기반 config 자동기록 확인 완료**: `posterior_sample.py`에서 `OmegaConf.to_container(args)`를 통해 모든 config가 merge된 최종 결과를 `results/<name>/config.yaml`에 자동 저장함. sh 명령어에서 override한 모든 argument가 기록됨.

### [실험 1] 4-Particle Full Run (Repulsion vs. Independence) → **구현 완료**

#### Sampler 코드 분석 (준비)
| 항목         | 이 코드                  |
|--------------|--------------------------|
| 파라미터화   | EDM σ (sigma)            |
| 예측 타겟    | x₀-prediction (denoiser) |
| ε-prediction | ❌ 아님                  |

- Forward diffusion: `x_t = x_0 + σ·ε` (EDM 형태)
- `DiffusionPFODE.derivative()`에서 `model.score()` 호출 → score = (D(x;σ) - x) / σ² 변환 사용
- 변수명 `x0hat`, `z0hat`이 x₀ 예측임을 명시

#### 구현 완료 사항 (RLSD → LatentDAPS 이식)
- ~~`repulsion.py` 모듈 생성~~: **완료**
  - `DinoFeatureExtractor`: DINO-ViT 모델 lazy loading 및 feature 추출 (`dino_vits16`, RLSD와 동일)
  - `compute_repulsion_gradient()`: SVGD-style repulsion gradient 계산 (RBF kernel + median heuristic bandwidth)
  - `RepulsionModule`: High-level repulsion 관리 (scale decay, metrics tracking)
  - **N=2 버그 수정**: `h = median(dist)^2 / max(log(N), eps)` 사용 (RLSD의 `log(N-1)` 대신)
- ~~`DiffusionPFODE` 수정~~: **완료**
  - `set_repulsion(repulsion, scale)` 메서드 추가
  - `derivative()`에서 score-level injection: `score' = score + scale * repulsion`
- ~~`LatentDAPS.sample()` 수정~~: **완료**
  - Repulsion module 초기화 및 각 annealing step에서 repulsion 계산
  - pfode에 repulsion 전달 및 metrics 수집
- ~~Config 업데이트~~: **완료**
  - `repulsion_scale`, `repulsion_sigma_break`, `repulsion_schedule`, `repulsion_dino_model`
- ~~Shell scripts 업데이트~~: **완료**
  - `exp1_repulsion.sh`, `exp3_2particle.sh`에 새 repulsion 파라미터 반영

#### Hyperparameter: `repulsion_scale`과 RLSD `gamma`의 관계

**RLSD 구현** (noise prediction space):
```python
noise_pred = ε - γ · √(1-α_t) · ∇Φ   # γ = gamma (50~150)
```

**우리 구현** (score space):
```python
score' = score + λ · ∇Φ              # λ = repulsion_scale
```

**핵심**: Score 공간에서 `+λ·r`을 하면, ε 공간에서는 이미 `σ·λ`가 곱해진 효과가 생긴다 (EDM score-ε 변환 관계).
따라서 **σ를 추가로 곱하지 않고**, `repulsion_scale`을 RLSD의 `gamma` 수준으로 올리면 동형(equivalent)하게 동작한다.

| RLSD gamma | 우리 repulsion_scale | 비고 |
|------------|---------------------|------|
| 30 | 30 | Phase retrieval 기본값 |
| 50 | 50 | HDR 등 다른 task |
| 100~150 | 100~150 | 강한 repulsion |

**결론**: `repulsion_scale=0.5~1.0`은 RLSD 대비 너무 약함. **30, 50, 100** 등으로 실험 필요.

#### Hyperparameter: `repulsion_sigma_break` 활성 구간

**우리 설정** (EDM sigma 기준):
```
sigma_max: 10
sigma_min: 0.001
repulsion_sigma_break: 1.0 (default)
```

**Repulsion 활성 구간**:
```
sigma:  10 -------- 1.0 -------- 0.001
         [  ON  ]   |   [  OFF  ]
                    ↑
              sigma_break
```

| sigma 범위 | Repulsion | 비고 |
|-----------|-----------|------|
| 1.0 ~ 10 | ✅ ON | 전체 50 step 중 ~30 step |
| 0.001 ~ 1.0 | ❌ OFF | 마지막 ~20 step |

**RLSD와 비교**: RLSD는 `sigma_break=999` (DDPM timestep)로 **거의 전 구간 ON**.
우리도 더 오래 켜두려면 `sigma_break`를 낮추면 됨 (예: 0.1 또는 0.01).

#### Shell Script 수정 (2025-12-14) - RLSD 유사 세팅

**배경**: 이전 실험(scale=0.5, 1.0)에서 repulsion 효과가 거의 없었음 (pairwise distance ~32로 동일).
RLSD와 최대한 유사하게 맞추기 위해 hyperparameter 및 주석 수정.

**수정 내용** (`exp1_repulsion.sh` ~ `exp5_final.sh` 전체):
```bash
REPULSION_SCALE=50            # RLSD gamma=50 (HDR task) 기준
REPULSION_SIGMA_BREAK=1.0     # σ < 1.0에서 OFF
REPULSION_SCHEDULE="constant" # 추가 decay 없음
```

**주석 수정 (엄밀성 강화)**:
- ~~"RLSD-동형 세팅"~~ → "RLSD gamma 기준" (완전 동형은 아님)
- ~~"자동 decay"~~ → "σ-decay는 score→ε 변환에서 자연 발생"
  - 정확히는: EDM score-ε 변환 관계에 의해 ε 관점에서 step별 σ가 곱해지는 효과가 나타남
  - 이것이 RLSD의 `gamma × sqrt(1-α_t)` (≈ `gamma × σ`)와 유사해지는 원리
- `sigma_break=1.0`은 σ ∈ [1,10] 구간만 ON (~30/50 step)
  - RLSD는 보통 더 오래 켜둠. 더 긴 ON 원하면 0.1 또는 0.01로 낮추기

**의도**:
1. `schedule=constant`: 추가 decay 제거 → RLSD처럼 gamma 상수 유지
2. `scale=50`: RLSD HDR task 기준값으로 점프 (0.5~1.0에서 효과 없었음)
3. 주석에서 "동형"이라는 과장 표현 제거, 정확한 메커니즘 설명

#### 디버깅 로깅 및 Assert 추가 (2025-12-14) → **완료**

**배경**: scale=50으로 올려도 repulsion이 실제로 적용되는지 확신이 없어서, assert와 상세 로깅 추가.

**구현 내용**:

1. **Assert 추가** (`cores/scheduler.py` - `DiffusionPFODE.derivative()`):
   - ON 상태: `repulsion is not None`, `scale > 0`, `isfinite(score)`, `isfinite(repulsion)`, `shape 일치`
   - OFF 상태: `scale == 0`
   - Warning: `ratio > 10`이면 폭주 위험 경고 출력

2. **repulsion.jsonl 로깅** (metrics.json과 별개):
   - 저장 위치: `results/<run_name>/repulsion.jsonl`
   - 샘플링 규칙: step<50은 매 5 step, step>=50은 매 25 step, 항상 {0,1,2,5,10} 포함
   - 필드:
     ```json
     {"image_idx": 0, "step": 0, "sigma": 10.0, "repulsion_on": true,
      "repulsion_scale_used": 50.0, "score_base_norm": 1234.5,
      "repulsion_norm": 0.123, "scaled_repulsion_norm": 6.15,
      "ratio_scaled_to_score": 0.005, "repulsion_cleared": false,
      "mean_pairwise_dino_dist": 32.1, "weights_mean": 0.45,
      "weights_max": 0.98, "weights_nonzero_frac": 1.0, "repulsion_time_sec": 0.23}
     ```

3. **수정된 파일**:
   - `cores/scheduler.py`: assert + `_last_score_info` + `begin/end_annealing_step()`
   - `repulsion.py`: `weights_mean`, `weights_max`, `weights_nonzero_frac` 추가
   - `sampler.py`: `repulsion_debug_logs` 수집, 샘플링 규칙 적용
   - `posterior_sample.py`: `repulsion.jsonl` 저장 로직

#### 결과 디렉토리 정리 (2025-12-14)

이전 scale=0.1 sanity check 결과를 보존하기 위해 디렉토리 이름 변경:
```
results/exp1_repulsion/imagenet_1img/exp1_sanity_check → exp1_sanity_check_scale0.1
results/exp3_2particle/imagenet_1img/exp3_sanity_check → exp3_sanity_check_scale0.1
```

#### Sanity Check 실행 (2025-12-14) - scale=50

```bash
# Exp1 (4-particle) sanity check
bash commands_gpu/exp1_repulsion.sh --1
# → results/exp1_repulsion/imagenet_1img/exp1_sanity_check (scale=50)

# Exp3 (2-particle) sanity check
bash commands_gpu/exp3_2particle.sh --1
# → results/exp3_2particle/imagenet_1img/exp3_sanity_check (scale=50)
```

**확인할 것**:
- `repulsion.jsonl`에서 `ratio_scaled_to_score`가 0이 아닌 값인지
- 초반 step에서 `repulsion_on=true`이고 `repulsion_scale_used=50`인지
- assert 통과 여부 (에러 없이 완료되면 OK)
- Exp3 (N=2)에서 bandwidth 버그 수정이 정상 작동하는지 (NaN/crash 없음)

* 설정: 입자 4개, 처음부터 끝까지($T \to 0$) 유지.
* 비교: Ours (Repulsion ON) vs. DAPS Baseline (Repulsion OFF, Independent)
* 확인할 지표:
    * Max PSNR: 4개 중 가장 잘 나온 놈의 점수. (우리가 더 높거나 비슷해야 함)
    * Std / Mode Coverage: 4개가 0도, 180도, 혹은 다른 Local Minima로 얼마나 잘 흩어졌는가?
        * DAPS: 운 나쁘면 4개 다 0도로 쏠림.
        * Ours: 0도, 180도 골고루 나와야 성공.
* 기대 결론: "단순히 여러 번 돌리는 것(DAPS)보다, 서로 밀어내며 돌리는 것(Ours)이 정답(Global Optima)을 찾을 확률(Success Rate)이 훨씬 높다."
* 여기에선 particle guidance를 잘 코딩하고 repulsion 강도 등 hyperparameter 값을 적절하게 설정하는 것이 관건. 
* 이에 대한 sanity check 및 가장 기본적인 경향성 체크를 위해 1 image 4 (particle) run 명령어를 적극 활용한 뒤 디버깅 완료된 코드베이스에서 합리적인 hyperparameter set으로 10 image 실험을 돌리자.
⚠️ 주의할 점 (Manifold):
* Repulsion을 위해 z.grad를 조작할 때, 너무 강하게 밀면 Latent가 학습된 분포 밖(Off-manifold)으로 튕겨 나가 이미지가 깨질 수 있습니다.
* 초반에는 강하게, 후반($t \to 0$)으로 갈수록 0에 수렴하도록 Decay Schedule을 꼭 넣으세요.
💡 팁 (Sanity Check):
* 1 Image 실험 시, 4개의 Latent Vector 간의 **평균 거리(Average Pairwise Distance)**를 매 스텝 로깅하세요.
* Baseline(독립 실행)보다 이 거리가 확실히 커야 성공입니다.

### [실험 2] 4 → 2 Pruning (Efficiency Verification)
* 설정: 4개로 시작 $\to$ $t=200$에서 2개로 압축 $\to$ 끝.
* 비교: Exp 2 (Pruning) vs. Exp 1 (Full Run)
* 확인할 지표:
    * Max PSNR 유지 여부: Exp 1과 결과가 거의 똑같아야 함. (떨어지면 Pruning 로직 실패)
    * Time / Memory: 시간이 얼마나 단축되었는가? (이게 논문의 세일즈 포인트)
* 기대 결론: "초반 탐색 후 가망 없는 놈을 버려도 성능 손실은 없다. 즉, Exp 1처럼 끝까지 4개를 끌고 가는 건 자원 낭비다."
* pruning 임계값 및 timestep과 같은 hyperparameter 값을 적절하게 설정하는 것이 관건. 이에 대한 sanity check 및 가장 기본적인 경향성 체크를 위해 1 image 4 (particle) run 명령어를 적극 활용한 뒤 디버깅 완료된 코드베이스에서 합리적인 hyperparameter set으로 10 image 실험을 돌리자.
⚠️ 주의할 점 (Indexing Hell):
* 배치 사이즈가 4에서 2로 줄어들 때, z뿐만 아니라 optimizer의 state, scheduler의 step, measurement y 등 관련된 모든 변수를 같이 줄여야(Slicing) 에러가 안 납니다.
* 헷갈리면 그냥 4개 유지를 하되, 탈락한 2개에 대해서는 Gradient 계산을 끄는 마스킹(Masking) 처리만 해도 연산량 이득은 증명할 수 있습니다. (메모리 이득은 없지만 구현은 쉬움) $\rightarrow$ 하지만 진짜 메모리 이득을 위해 Slicing을 추천합니다.
📊 GPU VRAM 측정 구간 분리 (구현 필요):
* Pruning 추가 시, VRAM 측정을 **pruning 전/후 두 구간**으로 쪼개야 함.
* `torch.cuda.reset_peak_memory_stats()`를 pruning 시점에 호출하여 각 구간별 peak를 독립 측정.
* metrics.json에 `vram.pre_pruning_peak_mb`, `vram.post_pruning_peak_mb` 형태로 기록.

### [실험 3] 2-Particle Full Run (Justification for '4')
* 설정: 처음부터 2개만 띄워서 끝까지($T \to 0$) 유지.
* 비교: Exp 2 (4 $\to$ 2 Pruning) vs. Exp 3 (Just 2)
* 핵심 질문: "그냥 처음부터 2개만 돌리면 안 돼? 굳이 4개로 시작해서 줄여야 해?" (리뷰어들이 무조건 물어볼 질문)
* 확인할 지표:
    * Success Rate (성공률): Exp 3은 가끔 둘 다 실패(Local Minima)하는 경우가 생겨야 함. 반면 Exp 2는 4개 중 골랐으므로 성공률이 더 높아야 함.
* 기대 결론: "처음부터 2개만 쓰면(Exp 3) 불안정하다. 4개로 넓게 탐색하고 줄이는 것(Exp 2)이 안정성(Stability) 측면에서 훨씬 우월하다."
* 전략: 여기서 실패 사례(0도/180도 모두 못 찾고 Local Minima 빠짐)가 단 하나라도 나오면 님의 논리는 완벽해집니다.
* 사실 여기선 앞선 실험들에서 추가되는 hyperparameter가 없으며, sample들 중 실패하는 것들의 비율을 제대로 재는 것이 관건이므로 1 image 실험이 의미가 없다. 최소한 10 image, 여건이 되면 100 image 실험을 돌리자.

### [실험 4] 실험 1~3 중 가장 잘 나온 세팅에 대해 ReSample의 hard data consistency in latent space optimization을 돌리자
- 정확한 횟수 및 기준은 ReSample 공식 레포의 구현에서 실제 몇 번의 optimization이 이루어지는지를 참고해서 결정하자. hyperparameter 튜닝에 1 image 실험을 활용하자.
- optimization 횟수 및 소요시간을 보고하자. batch element 간 optimization 및 termination이 independent해야 함에 유의하자 (ReSample 공식 레포는 그렇지 않았음!)
📊 GPU VRAM 측정 구간 분리 (구현 필요):
* Optimization 추가 시, VRAM 측정을 **optimization 전/후 두 구간**으로 분리해야 함.
* `torch.cuda.reset_peak_memory_stats()`를 optimization 시작 시점에 호출하여 각 구간별 peak를 독립 측정.
* metrics.json에 `vram.pre_optimization_peak_mb`, `vram.optimization_peak_mb` 형태로 기록.
* 만약 실험 2의 pruning과 함께 사용 시, 3구간으로 분리: `pre_pruning`, `post_pruning_pre_optimization`, `optimization`.

### [실험 5] 결과를 보고 제일 잘 나온 세팅에 대해 100 image 실험을 돌리자. 
- 이후 particle guidance, 유전알고리즘적 관점의 설명, phase retrieval with 2 oversampling이라는 2-mode task 자체의 특수성, DAPS와 ReSample과의 실행시간 및 GPU 및 연산량 비교
- 몇 particle이 필요했고 pruning 및 hard data consistency optimization이 얼마나 도움이 됐는지에 대한 보고
- 가능하면 ffhq 100 image에 대해서도 eval을 진행하여 table 만들기.
* FFHQ 100장: 시간이 남으면 돌리되, 안 되면 "ImageNet이 더 상위 호환(Superset) 문제이므로 생략했다"고 해도 무방합니다.
* 스토리텔링: "유전 알고리즘적 관점"과 "TDP의 Planning 관점"을 섞어서 설명하면, 단순한 엔지니어링이 아니라 **'생성 모델을 위한 탐색 알고리즘 제안'**으로 격상될 수 있습니다.



## 구현 예시. 구체적인 particle 수와 pruning 여부 등은 실험 2~4 세부 설정에 따름.

### Phase 1: 초기 탐색에서 Particle Guidance (PG)를 통한 "강제적 다양성" 확보
* 기존 DAPS의 한계: DAPS는 개별 샘플(Chain)이 독립적으로 MCMC를 수행한다. 우연히 초기화가 잘 되면 서로 다른 해를 찾을 수도 있지만, 대부분은 가장 '쉬운' 해(Dominant Mode)로 다 같이 쏠려버리는 경향이 있다.
* 구간: T=000 ~ 200 (약 80% 구간)
* 동작: LatentDAPS + Particle Guidance (Repulsive Force) - 여러 개의 궤적(Particle)을 동시에 생성하면서, 입자들끼리 서로 밀어내는 힘(Repulsive Force)을 적용. 유사도(Similarity)에 대한 페널티
* 목적: parent 단계에 해당하는 두 particle이 서로 밀어내며 해 공간을 탐색합니다. 하나가 Mode 0°로 가면, 다른 하나는 강제로 Mode 180° 쪽으로 가게 됩니다. 해 공간(Solution Space)을 훨씬 넓게 커버할 수 있습니다.
* ReSample 최적화: OFF (이때 최적화하면 Local Minima에 빠집니다).

💡 보완 제안 (Annealing the Repulsion):
* 문제점: Repulsive Force가 너무 끝까지 유지되면, 두 입자가 서로를 밀어내느라 정작 데이터 매니폴드(Manifold) 정중앙(가장 자연스러운 이미지)에 도달하지 못하고 약간 빗겨난(Off-manifold) 상태가 될 수 있습니다.
* 해결책: TDP 논문에서도 언급하듯, 초기(High Noise)에는 $\alpha_p$(Particle Guidance Scale)를 크게 가져가서 확실하게 갈라놓고, $t_{mid}$에 가까워질수록 $\alpha_p$를 서서히 줄여서(Decay) 입자들이 각자의 Basin(수렴 영역) 안착하도록 유도하는 것이 좋습니다.
* 초기 분기(Bifurcation)의 중요성: Phase Retrieval에서 0도/180도 결정은 노이즈가 매우 큰 초반 단계에서 결정됩니다. 따라서 초반 20~30% 구간에서의 PG 강도가 승패를 가를 것입니다.


### Phase 2:  Bi-level Tree Structure를 통한 Global Optima 탐색 중 가지치기 (Pruning)
Phase Retrieval은 대표적인 Non-convex(비볼록) 최적화 문제로, 잘못된 초기값에서 시작하면 Local Minima에 빠져 영영 못 나올 위험이 큽니다.
* 기존 DAPS의 한계: DAPS는 Noise Annealing을 통해 이를 극복하려 하지만, 하나의 궤적(Sequential)만 따라가기 때문에 만약 초반(t가 클 때)에 잘못된 방향(Local Basin)으로 들어서면 되돌리기 어렵다.
* TDP의 해결책 (Parent Branching & Sub-tree Expansion): TDP는 "Parent Trajectory(부모 궤적)"를 먼저 다양하게 뿌려놓고(Exploration), 가능성 있어 보이는 가지에서 "Child Trajectory(자식 궤적)"를 뻗어 나가며 정밀하게 다듬는다(Exploitation).
    * Phase Retrieval 적용:
        1. Parent 단계 (t: T \to t_{mid}): Particle Guidance를 켜고 DAPS를 수행하여 서로 다른 "대략적인 형태(Coarse Structure)"를 가진 여러 후보군을 확보합니다.
        2. Child 단계 (t: t_{mid} \to 0): 각 Parent에서 가지를 쳐서, 이제는 Repulsive Force를 끄고 강력한 Data Consistency(측정값 일치)를 적용해 정밀한 이미지를 복원합니다.
    * 이 방식은 단순히 하나의 길만 가는 것이 아니라, 여러 가능성을 동시에 탐색하다가 유망한 곳을 집중 공략하므로 Global Optima를 찾을 확률이 비약적으로 상승합니다.
* 시점: T=200 근처
* 동작: 두 입자의 measurement loss를 계산합니다.
* 결정:
    * 둘 다 Loss가 낮다면? 둘 다 살립니다 (하나는 0°, 하나는 180°일 확률 높음).
    * 하나가 압도적으로 낮다면? 나쁜 녀석을 버리고 좋은 녀석을 복제하거나, 좋은 녀석만 남깁니다.
    * 중간 단계(t_{mid})에서 "이 가지는 가망이 없다(Loss가 너무 크다)" 싶으면 가지치기(Pruning)를 해버릴 수 있다.
    * 남는 자원을 유망한 경로에 집중(Child Expansion)할 수 있으니 계산 비용 대비 성능(ROI)이 훨씬 높다.

💡 보완 제안 (Diversity-aware Pruning):
* 시나리오: 만약 두 입자(A, B)가 운 나쁘게 둘 다 0도 모드로 수렴했는데, A가 loss가 더 낮다고 칩시다. 단순히 loss만 보면 B를 버리겠지만, 만약 B가 180도 모드로 가는 중이었다면(아직 loss는 높지만), B를 살리는 게 나을 수도 있습니다.
* 전략: 가지치기를 할 때 단순히 Loss만 볼 것이 아니라, 두 입자 간의 거리(Distance)도 확인하세요.
    * Case 1: 거리가 가깝다 → Loss가 낮은 놈만 남김 (Local Refinement).
    * Case 2: 거리가 멀다 → Loss가 허용 범위 내라면 둘 다 살림 (Global Exploration 유지).


### Phase 3: 정밀 최적화 (Hard Data Consistency)
* 구간: T=200 ~ 0 (마지막 20% 구간)
* 동작: Latent Optimization ON
    * 이제 Repulsive Force를 끕니다 (서로 밀어낼 필요 없음).
    * 대신 ReSample의 Latent Optimization을 켜서, 현재 위치(z)를 측정값(y)에 강하게(Hard) 밀착시킵니다. DAPS도 계속 켭니다.
    * 주의: Pixel Optimization은 절대 금지 (Phase Retrieval에서는 독입니다).
* 목적: DAPS가 남긴 미세한 노이즈를 제거하고 PSNR을 극대화합니다.

ReSample이 필요한 순간: "마지막 한 뼘 (Fine-tuning)"
TDP의 Particle Guidance(PG)와 DAPS로 열심히 탐색해서, 운 좋게 원본 이미지와 유사한 형태(Mode)를 찾았다고 가정해 봅시다. 하지만 DAPS는 본질적으로 '노이즈를 섞는(Annealing)' 방식이기 때문에, 최종 결과물(t=0)에도 미세한 노이즈가 남아있거나 측정값 y와 완벽하게 일치하지는 않을 수 있습니다.
이때 ReSample의 "Latent Optimization"이 등장합니다.
* 역할: "이제 큰 그림(위상, 형태)은 맞았으니, 노이즈를 끄고 디테일을 측정값 y에 강제로(Hard Consistency) 맞춰라."
* 안전한 이유: 이미 DAPS+TDP가 '정답 근처(Basin of Attraction)'까지 데려다 놓았기 때문에, 이제는 최적화를 강하게 걸어도 Local Minima(엉뚱한 해)로 빠지지 않고 Global Optima(진짜 정답)로 쏙 빨려 들어갑니다 
* ReSample에서도 local proximity(정답에 가까운 곳) 안에서 optimiation을 함으로써 local minima에 빠지는 것을 방지하기 위해 DDIM unconditional x0 prediction을 optimization initial point로 삼았던 것과 비슷한 맥락!
ReSample 적용 시점: $T=200$ (Low noise) 시점은 이미 이미지가 거의 다 만들어진 상태입니다. 이때 ReSample의 Optimization을 너무 강하게(Learning rate를 높게) 걸면, 기껏 DAPS가 만들어놓은 자연스러운 텍스처가 망가질 수 있습니다. "Weak Optimization"으로 미세 조정만 하는 것이 더 안전할 수 있습니다.








## 실험 결과

### [실험 0] Baseline 결과 (LatentDAPS 4-run Independent)

#### ImageNet 10 Images (2025-12-13 KST)
| Metric | Value | 비고 |
|--------|-------|------|
| **Best PSNR Mean** | **17.50 dB** | 논문 100img: 20.54 dB |
| Best PSNR Std | 3.67 | |
| Mean of Means | 15.49 dB | |
| Best SSIM Mean | 0.550 | |
| Best LPIPS Mean | 0.558 | (↓ better) |
| Total Time | 2.5시간 (9,060초) | |
| Per Image | ~903초 (15분) | |
| Peak VRAM | 10,161 MB | A10 GPU |

#### 이미지별 상세 결과 (PSNR)
| Image | Sample 0 | Sample 1 | Sample 2 | Sample 3 | **Best** | Std |
|-------|----------|----------|----------|----------|----------|-----|
| 0 | 13.71 | 13.86 | 13.82 | **14.92** | 14.92 | 0.57 |
| 1 | **19.35** | 16.77 | 16.34 | 15.24 | 19.35 | 1.74 |
| 2 | **15.12** | 12.33 | 13.87 | 13.96 | 15.12 | 1.14 |
| 3 | 15.42 | **19.39** | 15.41 | 14.92 | 19.39 | 2.09 |
| 4 | **13.93** | 12.17 | 13.60 | 12.71 | 13.93 | 0.81 |
| 5 | 16.67 | **18.51** | 15.04 | 17.27 | 18.51 | 1.44 |
| 6 | 19.67 | 20.54 | 19.69 | **20.78** | 20.78 | 0.57 |
| 7 | **19.21** | 12.72 | 10.13 | 15.77 | 19.21 | 3.92 |
| 8 | 9.16 | **10.28** | 9.10 | 9.33 | 10.28 | 0.55 |
| 9 | **23.48** | 20.24 | 18.14 | 16.97 | 23.48 | 2.85 |

#### 관찰
- **4 samples 중 best idx 분포**: sample 0 (5회), sample 1 (2회), sample 3 (3회) → 4-run 필요성 확인
- **높은 std 이미지**: img 7 (3.92), img 9 (2.85) → Phase retrieval의 multi-modal 특성 반영
- **어려운 이미지**: img 8 (best=10.28) → 일부 이미지에서 성능 저하

### [실험 1] Repulsion Sanity Check (2025-12-13 KST) - scale=0.1

#### Exp0 vs Exp1 Sanity Check 비교 (1 Image, scale=0.1)
| Metric | Exp0 (Baseline) | Exp1 (Repulsion, s=0.1) | 차이 |
|--------|-----------------|------------------|------|
| PSNR samples | [8.46, 7.83, 8.44, 11.25] | [8.46, 7.82, 8.40, 11.24] | ~동일 |
| **Best PSNR** | 11.25 | 11.24 | -0.01 |
| Mean PSNR | 8.99 | 8.98 | -0.01 |
| Std PSNR | 1.53 | 1.53 | 동일 |
| Best SSIM | 0.565 | 0.565 | 동일 |
| Best LPIPS | 0.495 | 0.495 | 동일 |
| **Time** | 900초 | 910초 | +10초 (+1.1%) |
| **Peak VRAM** | 10,117 MB | 10,209 MB | +92 MB (+0.9%) |

#### Exp1 Repulsion 설정 (scale=0.1)
| Parameter | Value |
|-----------|-------|
| repulsion_scale | 0.1 |
| repulsion_sigma_break | 1.0 |
| repulsion_schedule | linear |
| repulsion_dino_model | dino_vits16 |
| repulsion_active_steps | 30/50 steps |
| repulsion_total_time | 11.4초 |
| mean_pairwise_distance | 32.13 |

#### 관찰 및 분석
- **PSNR 거의 동일**: repulsion이 켜졌음에도 결과가 baseline과 거의 같음
- **가능한 원인**:
  1. `repulsion_scale=0.1`이 너무 약할 수 있음 → scale 증가 실험 필요
  2. 1 image만으로는 통계적 의미 부족 → 10 image 실험 필요
  3. 같은 seed 사용으로 trajectory가 비슷하게 수렴했을 가능성
- **Overhead 미미**: 시간 +1.1%, VRAM +0.9% → repulsion 연산 비용 낮음
- **다음 단계**: `repulsion_scale` 조정 또는 10 image 실험으로 효과 검증 필요

### [실험 1/3] Overnight Scale Grid Search (2025-12-14 KST) - scale=0.5, 1.0

#### 핵심 요약
| 실험 | Particles | Scale | Best PSNR | Time | VRAM |
|------|-----------|-------|-----------|------|------|
| Exp0 Baseline | 4 | 0.0 | 17.50 | 100% | 100% |
| **Exp1-B (Best)** | 4 | 1.0 | **17.68** (+0.18) | +1.4% | +0.9% |
| Exp3-B | 2 | 1.0 | 17.35 (-0.15) | **-48%** | **-40%** |

#### 실험 설정
| 실험 | num_samples | repulsion_scale | 목표 |
|------|-------------|-----------------|------|
| Exp1-A | 4 | 0.5 | scale 증가 효과 확인 |
| Exp1-B | 4 | 1.0 | 더 강한 repulsion 효과 확인 |
| Exp3-A | 2 | 0.5 | 2-particle baseline |
| Exp3-B | 2 | 1.0 | 2-particle + 강한 repulsion |

#### 전체 비교 결과 (10 Images)

| 실험 | Particles | Scale | Best PSNR ↑ | Std | Mean of Means | Time (초) | VRAM (MB) |
|------|-----------|-------|-------------|-----|---------------|-----------|-----------|
| **Exp0 Baseline** | 4 | 0.0 | **17.50** | 3.67 | 15.49 | 9,060 | 10,161 |
| Exp1-A | 4 | 0.5 | 17.54 (+0.04) | 3.45 | 15.50 | 9,182 (+1.3%) | 10,252 |
| **Exp1-B** | 4 | 1.0 | **17.68 (+0.18)** | 3.20 | 15.56 | 9,191 (+1.4%) | 10,252 |
| Exp3-A | 2 | 0.5 | 17.20 (-0.30) | 3.67 | 16.04 | 4,694 (-48%) | 6,100 |
| Exp3-B | 2 | 1.0 | 17.35 (-0.15) | 3.68 | 16.14 | 4,694 (-48%) | 6,100 |

#### SSIM / LPIPS 비교

| 실험 | Best SSIM ↑ | Best LPIPS ↓ |
|------|-------------|--------------|
| Exp0 Baseline | 0.550 | 0.558 |
| Exp1-A (s=0.5) | 0.550 | 0.562 |
| Exp1-B (s=1.0) | 0.552 | 0.556 |
| Exp3-A (s=0.5) | 0.535 | 0.568 |
| Exp3-B (s=1.0) | 0.544 | 0.563 |

#### Repulsion Metrics

| 실험 | Mean Pairwise Distance | Repulsion Time (초) |
|------|------------------------|---------------------|
| Exp1-A (4p, s=0.5) | 32.09 | 10.9 |
| Exp1-B (4p, s=1.0) | 32.22 | 10.8 |
| Exp3-A (2p, s=0.5) | 32.21 | 5.6 |
| Exp3-B (2p, s=1.0) | 32.22 | 5.6 |

#### 이미지별 Best PSNR 상세 비교

| Img | Exp0 (4p,s=0) | Exp1-A (4p,s=0.5) | Exp1-B (4p,s=1.0) | Exp3-A (2p,s=0.5) | Exp3-B (2p,s=1.0) |
|-----|---------------|-------------------|-------------------|-------------------|-------------------|
| 0 | 14.92 | 14.92 | 14.92 | 13.86 | 13.86 |
| 1 | 19.35 | 19.38 | 19.35 | 19.35 | 19.38 |
| 2 | 15.12 | **15.85** | 14.41 | 14.52 | **15.91** |
| 3 | 19.39 | **19.57** | 18.97 | 19.27 | 19.32 |
| 4 | 13.93 | 13.72 | **13.68** | 12.20 | **13.95** |
| 5 | 18.51 | 18.47 | **18.89** | 18.45 | 18.47 |
| 6 | 20.78 | 20.55 | **20.99** | 20.53 | 20.55 |
| 7 | 19.21 | 18.38 | 18.39 | **18.49** | 18.40 |
| 8 | 10.28 | **11.10** | **13.65** ⭐ | 11.87 | 10.15 |
| 9 | 23.48 | 23.48 | **23.55** | 23.47 | 23.47 |

- ⭐ **Image 8**: Exp1-B(s=1.0)에서 +3.37 dB 극적 개선 (10.28→13.65)
- Bold: 해당 row에서 최고 성능

#### 관찰 및 분석

**1. Scale 효과 (Exp1)**
- scale=0.5 → scale=1.0으로 증가 시 Best PSNR +0.14 dB 개선 (17.54 → 17.68)
- Baseline 대비 scale=1.0에서 +0.18 dB 개선
- **결론**: Repulsion이 약간의 성능 향상에 기여하나, 효과가 크지 않음

**2. Particle 수 효과 (Exp1 vs Exp3)**
- 4 particle → 2 particle 감소 시 Best PSNR ~0.3 dB 하락
- 하지만 **Mean of Means는 2 particle이 더 높음** (16.04~16.14 vs 15.50~15.56)
  - 이는 4 particle 중 일부가 낮은 점수를 기록하기 때문
- 시간/메모리 약 절반으로 절약 (~48% 시간, ~60% VRAM)

**3. Mean Pairwise Distance 이슈** ⚠️
- **모든 실험에서 ~32로 거의 동일**
- scale=0.5나 1.0이나 pairwise distance 차이 없음
- **가능한 원인**:
  1. DINO feature space에서 이미 충분히 분리되어 있음
  2. Repulsion gradient가 실제 latent trajectory에 큰 영향을 주지 못함
  3. Scale이 여전히 부족하거나, score injection 방식의 한계

**4. 효율성 분석**
- Exp3 (2 particle): 시간 48%, VRAM 60%로 성능 유지 가능
- Best PSNR은 4 particle이 우세하지만, 효율성 면에서 2 particle도 고려 가능

#### 다음 단계 제안

1. **Scale 추가 실험**: scale=2.0, 5.0 등 더 큰 값으로 repulsion 효과 확인
2. **Sigma break 조정**: repulsion이 더 오래 유지되도록 sigma_break 낮추기 (0.5, 0.1)
3. **Pairwise distance 디버깅**: repulsion이 실제로 latent separation을 유도하는지 step별 로깅 강화
4. **실험 2 (Pruning) 진행**: 4→2 pruning으로 효율성 + 성능 양립 검증


## 프로젝트 기대 결과: 보다 적은 연산으로 비슷하거나 더 좋은 성능을!
- DAPS에서 Phase Retrieval의 불안정성을 고려하여, 4번의 independent runs을 수행한 뒤 가장 좋은 결과를 선택하여 보고했으니, 우리플젝을 DAPS 4 run이랑 비교했을때 시간xGPU 사용량이 비슷하거나 작으면서 성능이 비슷하거나 높음을 보이면 되는 것!
- 실험 2 (4 → 2 Pruning)**는 이론적 최적점(2 Modes)과 현실적 안전장치(4 Runs) 사이의 **"Sweet Spot"**을 찾는 설정
- max 값 뿐만 아니라 std 등 분포를 가지고도 의미있는 분석을 해볼 수 있을 것.


## 구현 가이드
- 모든 Measurement Operator($\mathcal{A}$)와 Loss Function은 (B, C, H, W) 형태의 입력을 받아 **배치 단위로 병렬 연산(Broadcasting)**이 가능하도록 작성되어야 한다. for 루프로 배치를 처리하지 말고 PyTorch의 텐서 연산을 쓸 것!
- 우리는 하나의 $y$(측정값)에 대해 2~4개의 서로 다른 $z_T$(초기 노이즈)를 생성해야 합니다. Data Loader에서 이미지 1장을 가져오면, 이를 **batch_size=2~4로 복제(repeat)**하되, 초기 노이즈 $z_T$는 torch.randn(2~4, ...)로 서로 다르게 생성되도록 코드를 짤 것!
- ~~보통 Diffusion Inference는 with torch.no_grad(): 안에서 돕니다. 하지만 우리는 **Repulsion($\nabla_z \Phi$)**과 ReSample Optimization($\nabla_z \|y - Ax\|^2$) 때문에 실험 1~5에서 Gradient가 필요할 예정이다. 따라서, Sampler의 메인 루프는 기본적으로 Gradient 계산이 가능하도록 열어두고(enable_grad), 필요한 부분에서만 메모리 절약을 위해 no_grad를 쓰거나, 혹은 반대로 no_grad 베이스에 특정 스텝(PG, Optimization)에서만 enable_grad를 켜는 토글(Toggle) 구조를 미리 실험 0에서부터 만들어야 한다!~~ → **완료**: `sampler.py`의 `LatentDAPS.sample()`에서 `torch.set_grad_enabled(step_needs_grad)` 구조 구현. `do_repulsion`과 `do_optimization` flag로 step별 gradient 활성화 제어. 실험 1, 2, 4 로직은 TODO 주석으로 준비됨.
- ~~실험 0~5를 스크립트 하나로 제어하려면 Flag 설계가 중요하다. 다음 Argument들을 미리 정의해 둘 것!~~ → **완료**: `configs/default.yaml`에 정의됨
    - `num_samples` (int): 한 번에 생성할 입자(이미지)의 개수 (기존 DAPS의 num_samples를 그대로 활용, particle_num 역할)
    - `repulsion_scale` (float): 입자끼리 밀어내는 힘의 초기 강도. 0.0이면 독립 실행 (DAPS baseline), >0.0이면 서로 밀어냄
    - `pruning_step` (int): 가지치기 수행 timestep. -1이면 pruning 없음
    - `optimization_step` (int): latent optimization 시작 timestep. -1이면 optimization 없음
    - `use_tpu` (bool): TPU 사용 여부.
    - (num_eval_images는 data config에서 제어)
- 실험별 argument 세팅 가이드:
    Exp 0: Baseline (DAPS Replication)particle_num=4, repulsion_scale=0.0:이렇게 설정하면 4개의 입자가 서로 간섭하지 않으므로, DAPS 논문에서 "1개씩 4번 돌린 것(4 runs)"과 수학적으로 완전히 동일한 결과를 냅니다. (시드만 잘 제어된다면)이것이 우리의 Reference 성능이 됩니다.
    Exp 1: Repulsion Onlyrepulsion_scale > 0:이제 4개의 입자가 서로 밀어냅니다.목표: Exp 0보다 **다양성(Std)**이 높고, **최고점(Max PSNR)**이 높게 나오는지 확인합니다.
    Exp 2: Efficiency (Pruning)pruning_step=200:코드는 $t=200$이 되는 순간, Loss와 Distance를 계산하여 **4개 중 2개를 메모리에서 삭제(또는 Masking)**해야 합니다.목표: Exp 1과 성능은 비슷한데, **시간(Time)과 메모리(VRAM)**가 줄어드는지 확인합니다.
    Exp 4: Quality (Optimization)optimization_step=200:$t=1000 \to 201$까지는 Repulsion으로 탐색하고,$t=200 \to 0$부터는 Repulsion을 끄고(scale=0 강제 적용), Latent Optimization을 켭니다.목표: Exp 2보다 PSNR이 확실히 더 올라가는지 확인합니다.
- metric.json에 phase별 time, gpu, optimization 횟수/시간을 기록할 것
- metric.json을 Parsing하는 코드를 만들 것
- 코드 실행을 통한 sanity check는 GPU가 달린 서버에서 진행할 예정! (로컬 맥북 X)
- TODO를 완료한 경우 이 PROJECT.md 파일에 취소선을 그어 표시할 것! 만약 논의 결과 md 설명보다 더 적합한 선택지가 있어서 실제 구현에 차이가 생긴 경우 PROJECT.md를 업데이트할 것!
- git commit message는 한/영 혼용 가능, 실험 몇을 준비하고 있는지 명시, 한 줄 이내로 작성. commit은 vscode gui로만 진행
- 현 폴더는 DAPS 레포를 베이스로 수정 중에 있으며, TDP 및 ReSample 관련 세부사항은 추후 해당 실험 구현 단계에서 추가 예정
- ~~command 파일들에 새로운 argument들 반영 및 1/10/100 image용 command 추가~~ → **완료**: 폴더 구조:
    - `commands_gpu/`: GPU (CUDA) 전용 명령어 (use_tpu=false)
    - 각 폴더에 `exp0_baseline.sh` ~ `exp5_final.sh` 포함
    - 모든 command에 `repulsion_scale`, `pruning_step`, `optimization_step`, `data.end_id` 반영





안녕, 다음에 따라 “RLSD repulsion을 LatentDAPS(EDM σ + x₀-pred)로 score-level injection 방식으로 이식”하는 작업을 가장 안전하게 진행해주라.
(pruning/optimization은 아직 제외, 이 문서에서 Exp1/3만 타겟)

You have access to two local repos:
	•	Repo DAPS (target): my modified LatentDAPS / DAPS codebase, I changed vanilla DAPS so please be aware that content's different from original DAPS repo. You can refer to this PROJECT.md for concrete implementation done and planned. Though the repo name is DAPS, I am interested in LatentDAPS setting only.
	•	Repo RLSD (source): RLSD (Repulsive Latent Score Distillation) official repo
	(Please ignore repo named 'DAPS_modified' since it's outdated.)

Goal: implement particle repulsion for Exp1/Exp3 (full-run 4 or 2 particles, no pruning, no optimization) by porting RLSD’s repulsion module into LatentDAPS.

Context / What we know (important)

Target repo (LatentDAPS) uses:
	•	EDM sigma parameterization (σ), not DDPM alpha. Evidence:
	•	uses annealing_scheduler.sigma_steps[step]
	•	forward diffusion like x_t = x0 + σ * ε
	•	has sigma_max and prior_sigma
	•	Prediction target is x0-prediction (denoiser), NOT eps-pred.
	•	In DiffusionPFODE.derivative():
		return dst / st * xt - st * dsigma_t * sigma_t * self.model.score(xt/st, sigma=sigma_t)
	•	model.score() is derived from denoiser D(x;σ) via EDM relation:
		score = (D(x;σ) - x) / σ^2

We want repulsion injection method:

Method to use: Add repulsion directly to score.
Rationale:
	•	Equivalent to modifying denoiser output by + γ σ^2 repulsion, but simpler:
	•	If score = (D-x)/σ^2, then D' = D + γ σ^2 repulsion ⇒ score' = score + γ repulsion
	•	Matches DAPS update form (prior gradient + data gradient). Repulsion is an extra regularizer/guidance term added to the “prior gradient direction”.
	•	We will implement sampler-loop computed repulsion, store it in pfode, and add it in DiffusionPFODE.derivative():

		# sampler loop
		repulsion = compute_repulsion(zt)
		pfode.set_repulsion(repulsion, scale=current_scale)

		# DiffusionPFODE.derivative
		score = self.model.score(...)
		if self.repulsion is not None:
			score = score + self.repulsion_scale * self.repulsion

		do not wrap model, maintain sampler -> pfode passing structure for clarity in responsibility and on/off control and debuggability.

Repulsion should be ON only for early/high-noise interval and decay to 0:
	•	RLSD uses if sigma > sigma_break to enable repulsion.
	•	We will implement interval on/off with alpha(σ) schedule (linear or cosine) such that alpha -> 0 as σ→0.
	•	For now implement something simple: repulsion active for sigma > sigma_break, and within that, scale by alpha = repulsion_scale * schedule(sigma).

RLSD repulsion reference implementation (source repo):

RLSD repulsion core in rsd.py lines ~123-165 (already identified). It:
	•	decodes latent to image
	•	extracts DINO features
	•	computes pairwise differences in feature space
	•	uses RBF kernel with median heuristic bandwidth
	•	computes SVGD-style repulsive gradient
	•	backprops from feature space to latent using vector-Jacobian trick:
		eval_sum = torch.sum(dino_out * grad_phi.detach())
		deps_dx_backprop = torch.autograd.grad(eval_sum, latent_pred_t)[0]
	•	normalizes by kernel sum

⚠️ Important fix:
RLSD code uses h = median(dist)^2 / log(N-1). This breaks for N=2 (log(1)=0) which is our project's Exp3 setting.
We will use a safe denominator: log(N) (or max(log(N), eps)) so that Exp3 (2 particles) works.

Deliverables / Tasks

Task 1: Identify integration points in LatentDAPS (target repo)

Find where:
	•	multiple particles are represented as a batch (num_samples already exists)
	•	sampler loop produces the latent state zt (or equivalent) each step
	•	pfode (DiffusionPFODE) is called for derivative or stepping

We need:
	1.	A function to compute repulsion given current latent batch.
	2.	Store repulsion in pfode (or scheduler) object so derivative() can access it.
	3.	Add repulsion to score in DiffusionPFODE.derivative() before it’s used in drift.

Task 2: Port RLSD repulsion computation into target repo

Implement something like:

repulsion = compute_repulsion(latents, sigma_t)

	•	Use RLSD’s DINO-based feature space repulsion.
	•	Use decode_latents(latents) from target repo (or equivalent) to get images.
	•	Use DINO-ViT model (frozen, eval mode).
	•	Ensure gradients flow back to latent: need latent.requires_grad_(True) in repulsion-on steps.
	•	Use vector-Jacobian trick like RLSD (avoid second-order).
	•	Bandwidth: median(dist)^2 / max(log(N), eps)
	•	compute pairwise distances
	•	h = median(dist)**2 / max(log(N), eps)  ✅ (fix N=2)
	•	Normalize by kernel sum similarly.

Task 3: Add config options (minimal)

Expose in config / args:
	•	repulsion_scale (float, default 0.0)
	•	sigma_break or repulsion_sigma_break (float or step index)
	•	optional repulsion_schedule (linear default)

Repulsion behavior:
	•	If repulsion_scale == 0, it must exactly reproduce DAPS baseline (independent chains).
	•	If on, only apply when sigma > sigma_break.

Task 4: Ensure correct grad toggling and memory safety
	•	Only enable autograd for repulsion steps.
	•	DINO parameters must have requires_grad=False, but input must require grad.
	•	DINO input should be resized to 224 (if RLSD does). Use same preprocessing as RLSD repo. But be aware that the input images' resolutions are different; DAPS handles 256x256 while RLSD handles 512x512 and latent resolution may be different across two repos. Adjust accordingly and tell me what you did. Ask me if any uncertainty. Place debugging code if needed and tell me that there are debugging code I have to check the outputs.

Task 5: Logging for sanity (Exp1/3)

Add to metrics/logging:
	•	mean pairwise DINO feature distance per step
	•	whether repulsion was on/off
	•	optionally norm of repulsion gradient
	•	time per repulsion computation
Be aware of the bugs resulting from new logging terms.

This is crucial to verify repulsion is actually acting.

Task 6: Provide a minimal test run plan - note that each sanity test may take 15+ minutes.
	•	run 1 image sanity with 4 particles, repulsion on; check:
	•	pairwise distance increases early
	•	later decreases/stabilizes when repulsion off/decayed
	•	no NaNs
	•	run 1 image sanity with 2 particles; ensure no crash (bandwidth fix works).

Implementation guidance (preferred architecture)

Do NOT wrap/modify model.score() logic deeply. Keep responsibilities separated:
	•	sampler computes repulsion and sets it on pfode each step:
		pfode.set_repulsion(repulsion, scale=alpha_sigma)
	•	pfode’s derivative() adds it:
		score = score + self.repulsion_scale * self.repulsion
구조:
	•	sampler: “언제/얼마나 repulsion”
	•	model: “순수 denoising”
	•	pfode: “ODE drift 계산”

This allows easy on/off scheduling and debugging.

Output requirements
	•	Make a clean PR-style change:
	•	new module file for repulsion (e.g., repulsion.py)
	•	minimal changes in sampler loop and pfode derivative
	•	config updated, especially for exp1/3 sh commands file! and other files too; turn off repulsion option for baseline exp0.
	•	Summarize:
	•	files changed
	•	key design decisions
	•	how to run Exp1/Exp3
	•	any assumptions or TODOs

Please proceed by:
	1.	scanning target repo to find exact insertion points and existing decoding utilities
	2.	mapping RLSD code dependencies (DINO loading, preprocessing, additionally required env setting: pip requirements, download sh, etc.)
	3.	implementing and testing quickly with 1-image run (if any additional installation or download is needed, do so, and document it inside DAPS requirements and download scripts)
	4.	report back with patch summary and instructions.