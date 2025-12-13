# My Project: LatentDAPS로 Langevin Dynamics sampling + TDP-style 탐색으로 0° 180° 찾기 + 맨 마지막 hard data consistency 적용

## 실험 진행 및 구현 과정 설계

### [데이터] imagenet 10장으로 method 비교, 마지막 eval은 ffhq imagenet 100장씩으로 하는걸 목표로, 여건 안되면 ffhq는 버리기 / 시드 고정 (이미 DAPS에서는 42)

### [실험 0] LatentDAPS 논문에 eval 데이터는 100 image에만 나와있으니까 비교를 위해 LatentDAPS(with Langevin Dynamic)의 imagenet first 10 image에 대한 phase retrieval 성능 측정. 
- 단, 이때 image별로 전부 돌아간 뒤 다음 run이 실행되는 구조로 4 run이 구현돼있는데, 이후 실험들과의 원활한 비교를 위해 eval 명령어를 4 batch = 4 run 구조로 변경해야 함.
- 또, time도 logging하도록 코드를 수정해야 함. time.time()으로 구간별(Phase 1, 2, 3) 소요 시간을 따로 찍어두면, 나중에 "Pruning으로 Phase 3 시간을 얼마나 줄였는지" 그래프 그리기 좋을 것. 이후, logging 코드에 대한 sanity check 차원에서 1 image 4 run 명령어만 먼저 한번 돌려볼 것.
- 시간에 따른 GPU VRAM 소모량 변화를 기록해두는 것도 향후 pruning 메소드와의 연산량 비교를 위해 도움이 될듯. VRAM 기록 시 torch.cuda.max_memory_allocated()를 활용하세요.

### [실험 1] 4-Particle Full Run (Repulsion vs. Independence)
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








## 프로젝트 기대 결과: 보다 적은 연산으로 비슷하거나 더 좋은 성능을!
- DAPS에서 Phase Retrieval의 불안정성을 고려하여, 4번의 independent runs을 수행한 뒤 가장 좋은 결과를 선택하여 보고했으니, 우리플젝을 DAPS 4 run이랑 비교했을때 시간xGPU 사용량이 비슷하거나 작으면서 성능이 비슷하거나 높음을 보이면 되는 것!
- 실험 2 (4 → 2 Pruning)**는 이론적 최적점(2 Modes)과 현실적 안전장치(4 Runs) 사이의 **"Sweet Spot"**을 찾는 설정
- max 값 뿐만 아니라 std 등 분포를 가지고도 의미있는 분석을 해볼 수 있을 것.


## 구현 가이드
- 모든 Measurement Operator($\mathcal{A}$)와 Loss Function은 (B, C, H, W) 형태의 입력을 받아 **배치 단위로 병렬 연산(Broadcasting)**이 가능하도록 작성되어야 한다. for 루프로 배치를 처리하지 말고 PyTorch의 텐서 연산을 쓸 것!
- 우리는 하나의 $y$(측정값)에 대해 2~4개의 서로 다른 $z_T$(초기 노이즈)를 생성해야 합니다. Data Loader에서 이미지 1장을 가져오면, 이를 **batch_size=2~4로 복제(repeat)**하되, 초기 노이즈 $z_T$는 torch.randn(2~4, ...)로 서로 다르게 생성되도록 코드를 짤 것!
- 보통 Diffusion Inference는 with torch.no_grad(): 안에서 돕니다. 하지만 우리는 **Repulsion($\nabla_z \Phi$)**과 ReSample Optimization($\nabla_z \|y - Ax\|^2$) 때문에 실험 1~5에서 Gradient가 필요할 예정이다. 따라서, Sampler의 메인 루프는 기본적으로 Gradient 계산이 가능하도록 열어두고(enable_grad), 필요한 부분에서만 메모리 절약을 위해 no_grad를 쓰거나, 혹은 반대로 no_grad 베이스에 특정 스텝(PG, Optimization)에서만 enable_grad를 켜는 토글(Toggle) 구조를 미리 실험 0에서부터 만들어야 한다!
- 실험 0~5를 스크립트 하나로 제어하려면 Flag 설계가 중요하다. 다음 Argument들을 미리 정의해 둘 것! 
    --particle_num (int): 한 번에 생성할 입자(이미지)의 개수입니다. 즉, Batch Size입니다. 역할: DAPS의 4번 실행을 재현하거나(4), 2개로 줄였을 때(2)를 제어합니다.
    --repulsion_scale (float): 입자끼리 밀어내는 힘(Particle Guidance)의 **초기 강도($\alpha_p$)**입니다. 역할: 0.0이면 서로 무시하고 독립적으로 생성(DAPS Baseline)하며, >0.0이면 서로 밀어내며 다양성을 확보합니다. (Time-decay 적용 필요)
    --pruning_step (int): 가지치기를 수행할 **Diffusion Time Step ($t$)**입니다. 역할: -1이면 가지치기 없이 끝까지 갑니다. 200이면 $t=200$ 시점에서 하위 입자를 제거하고 상위 2개만 남깁니다.
    --optimization_step (int): ReSample 방식의 Hard Data Consistency(Latent Optimization)를 **시작할 시점($t$)**입니다. 역할: -1이면 최적화 없이 DAPS 샘플링만 수행합니다. 200이면 $t=200$부터 $0$까지 Repulsion을 끄고 Optimization을 켭니다.
    --num_eval_images (int): 평가할 전체 이미지의 수입니다. 역할: 1(Sanity Check), 10(Tuning), 100(Final Eval)을 제어합니다.
- 실험별 argument 세팅 가이드:
    Exp 0: Baseline (DAPS Replication)particle_num=4, repulsion_scale=0.0:이렇게 설정하면 4개의 입자가 서로 간섭하지 않으므로, DAPS 논문에서 "1개씩 4번 돌린 것(4 runs)"과 수학적으로 완전히 동일한 결과를 냅니다. (시드만 잘 제어된다면)이것이 우리의 Reference 성능이 됩니다.
    Exp 1: Repulsion Onlyrepulsion_scale > 0:이제 4개의 입자가 서로 밀어냅니다.목표: Exp 0보다 **다양성(Std)**이 높고, **최고점(Max PSNR)**이 높게 나오는지 확인합니다.
    Exp 2: Efficiency (Pruning)pruning_step=200:코드는 $t=200$이 되는 순간, Loss와 Distance를 계산하여 **4개 중 2개를 메모리에서 삭제(또는 Masking)**해야 합니다.목표: Exp 1과 성능은 비슷한데, **시간(Time)과 메모리(VRAM)**가 줄어드는지 확인합니다.
    Exp 4: Quality (Optimization)optimization_step=200:$t=1000 \to 201$까지는 Repulsion으로 탐색하고,$t=200 \to 0$부터는 Repulsion을 끄고(scale=0 강제 적용), Latent Optimization을 켭니다.목표: Exp 2보다 PSNR이 확실히 더 올라가는지 확인합니다.
- metric.json에 phase별 time, gpu, optimization 횟수/시간을 기록할 것
- metric.json을 Parsing하는 코드를 만들 것