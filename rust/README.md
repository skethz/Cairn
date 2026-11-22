# Cairn Rust 실행 구현과 Python 대조

Cairn의 모델 연산, 학습, 평가, 데이터 처리, 체크포인트, 로컬 FP4 추론과 실행 관리 경로를 Rust로 옮겼다. 서버 위치는 `/scratch/hongse/mythos/cairn-rust`이다. 검증 기준은 **우리가 재작성한 Python 구현**이며, 의심했던 외부 저장소의 코드나 가중치를 실행·사용하지 않았다.

Rust 실행 파일은 Python 인터프리터를 호출하지 않는다. GPU 계산은 **LibTorch 2.11.0 / CUDA 12.8**을 사용한다. 모델·학습 관리는 Rust이고, BF16 설정·fused AdamW·gradient clipping·난수 상태 접근과 순차 delta 역전파에는 C++ 연결부가 있다. v2에는 직접 작성한 CUDA 정수 라우팅 커널도 포함한다. CUDA 라이브러리 전체를 Rust로 재작성했다는 의미는 아니다. [tch 0.24.0 문서](https://docs.rs/tch/0.24.0/tch/)

v1의 짧은 입력 검사 이후, v2에서는 실제 1.3B 모델 4개의 코퍼스 학습 재개까지 검증했다. 현재 학습과 자동 실행 대기열은 사용자 요청으로 일시 중단되어 있다. 마지막 Cairn 실행은 서버의 읽기 전용 `releases/20261103-v4/bin/`을 사용했고, 이후 요청 시 재개할 순차 대기열에는 `releases/20261104-v5/bin/`을 사용한다. v4와 v5의 모델 worker는 v2 검증 및 v3 운영 worker와 파일 바이트가 동일하다. v5 관리 도구는 선행 실행의 완료를 기다리고 모델별로 검증된 메모리 설정을 유지한다. `training/`과 `verification/owned_local/`의 Python은 수치 대조 및 이전 실행의 재현을 위해 보존한다.

## 사용자 요청으로 일시 중단 — 2026-11-04

06:12 CET에 Cairn을 step **27,814**, **882,864,406 유효 토큰**에서 checkpoint `checkpoint_00027814`에 저장하고 중단했다. GPU 4의 사용 메모리 0 MiB와 Cairn worker, 기존 controller, 후속 대기 controller의 종료를 확인했다. 자동 실행은 비활성 상태이며 다음 사용자 요청 전에는 재개하지 않는다. [중단 및 무결성 확인 기록](evidence/gpu4_pause_20261104T051200Z/summary.json)

가중치·Adam 상태 파일 전체 SHA-256, optimizer 항목, CPU/CUDA RNG, 데이터 cursor와 config 해시를 확인했다. 이번 중단으로 잃은 update는 0이다. 비교 모델의 checkpoint는 아래 03:14 CET 전환 기록과 동일하게 보존한다.

다음에 요청받을 재개 순서는 **Cairn → GDN-H → KDA + MLA → GDN + GQA**이다. 네 모델의 `stop.request`를 유지하고, 재개 요청 후 GPU 4의 사용 가능 여부와 checkpoint를 확인한 다음 해당 요청 파일을 보관하고 기록된 순차 실행 명령을 시작한다. 이전 계획의 Dense GQA와 Full MLA는 보류하며 이 네 모델의 재개 명령에 포함하지 않았다.

## GPU 4 순차 학습 전환 — 2026-11-04

사용자 요청에 따라 GPU 5–7의 비교 모델을 현재 update 완료 후 checkpoint에 저장하고 GPU를 반환했다. 03:14 CET 확인에서 GPU 5–7은 모두 사용 메모리 0 MiB였고, Cairn은 같은 GPU 4 worker에서 중단 없이 step 24,745, 785,631,637 유효 토큰까지 진행했다. [전환 확인 기록](evidence/queue_consolidation_20261104T020744Z/summary.json)

03:14 CET에는 GPU 4에서 Cairn의 학습과 최종 평가가 완료되면 아래 순서로 자동 실행하도록 설정했다. 기존에 대기 중이던 Dense GQA와 Full MLA도 뒤로 옮겼다. 이전 GPU 5–7의 실행 대기열은 종료했다. **이 계획은 이후 06:12 CET 사용자 요청으로 일시 중단되었으며 현재 자동 실행되지 않는다.**

| 순서 | 모델 | 재개 checkpoint | 저장된 유효 학습 토큰 | 보관 블록 수 |
|---|---|---:|---:|---:|
| 1 | GDN-H recipe proxy | 25,277 | 802,400,394 | 14 |
| 2 | KDA + MLA operator control | 11,792 | 373,917,428 | 5 |
| 3 | GDN + GQA | 25,932 | 823,137,819 | 14 |
| 4 | Dense GQA | 처음부터 | 0 | 8 |
| 5 | Full MLA | 처음부터 | 0 | 4 |

세 checkpoint의 가중치·Adam 상태 파일 SHA-256, 모든 파라미터의 optimizer 항목, CPU/CUDA RNG, 데이터 cursor와 config 해시를 확인했다. 이번 정지에서 미저장 update 손실은 0이다. 기존의 데이터·학습 예산·worker 실행 파일을 유지하며 새 worker SHA-256은 `d2286b751d25d5d8e8ccb47adc762719d982569f1b974611d3796e90de3e97dd`이다.

v5 `cairn-control queue --after-run cairn_1p3b_seed17_v1`은 GPU를 점유하지 않고 선행 실행의 `completed` 상태와 worker/controller lock 해제를 기다린다. 대기 중인 계획은 서버의 `runs/queue_gpu4.pending.json`, 실행 후 상태는 `runs/queue_gpu4.json`에 기록된다. 선행 실행이 정지되거나 실패하면 후속 실행도 시작하지 않는다. 시작 직전 GPU 점유와 run lock, 새 stop 요청을 다시 검사한다. [고정 실행 명령](evidence/queue_consolidation_20261104T020744Z/launch.json)

관리 도구는 offline 빌드와 3개 CPU 단위 검사를 통과했다. 별도 실행 검사에서 GPU 점유 없는 대기, 중복 대기열 거부, 선행 실행 정지 시 후속 모델 미실행을 확인했다. 모델 연산은 변경하지 않았다. [대기열 검증](evidence/queue_consolidation_20261104T020744Z/controller_verification.json)

## Python에서 Rust로 처음 전환한 이력

2026-11-03, 기존 Python 모델 worker 4개를 종료하고 **GPU 4–7에서 Rust 학습을 재개했다**. 새 실행 폴더는 `/scratch/hongse/mythos/cairn-rust/runs/`이며 [실행 확인 기록](evidence/optimization/live_snapshot.json)과 [전환 이력](evidence/optimization/migration_manifest.json)에 PID·시각·복원 지점을 보관한다.

| GPU | 모델 | 복원한 checkpoint | 재계산할 미저장 update |
|---|---|---:|---:|
| 4 | Cairn | 1000 | 484 |
| 5 | GDN-H recipe proxy | 1500 | 215 |
| 6 | KDA+MLA operator control | 500 | 372 |
| 7 | GDN+GQA | 1500 | 214 |

기존 Python은 500 update마다 checkpoint를 저장했다. 종료 시점의 미저장 진행 상태는 보존되지 않았고 위 구간을 같은 문서 순서로 재계산한다. 저장된 가중치·Adam moments/step·CPU/CUDA RNG·데이터 cursor는 파일 해시까지 확인하여 복원했다. 원래 실행 폴더와 기록은 남겨 두었고, 새 이력에는 복원 checkpoint까지만 이전 지표를 이어 붙였다. GPU 0–3은 변경하지 않았다.

## B200 최적화와 실제 크기 검증

최적화 설정은 모델 config와 별도로 `execution.json`에 남긴다. 데이터·토크나이저·모델 크기·문서 순서·미니배치·누적 횟수·AdamW·학습률·1B 유효 토큰 예산은 동일하다. 비교 검사와 새 실행에는 deterministic CUDA 설정을 명시적으로 적용한다. 이전 Python 실행은 이 설정을 강제하지 않았으므로 그 구간의 실행 정책도 이력에 보존한다.

1. **CUDA 라우팅:** expert별 `nonzero`를 반복하던 경로를 세 정수 커널(count/prefix/pack)로 바꿨다. expert마다 행·top-k 순서를 보존하며, host 동기화는 한 번이다. 1행, 비정렬 길이, 65,536행, 빈 expert, 64 experts, 한 expert에 몰린 입력을 검사했다.
2. **MoE 누적:** expert마다 전체 출력 텐서를 복사하지 않고 같은 순서로 제자리 누적한다.
3. **순차 delta 역전파:** 상태 전이의 행렬곱 순서를 유지하고 각 chunk의 기울기를 목적지에 한 번씩 기록한다. 기존 slice 역전파가 만들던 전체 크기의 0 텐서와 반복 복사를 줄인다. 기준과 같은 contiguous 기울기 배치를 유지해야 후속 reduction의 합산 순서도 같아진다.
4. **재계산 감소:** 초기 v2 검증에서는 Cairn/GDN-H/GDN+GQA 8개 블록, KDA 2개 블록의 그래프를 보관했다. 현재 설정은 아래 170 GiB 목표 절에 기록한다. GPU 메모리를 더 사용하여 해당 블록의 forward 재계산을 생략하며, 이 개수는 모델의 층수나 학습 조건을 바꾸지 않는다.

v1의 긴 BF16 입력에서 발견한 오류도 수정했다. Python은 여러 loss tile에서 출력층 가중치의 BF16 변환을 공유한다. v2는 tile 기울기를 BF16에서 먼저 합산한 후 FP32 master 가중치로 한 번 변환한다. tile마다 FP32로 변환해 합산하면 초기 forward가 같아도 다음 업데이트가 달라진다. 수정 후 BF16 길이 769의 1,270개 비교 텐서가 모두 일치한다.

| 실제 코퍼스 재개 검사 | 업데이트 | 가중치 + Adam/RNG + 마지막 기울기 | 결과 |
|---|---:|---:|---|
| Cairn 1.320B | 1000 → 1003 | 7,417 | 전부 정확히 일치 |
| GDN-H 1.314B | 1000 → 1002 | 7,387 | 전부 정확히 일치 |
| KDA+MLA 1.329B | 500 → 502 | 7,417 | 전부 정확히 일치 |
| GDN+GQA 1.323B | 1000 → 1002 | 7,327 | 전부 정확히 일치 |

미니배치 16, 누적 2, 최대 문맥 4096의 실제 데이터 배치를 사용했다. 총 **29,548개 텐서**와 각 단계의 NLL·router loss·gradient norm·LR·유효 토큰 수·expert load가 일치한다. 축소형 6개 계열의 초기화·중간 출력·전체 기울기·3회 AdamW 업데이트 6,632개 텐서도 최적화 경로에서 일치했다. 이는 시험한 입력과 업데이트의 결과이며 1B-token 전체 궤적에 대한 형식 증명은 아니다.

순차 역전파는 `Y_i = L_i + R_i S_i`, `S_(i+1) = A_i S_i + B_i`에 대해 다음 VJP를 계산한다.

```text
dL_i = dY_i
dR_i = dY_i S_i^T
dA_i = dS_(i+1) S_i^T
dB_i = dS_(i+1)
dS_i = A_i^T dS_(i+1) + R_i^T dY_i
```

이는 행렬곱 미분과 chain rule의 직접 적용이다. nonzero 초기 상태, scalar/channel decay, 길이가 chunk의 배수가 아닌 경우, chunk 1/4/8/16에서 출력·최종 상태·모든 입력 기울기 64개를 순차 참조와 FP64 `rtol=1e-7, atol=1e-8`로 확인했다.

CUDA 소스는 빌드할 때 로컬 NVIDIA NVRTC로 B200용 cubin을 만들고 실행 파일에 포함한다. worker는 커널을 다운로드하거나 외부 코드를 실행하지 않는다. 커널을 빌드하는 도구와 학습 실행 경로는 분리한다. [NVIDIA NVRTC 문서](https://docs.nvidia.com/cuda/nvrtc/)

## 무엇을 옮겼는가

| 경로 | Rust 구현 |
|---|---|
| Cairn 및 5개 제어군 | `src/model.rs`: GDN/KDA, MLA/GQA, RoPE, RMSNorm, dense/Sparse SwiGLU, shared experts, grouped routing |
| Delta recurrence | `src/recurrence.rs`: scalar/channel decay, chunk solve, 초기 상태, 모든 입력의 역전파 |
| 학습 | `src/trainer.rs`: BF16 autocast, FP32 가중치·기울기·Adam 상태, 미니배치 누적, 토큰 기준 LR, 마지막 토큰 예산 마스킹 |
| 메모리 절약 | Rust에서 블록을 재계산하고 역순으로 기울기를 전파한다. 전체 final RMSNorm과 vocabulary loss 분할의 합산 순서를 보존한다. |
| 옵티마이저 | `src/optimizer.rs`, `native/bridge.cpp`: 같은 native fused AdamW 및 foreach clipping |
| 저장·재시작 | `src/checkpoint.rs`: safetensors, 제한된 JSON, 해시, 문서 cursor, CPU/CUDA RNG, 최근 2개 checkpoint |
| 학습 데이터 | `src/data.rs`, `src/prepare.rs`: 로컬 mmap, 같은 순열·문서 경계·padding 규칙, 로컬 Parquet 준비 |
| 토크나이저 | `src/tokenizer.rs`: 고정 32K tokenizer JSON을 native Rust `tokenizers 0.22.2`로 읽고, 별도의 기존 byte-BPE도 재구현 |
| 평가 | `src/benchmarks.rs`: validation/test NLL 및 CairnLocal-v1 BoolQ·PIQA·ARC 후보 점수 |
| 기존 packed 추론 | `src/packed.rs`: NVFP4 E2M1, 저장된 scale, 순차 GDN, MLA, correction-bias routing, MTP |
| 실행 | `cairn-run`: 외부 통신을 차단한 뒤 worker 실행. `cairn-control`: 상태 확인·GPU별 순차 실행·GPU 사용 중이면 중단 |

기존 논문 작성·문헌 조사·과거 정리 기록은 모델 실행 경로가 아니다. 공개 데이터/패키지를 외부에서 준비하는 단계도 모델 worker와 분리한다. Rust worker에는 다운로드·Hub·telemetry·원격 코드 로더·Python 실행 기능이 없다.

## 수치 검증 결과

이 절은 [v1 검증 요약](evidence/validation_summary.json)의 과거 시험 범위다. v2의 큰 배치 검사와 최적화 결과는 위 절 및 [최적화 요약](evidence/optimization_summary.json)에 별도로 기록한다.

| 시험 | 결과와 범위 |
|---|---|
| 6개 계열, FP32 CPU | 각 계열 초기화·블록 출력·logits·router·모든 파라미터 기울기·3회 AdamW 업데이트가 비트 단위 일치 |
| 6개 계열, BF16 CUDA | 같은 항목이 비트 단위 일치. 일반 역전파와 Rust 블록 재계산 모두 확인 |
| 실제 Cairn 1.3B | **1,320,163,008 파라미터**, 우리 학습의 step-500 checkpoint, 입력 `[1,65]`: 출력·손실·전체 기울기 **1,487개 텐서** 비트 단위 일치 |
| 긴 입력 | Cairn 축소형, CPU, 입력 `[2,769]`, 여러 vocabulary loss 구간과 3회 업데이트: **1,270개 텐서** 비트 단위 일치 |
| checkpoint 재시작 | 메모리의 모델·Adam 상태를 폐기하고 Rust checkpoint에서 복원한 다음 업데이트가 Python 연속 실행과 일치 |
| 실제 학습 명령 | 고정 코퍼스 일부, 32K 어휘, 4,273,320 파라미터, BF16, 누적 2회, 총 400 유효 토큰. 가중치·Adam·CPU/CUDA RNG와 학습 지표가 일치 |
| 데이터·토크나이저 | 서로 다른 cursor/context의 실제 배치 6개, 한글·Unicode 등 입력 8개, 기존 byte-BPE 학습/복호화, 후보 continuation 마스크 일치 |
| Delta-rule 방정식 | 순차 Python 참조와 scalar/channel decay, nonzero 초기 상태, 4개 chunk 크기, 출력/최종 상태/모든 입력 기울기 비교. FP64 `rtol=1e-7, atol=1e-8` |
| Packed FP4·MTP | 양수/0/음수 input scale, mask/position, conv bias, 정규화 on/off, SiLU/GELU/ReLU 조건의 logits와 draft 일치 |
| 벤치마크 준비 | BoolQ 3,270 + PIQA 1,838 + ARC 1,172 = **6,280개** JSON 항목의 의미상 내용 전부 일치. 공백 등 직렬화 방식은 다를 수 있다. |

BF16의 RMSNorm에서는 `x.float()` 두 번을 한 번으로 합치는 것만으로도 역전파 반올림이 달라졌다. 이를 Python과 같은 두 개의 변환 경로로 보존했다. 긴 입력에서는 final norm의 기울기를 전체 길이에 대해 한 번 합산하고, loss tile 역전파 순서도 맞췄다. 새 학습의 checkpoint RNG까지 맞추기 위해 Python 생성자의 전역 난수 소비와 이후 모듈별 독립 초기화를 모두 보존한다.

위 v1의 1.3B 검증은 입력 길이 65였다. v2는 실제 문맥 4096 프로토콜의 배치까지 검사했지만, 전체 학습 예산의 모든 업데이트가 비트 단위로 같다는 주장은 하지 않는다. 새로 전체 코퍼스/BPE를 재생성하는 과정의 전수 동등성도 확인하지 않았다. 현재 비교 실험은 기존의 해시로 고정된 코퍼스와 tokenizer를 그대로 사용한다. 원본 의심 저장소의 pretrained tokenizer/weights는 쓰지 않는다.

## Rust가 더 빠른가

언어를 바꾸는 것만으로 GPU 학습이 빨라진다고 보장할 수 없다. Python도 행렬 연산은 native CUDA에서 처리한다. GPU 커널, expert dispatch, memory traffic, batch와 context가 실제 속도에 크게 영향을 준다.

작은 CPU 모델에서 같은 입력 `[2,128]`, 8 threads, 준비 업데이트 2회 뒤 10회 측정한 중앙값은 다음과 같다. 초기화·파일 저장 시간은 제외하고 forward/backward, clipping, AdamW를 포함한다. 두 구현의 측정 중 NLL 값은 같았다.

| CPU 파일럿 | 단계당 시간 | 유효 토큰/초 |
|---|---:|---:|
| Python | 45.20 ms | 5,663 |
| Rust | 30.57 ms | 8,375 |

이 시험에서는 Rust 처리량이 약 **1.48배** 높았다. **210,472 파라미터의 작은 CPU 모델과 공유 서버에서의 결과**이며, B200·1.3B의 속도 향상 비율로 인용하면 안 된다. v2에서는 GPU 4의 기존 학습 계산을 정지하고 Python–Rust–Rust–Python 순서로 실제 Cairn 1.3B를 별도 측정한다. CPU/디스크는 공유 서버이므로 전용 서버에서의 결과와 구분한다.

Rust CUDA 검증은 우리가 사용 중인 GPU 4에서 실행했다. GPU 4와 서버의 CPU/디스크를 공유한 검증 구간은 기존 실험의 처리량 비교에서 제외한다. [간섭 구간 기록](evidence/runtime_interference.json)에 보수적인 step 범위를 남겼다. 토큰 순서·유효 토큰 예산·학습 설정과 기존 checkpoint는 바꾸지 않았다.

## B200 실측 속도

동일한 Cairn 1.320B checkpoint와 실제 데이터 배치에서 Python–Rust–Rust–Python 순서로 비교했다. 각 실행은 10 update이며 초기 2회를 제외했다. 구현당 16회 측정, 총 542,988 유효 토큰을 계산했다. 네 실행의 **10단계 NLL·gradient norm·router 지표·LR·유효 토큰 수가 모두 일치**했다. [측정 원자료 요약](evidence/optimization/timing.json)

| 지표 | Python | Rust + 검증된 최적화 |
|---|---:|---:|
| update 시간 중앙값 | 6.985초 | 5.054초 |
| 전체 측정 토큰 / 측정 시간 | 4,985 tokens/s | 6,820 tokens/s |
| 최대 할당 GPU 메모리 | 36.67 GiB | 109.69 GiB |

측정한 처리량은 **1.368배**, 총 update 시간은 **26.9% 감소**했다. 모델 초기화·평가·checkpoint 저장 시간은 제외한다. GPU 4의 다른 계산은 정지했지만 CPU/디스크와 다른 GPU는 공유한 서버 측정이다. 재계산을 줄이려고 메모리를 더 사용한 결과도 포함하므로 언어 변경만의 이득이 아니다. 초기 Rust v1의 별도 비교에서는 약 5% 차이였다.

전체 학습 파이프라인이나 모든 모델·배치에서 같은 배율을 보장하지 않는다. 정수 라우팅 외에 행렬곱/attention/원소별 연산은 기존 CUDA backend를 사용한다. 전문가별 작은 행렬곱 호출과 padding 처리에는 추가 개선 여지가 있다. 수치 합산 순서나 통제된 데이터 배치를 바꾸는 최적화는 별도 검증이 필요하다.

## 170 GiB 목표 적용 — 2026-11-03

사용자의 요청에 따라 GPU 4–7의 현재 모델 네 개에 대해 중간 계산 결과의 보관량을 늘렸다. 같은 worker와 모델·데이터·배치·학습률·학습 예산을 유지한다. 실측 B200 총 용량은 179.06 GiB이다. 블록 수로 메모리 사용량을 조절하므로 아래 수치는 고정 할당량이 아닌 측정 최댓값이다.

| GPU | 모델 | 보관 블록 수 | 반복 시험 최대 GPU 점유 | 이전 Rust 설정 대비 처리량 |
|---|---|---:|---:|---:|
| 4 | Cairn | 8 → 14 | 168.47 GiB | +8.84% |
| 5 | GDN-H recipe proxy | 8 → 14 | 165.76 GiB | +9.34% |
| 6 | KDA + MLA operator control | 2 → 5 | 169.66 GiB | +3.56% |
| 7 | GDN + GQA | 8 → 14 | 168.10 GiB | +8.51% |

재개 후 2026-11-03 02:26 CET 확인에서 GPU 점유량은 약 166.1–170.5 GiB였다. KDA는 캐시 예약 증가로 짧은 시험의 최대치보다 약 0.8 GiB 높게 표시됐다. 실제 텐서 최대 할당량은 약 163.62 GiB이고, 네 GPU의 남은 공간은 약 8–12 GiB였다. 메모리 사용량은 이후 배치와 캐시 상태에 따라 달라질 수 있다.

각 모델의 같은 체크포인트와 같은 실제 데이터에서 기존–후보–후보–기존 순서로 비교했다. 중간의 후보 탐색 실행은 최종 속도 집계에서 제외했다. 각 실행 10 update 중 초기 2개를 제외하여 설정당 16 update의 총 유효 토큰/총 시간을 사용한다. 표의 GPU 점유는 `nvidia-smi`로 측정한 예약 메모리와 실행 환경을 포함하며, 위의 이전 표에 있는 텐서 할당량과 구분한다. 네 실행의 10개 update별 NLL·gradient norm·router loss·LR·유효 토큰·expert load가 모두 일치했다. 10 update 후 가중치·Adam/RNG 상태·마지막 기울기 **29,548개 텐서**는 safetensors 바이트와 저장 시 SHA-256으로 동일성을 확인했다. 검사한 실행에 대한 결과이며 전체 학습 궤적의 형식 증명은 아니다.

정지 요청으로 현재 update를 끝낸 뒤 저장했으며 Cairn/GDN-H/KDA/GDN+GQA를 각각 **1273/1801/650/1806**에서 재개했다. 이번 변경에서 미저장 update 손실은 **0**이다. 시험은 별도 폴더에서 실행했고 실제 학습 이력에 시험 update를 포함하지 않는다. 모델 worker SHA-256은 이전과 같은 `d2286b751d25d5d8e8ccb47adc762719d982569f1b974611d3796e90de3e97dd`이다. 기존 통신 차단도 유지한다.

당시 대기 중인 Dense GQA와 Full MLA는 기존의 보관 블록 수 8개/4개를 유지했다. 11월 3일에는 GPU 5에서 GDN-H 후 Dense GQA, GPU 7에서 GDN+GQA 후 Full MLA 순서였으며, 11월 4일에는 위 GPU 4 순차 대기열로 전환했다. 이 두 모델도 별도의 짧은 실행으로 메모리 적합성과 ETA용 처리량을 확인했다.

[검증·속도·적용 요약](evidence/retention_170gib_20261103/summary.json), [재개 후 실제 상태](evidence/retention_170gib_20261103/live_snapshot.json), [ETA 계산 근거](evidence/retention_170gib_20261103/eta.json)에 원자료를 남겼다. 검증용 Python 도구는 고정한 Rust 실행 파일을 관리하거나 결과를 읽으며 모델 학습 연산을 수행하지 않는다.

모델당 10억 유효 토큰 학습 완료까지의 초기 ETA는 Cairn 약 **32시간**, GDN-H 약 **31시간**, KDA 약 **69시간**, GDN+GQA 약 **31시간**이었다. 당시 대기 시간을 포함하면 Dense GQA 약 **52시간**, Full MLA 약 **72시간**이었다. 이는 2026-11-03 02시대 CET의 네 GPU 병렬 계획이며, 이후 일시 정지와 11월 4일 단일 GPU 순차 전환 때문에 현재 완료 시각으로 사용할 수 없다. `eta.json`은 당시 계산 근거를 보관한다.

ETA는 이전 Rust 구간 최대 200 update에 측정된 속도 개선을 적용하고 새 실행 구간을 합산한다. 주기적 평가·저장에 10% 시간을 더한 계획값이며, 현재 모델 ±15%, 대기 모델의 추가 실행 시간 ±25% 범위를 함께 기록했다. 이는 통계적 신뢰구간이 아니다. 대기 모델은 10 update의 측정만 있으므로 불확실성이 더 크다. 최종 held-out/벤치마크 평가 시간과 그에 따른 다음 모델의 시작 지연은 별도이며 서버 부하에 따라 ETA가 달라진다.

## 실행 방법

현재 검증된 실행 환경은 **zinal의 Linux + B200**이다. macOS native 실행은 이번 검증 범위에 포함하지 않으며, 지원하지 않는 OS에서는 통신 차단을 생략하지 않고 실행을 거부한다. 아래 명령은 서버 Rust 폴더에서 사용한다.

```bash
# Python 없이 CPU 연산과 OS 차단 점검
./releases/20261103-v4/bin/cairn-run cpu probe

# 모델을 만들지 않고 파라미터 수 확인
./releases/20261103-v4/bin/cairn-run cpu count configs/cairn_1p3b.json

# 현재 진행 중인 기존 실험 상태 읽기
./releases/20261103-v4/bin/cairn-control status /scratch/hongse/mythos/cairn-rust/runs
```

새 학습에는 별도 run 폴더를 사용한다. GPU 4–7에서 Rust 실험이 진행 중이므로 아래 새 queue 예시는 해당 GPU가 비었을 때 사용한다. 관리 도구는 사용 중인 GPU에 추가 학습을 올리지 않는다.

```bash
./releases/20261103-v4/bin/cairn-control queue runs 4 configs/cairn_1p3b.json

# 이미 저장된 실행을 이어갈 때
./releases/20261103-v4/bin/cairn-run 4 train configs/cairn_1p3b.json runs/cairn_1p3b_seed17_v1 --resume

# checkpoint를 불러 평가만 할 때
./releases/20261103-v4/bin/cairn-run 4 train configs/cairn_1p3b.json runs/cairn_1p3b_seed17_v1 --resume --evaluate
```

`train --profile --steps N`은 저장·최종 평가를 생략하는 별도 측정 실행이다. 기존 학습 run에서 측정하지 않는다. `prepare-data`와 `prepare-benchmarks`는 이미 로컬에 준비된 자료만 읽는다. `packed-infer`는 로컬 config/weights/입력 safetensors에서 logits 또는 `--draft` 출력을 만든다. 각 명령은 `--help`로 인자를 확인할 수 있다.

## 빌드와 보안 경계

빌드는 서버에 준비된 Rust 1.99.1, 고정한 `Cargo.lock`, 로컬 LibTorch 2.11.0을 사용한다. 패키지를 준비한 이후에는 `build_offline.sh`가 `cargo build --release --locked --offline`으로 빌드한다. 버전 검사를 우회하거나 LibTorch를 자동으로 다운로드하지 않는다.

```bash
export LIBTORCH=/scratch/hongse/mimo_ising_20261016/venv/lib/python3.12/site-packages/torch
export CUDA_HOME=/usr/ela/local/linux-local/cuda-12.8
bash build_offline.sh
```

위 경로에 `python3.12`라는 디렉터리 이름이 있지만 사용 대상은 그 안의 **native headers와 공유 라이브러리**다. worker는 `libpython`/`libtorch_python`을 연결하지 않고, launcher는 tensor 라이브러리도 연결하지 않는다.

`cairn-run`은 환경변수와 상속된 추가 descriptor를 정리하고 worker를 실행하기 전에 seccomp를 적용한다. 처음부터 IP 소켓·raw packet·io_uring을 차단한다. NVIDIA 드라이버 초기화에 필요한 로컬 UNIX IPC만 초기 단계에 허용하고, CUDA 초기화 직후 connect/bind 및 socket send/receive와 새 프로그램 실행을 차단한다. 기존 thread에도 같은 정책을 적용한다. CPU/GPU `probe`에서 OS 거부를 직접 확인한다.

가중치·optimizer는 제한된 JSON과 safetensors만 읽는다. pickle·TorchScript·Hub·원격 함수·실행 가능한 모델 설정을 받지 않는다. 경로, 파일 종류, 크기, tensor shape/dtype/offset 및 checkpoint 해시를 확인한다. 모델 배분 예산을 먼저 검사한다. 이는 사용한 라이브러리·OS·GPU 드라이버 전체에 결함이 없다는 증명은 아니다.

검증 전용 `verification/run_oracle.sh`만 보존한 Python 구현을 실행한다. Rust worker/launcher는 이를 호출하지 않는다. 검증 자료에 원본 의심 저장소의 코드·가중치를 사용하지 않는다.

## 체크포인트에서 정지·재개

실행 폴더에 로컬 `stop.request` 파일을 만들면 현재 update를 마친 뒤 checkpoint를 저장하고 종료한다. queue는 뒤의 모델을 자동 시작하지 않는다. 재개 전에는 이 요청 파일을 제거하고 `--resume`을 사용한다. `run.lock`이 있다면 해당 PID가 종료되었는지 먼저 확인해야 한다. 진행 중인 worker의 바이너리를 교체하지 않고 별도의 읽기 전용 release를 사용한다.

새 queue의 공통 옵션은 `--resume --optimized --fused-scan --deterministic --retained-blocks N`이다. GPU 5의 Dense GQA와 GPU 7의 Full MLA 순서는 유지한다. 관리 도구는 대기 모델의 기존 설정을 유지하도록 Dense GQA의 보관 블록 수를 최대 8개, Full MLA를 최대 4개로 제한한다.
