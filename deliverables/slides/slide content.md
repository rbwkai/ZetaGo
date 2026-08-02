## ZetaGo: Slide Content Design

---

## Slide 1: Title Slide

### ZetaGo: Move Prediction and Position Evaluation in 7x7 Go



* **Project Title**: ZetaGo: Move Prediction and Position Evaluation in 7x7 Go


* **Team Name**: Team ZetaGo


* **Team Members**: ZetaGo Research Group


* **Submission Date**: 2 August 2026



---

## Slide 2: Problem and Motivation (1/2)

### The Challenge of Board Representation & Capacity

* **The Core Task**: Predict KataGo's move selection (policy prediction across 50 discrete output classes) and score margin (value regression) on 7x7 Go boards.


* **Feature Engineering vs. Deep Capacity**: Do deep neural networks require hand-crafted spatial features (e.g., liberty counts, history planes) or can convstacks extract higher-order spatial features natively from raw stone positions?


* **Data Scalability**: Are classical models constrained by training volume or by internal model capacity?


* **Search vs. Pure Policy**: How much playing strength originates from policy network imitation versus online inference search (MCTS)?



---

## Slide 3: Problem and Motivation (2/2)

### Core Research Questions

* **Factor A (Feature Richness)**: How do feature inputs $N \in \{2, 4, 7\}$ impact classical linear baselines versus convolutional stacks?


* **Factor B (Dataset Volume)**: How do model architectures scale as dataset volume increases from 1,000 to 107,969 games (3.3M positions)?


* **Supervised Latent Geometry**: How do task-supervised CNN latents compress board states and structure game-outcome clusters?


* **Methodological Rigor**: Can policy performance claims survive game-disjoint test splits and formal statistical hypothesis testing (ANOVA, McNemar)?



---

## Slide 4: Related Work

### Contextualising ZetaGo

* **Prior Work**:
* **AlphaGo / AlphaGo Zero / KataGo**: Established policy-value convolutional architectures combined with Monte Carlo Tree Search (MCTS).


* **Classical ML Baselines**: Random Forests, k-NN, SVMs, and Logistic Regression historically used for featurized game representations.




* **Gaps in Existing Research**:
* Prior studies rarely isolate whether spatial feature planes are redundant when using deep convolutional stacks.


* Lack of compute-matched capacity comparisons between classical ML models and deep CNNs across scaling volumes.




* **ZetaGo's Contributions**:
* Systematic $5 \times 3 \times 4$ factorial study (5 models $\times$ 3 encodings $\times$ 4 dataset volumes).


* Strictly enforced game-disjoint sealed test split evaluated under single-write guards.


* Representation analysis evaluating how supervised CNN latents organize game-state geometries.





---

## Slide 5: Dataset Description

### Corpus Structure & Baseline Traps

* **Corpus Source**: 120,000 KataGo self-play games on a 7x7 board with komi 9.5.


* **Split Partitioning**: Pairwise game-disjoint splits generated via CRC32 hashing:


* **Train**: 3,310,156 positions (107,969 games).


* **Validation**: 184,834 positions (6,027 games).


* **Sealed Test**: 182,102 positions (6,004 games).




* **Class Distribution & Key Baselines**:
* **50 Output Classes**: 49 board intersections + 1 `pass` action.


* **Majority-Class Baseline**: **9.17%** top-1 accuracy (predicting `pass`).


* **Uniform Random Legal Baseline**: **3.50%** top-1 accuracy.


* **Side-to-Move Value Baseline**: **86.44%** win/loss sign accuracy ("predict win iff White is to move", due to White's 91% win rate under komi 9.5).





> **Dataset Challenge**: At full volume, each unique position appears ~58 times. Scaling games by 108$\times$ yields only a 10.9$\times$ increase in unique positions (56,798 unique positions total).
> 
> 

---

## Slide 6: Methodology (1/2)

### Feature Encodings & Model Architectures

* **Factor A Feature Encodings**:
* **$N=2$ (Raw Stones)**: Player stone occupancy + opponent stone occupancy.


* **$N=4$ (Tactical Planes)**: $N=2$ + liberty count planes + side-to-move turn indicator.


* **$N=7$ (Historical Context)**: $N=4$ + 2 move history planes + occupancy change plane.




* **Evaluated Model Families**:
* **CNN**: Dual-head architecture (50-class policy softmax head + continuous score margin regression head).


* **Classical Baselines**: Random Forest, k-NN, SVM, Logistic Regression (trained on compute-matched 80,000-row subsample).


* **Inference Search**: 100-simulation PUCT MCTS algorithm wrapping frozen policy/value network weights.





---

## Slide 7: Methodology (2/2)

### Verification Protocol & Evaluation Strategy

* **Evaluation Metrics**:
* **Policy Tasks**: Top-1 Accuracy, Top-3 Accuracy, and Macro-F1 across all 50 classes.


* **Value Tasks**: Score Margin MSE/MAE and Derived Sign Accuracy (`sign(predicted margin) == sign(true margin)`).




* **Sealed Test Set Guardrails**:
* **Single-Write Enforcement**: Script refuses to overwrite existing output files without explicit flags.


* **No Default Split**: Explicit `--split test` argument required.




* **Statistical Verification Strategy**:
* **Two-Way ANOVA**: $5 \times 3 \times 5$ design (Model $\times$ Encoding $\times$ Seed) evaluating interaction effects.


* **Exact McNemar Test**: Binomial test on paired, identical positions to evaluate CNN vs. Random Forest.





---

## Slide 8: Experiments and Results (1/2)

### Sealed Test Split Performance Benchmarks

* **Configuration**: $N=4$ Encodings, Full Volume (107,969 games), 5 Seeds (Mean ± SD):



| Model | Top-1 Acc | Top-3 Acc | Macro-F1 | Value Acc | Margin MAE | Margin MSE |
| --- | --- | --- | --- | --- | --- | --- |
| **CNN**<br> | **90.37% ± 0.05%** | **98.72% ± 0.01%** | **0.9158 ± 0.0007** | 92.06% ± 0.66% | 3.430 ± 0.023 | **63.92 ± 0.28** |
| **Random Forest**<br> | 87.52% ± 0.03% | 96.00% ± 0.05% | 0.8826 ± 0.0005 | **93.43% ± 0.13%** | **3.339 ± 0.014** | 67.50 ± 0.13 |
| **k-NN**<br> | 81.37% ± 0.30% | 93.64% ± 0.22% | 0.8110 ± 0.0017 | 92.65% ± 1.00% | 3.524 ± 0.016 | 74.83 ± 1.07 |
| **Logistic Regression**<br> | 79.63% ± 0.17% | 91.96% ± 0.05% | 0.7952 ± 0.0015 | 71.45% ± 0.66% | 6.236 ± 0.010 | 110.89 ± 0.02 |
| **SVM**<br> | 79.12% ± 0.14% | 91.57% ± 0.08% | 0.7878 ± 0.0017 | 77.27% ± 0.73% | 5.531 ± 0.002 | 122.29 ± 0.42 |
| *Majority-Class Floor*<br> | *9.17%* | — | *~0.003* | — | — | — |
| *Side-to-Move Baseline*<br> | — | — | — | *86.44%* | — | — |

* **Key Finding**: Zero generalisation gap — every model's test accuracy lands within 0.002 of its validation accuracy.



---

## Slide 9: Experiments and Results (2/2)

### Formal Statistical Tests & Search Performance

* **Factor A Interaction (Two-Way ANOVA)**:
* Model Main Effect: $F(4, 60) = 14898, p = 3.7 \times 10^{-89}$.


* Encoding Main Effect: $F(2, 60) = 3361, p = 2.5 \times 10^{-62}$.


* **Interaction (Model $\times$ Encoding)**: $F(8, 60) = 1295, p = 3.4 \times 10^{-64}$ (accounts for 13.5% of total variance).




* **CNN vs. RF Crossover (Exact McNemar Test)**:
* 8,847 discordant test positions: CNN correct / RF wrong **7,130** times vs. RF correct / CNN wrong **1,717** times (**4.15 : 1 ratio**, $p < 10^{-300}$).




* **Inference-Time MCTS Tournament**:
* Head-to-Head (150 games, 4 random opening plies):


* **MCTS (100 sims) vs. Greedy Policy**: MCTS wins **45–5** (**+401.4 Elo gain** over greedy policy, $p = 4.1 \times 10^{-3}$).







---

## Slide 10: Analysis and Discussion (1/2)

### Core Theoretical Insights

* **1. Feature Planes and Neural Capacity are Substitutes**:
* Adding feature planes ($N=2 \rightarrow N=7$) improves linear models (SVM/LogReg) by **+0.09 to +0.11** top-1 accuracy.


* For the CNN, extra feature planes provide **+0.0008** improvement (inside noise floor). Convolutional layers compute connected spatial features natively.




* **2. Capacity Limits vs. Data Limits**:
* Classical models plateau after 5,000 games (+0.002 top-1 gain moving 5k $\rightarrow$ 108k games, despite 4.5$\times$ unique positions).


* The CNN continues climbing (+0.027 top-1 gain over same range), proving classical baselines are capacity-limited.




* **3. Search Dominates Architecture**:
* Wrapping static CNN weights in 100-simulation PUCT search yields a ~400 Elo strength gain, exceeding all offline training adjustments.





---

## Slide 11: Analysis and Discussion (2/2)

### Model Failures & Representation Analysis

* **Where Linear Models Fail**:
* Logistic Regression (71.45%) and SVM (77.27%) fall substantially below the trivial side-to-move heuristic (86.44%). Score margin regression requires non-linear boundary learning.




* **Policy Gap Analysis**:
* CNN's top-3 accuracy reaches **98.72%** (an 8.35 point gap over top-1). Errors occur when ranking several similarly strong candidate moves rather than tactical misses.




* **Supervised CNN Latent Geometry**:
* The supervised CNN discards raw board state details (low $R^2$ for raw stone counts) to build tight outcome-based clusters (high silhouette score of 0.468 at $k=3$).


* Internal representations prioritise game-winning tactical features over literal pixel-level reconstruction.





---

## Slide 12: Limitations

### Methodological Boundaries

* **Imitation Fidelity vs. Game Skill**: Labels measure KataGo move choice (imitation accuracy), not absolute optimal game play.


* **Single-Distribution Scope**: Evaluation measures generalisation to unseen games from the *same* generation run. Does not establish transfer to human games or larger boards (9x9/19x19).


* **Compute-Matched Subsampling**: Classical baselines were trained on an 80,000-row subsample due to memory/scaling limits, whereas CNN trained on all 3.3M rows.


* **Value Head Margin**: CNN value accuracy (92.06%) beats the baseline (86.44%) by only 5.62 percentage points.


* **Positional Redundancy**: Positional duplicates (~58 rows per unique position) inflate absolute accuracy across all models equally.



---

## Slide 13: Conclusion and Future Work

### Summary & Next Steps

* **Summary of Key Findings**:
* Deep CNNs achieve **90.37% top-1** move prediction on held-out test data without needing hand-crafted feature planes.


* Classical models face strict capacity ceilings and fail on score margin estimation.


* Online MCTS search provides the largest single playing strength boost (+400 Elo).




* **Future Work**:
* **Unique Position Training**: Benchmark performance on fully deduplicated datasets (`--dedup unique`).


* **Domain Transfer**: Test generalisation against human game databases (OGS/KGS 7x7 games).


* **Deeper Residual Architectures**: Evaluate deeper ResNet backbones to scale representation capacity on larger boards.





---

## Slide 14: References

### Cited Literature & Frameworks

* Silver, D., et al. (2017). *Mastering the game of Go without human knowledge*. Nature, 550(7676), 354-359.


* Wu, D. (2019). *KataGo: Research and Open-Source Go Engine*. GitHub repository.


* McNemar, Q. (1947). *Note on the sampling error of the difference between correlated proportions or percentages*. Psychometrika, 12(2), 153-157.


* ZetaGo Research Team. (2026). *Sealed Test Set Evaluation and Representation Analysis Benchmark Results*.