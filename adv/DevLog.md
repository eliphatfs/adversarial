# DevLog

## May.18
- Frank-Wolfe, 20 steps, adaptively amplified.
  + model1. Natural Acc: 0.94290, Robust acc: 0.00000, distance: 0.03137.
  + model2. Natural Acc: 0.83020, Robust acc: 0.47820, distance: 0.03137.
  + model3. Natural Acc: 0.80330, Robust acc: 0.44640, distance: 0.03137.
  + model4. Natural Acc: 0.84920, Robust acc: 0.53580, distance: 0.03137.
  + model5. Natural Acc: 0.81430, Robust acc: 0.52290, distance: 0.03137.
  + model6. Natural Acc: 0.88250, Robust acc: 0.61300, distance: 0.03137.
- Frank-Wolfe, 20 steps, amplified.
  + model1. Natural Acc: 0.94290, Robust acc: 0.60320, distance: 0.03137.
  + model2. Natural Acc: 0.83020, Robust acc: 0.48110, distance: 0.03137.
  + model3. Natural Acc: 0.80330, Robust acc: 0.56340, distance: 0.03137.
  + model4. Natural Acc: 0.84920, Robust acc: 0.53500, distance: 0.03137.
  + model5. Natural Acc: 0.81430, Robust acc: 0.53000, distance: 0.03137.
  + model6. Natural Acc: 0.88250, Robust acc: 0.60920, distance: 0.03137.

## May.17
- Frank-Wolfe, same complexity as PGD, 20 steps.
  + model1. Natural Acc: 0.94290, Robust acc: 0.00000, distance: 0.03137.
  + model2. Natural Acc: 0.83020, Robust acc: 0.50250, distance: 0.03137.
  + model3. Natural Acc: 0.80330, Robust acc: 0.63120, distance: 0.03137.
  + model4. Natural Acc: 0.84920, Robust acc: 0.55300, distance: 0.03137.
  + model5. Natural Acc: 0.81430, Robust acc: 0.53460, distance: 0.03137.
  + model6. Natural Acc: 0.88250, Robust acc: 0.63790, distance: 0.03137.

## May.14
**Stumbled on an attack method**
[github link](https://github.com/Line290/FeatureAttack)

**Paper of Model 5**
[Boosting Adversarial Training with Hypersphere Embedding](https://arxiv.org/pdf/2002.08619.pdf)

[github repo](https://github.com/ShawnXYang/AT_HE)

- Feature Normalization and Weight Normalization, FN and WN
  - Normalize feature and weight at the second-last layer (layer before softmax)
  - When both feature and weight are normalized, $out = \cos(\theta) = \frac{Wz}{\|W\|\|z\|}$
    - Is this equivalent to the `CosineLinear` layer in `model3`?
    - Seems that `model5` did not normalize either of feature or weight. So why does `model5` work? Did it use AM and AT only?
- Angular Margin, AM
  - Performed only in training.
  - Modifies CE loss
  - $-\mathbf{1}_y^T\log Softmax\left(s\cdot \left(\cos\theta - m\cdot \mathbf{1}_y\right)\right)$
  - Adds a margin $m$ to softmax.
  - Similar to the margin in SVM. Improves robustness.

**Paper of Model 6**
[Adversarial Weight Perturbation Helps Robust Generalization](https://www.researchgate.net/profile/Dongxian-Wu/publication/349101174_Adversarial_Weight_Perturbation_Helps_Robust_Generalization/links/601fe92f92851c4ed5560c53/Adversarial-Weight-Perturbation-Helps-Robust-Generalization.pdf)
- Observation: **Generalization gap**. Many AT models have **poor test robustness** as training epochs go up. i.e. Generalization of adversarial-trained networks need more data to train for robustness on test sets.
  - **Why is this?** The weight-loss landscape of AT models becomes sharper when training epochs goes up. Therefore small purtabations on the **weights** of the model may cause the model to produce poor output. This paper found that the weight-loss landscape is well-correlated to the generalization gap.
  - Most AT methods mainly focused on smoothing the **input-loss landscape** of models.
  - But some AT methods including TRADES (model4) implicitly smoothed **weight-loss landscape**.
- Proposed method: Adversarial Weight Purtabation.
  - Explicitly add the weight-loss landscape into the loss function as a regularization term.
  - This is done by generating purtabation $v$ on the weight $w$ of a model and minimize the loss of the model with pertubed weight $w + v$.
- Procedure:
  - Given a batch of data and a trained model.
    1. Generate adversarial example $x'$ (using PGD, etc.) for each sample in the batch. (maximize training loss of each sample w.r.t. input $x$)
    2. Generate pertubation $v$ on weight using all $x'$'s. (maximize training loss of the entire batch w.r.t. weight $w$)
    3. Re-train the model with pertubed weight $f(x; w+v)$. (update $w+v$)
    4. Update $w$ (remove pertubation $v$ from updated $w+v$), repeat until convergence.


## May.09
- Barrier method
  + Maybe not enough steps. Poor performance.
- Krylov subspace method by gradients
  + Merely reached PGD20 baseline...
  + model4. Natural Acc: 0.84920, Robust acc: 0.56030, distance: 0.03137.

## May.08
- Abstract Interpreter paper ref: http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf
- Graph tracer via torch.jit.
- Abstract Interpreter with hybrid zonotope domain from DiffAI.
- Implemented the prover. However, HBox is too expensive to run on the models, while Box with no correlations does not have enough analysis precision. Too bad!
- Sudden idea and rather interesting findings.
  + Since we have implemented a tracer we can tweak and play around. If ReLU is removed the problem is easily solvable. So let's remove it! Interesting thing is going on...
  + We relax ReLU to be leaky (that is, `max(0, x) * (1 - p) + x * p`). When `p` is set to 0.02 nothing seem to happen on any model. But when `p` is set to 0.2 or even larger...
  + model1. Natural Acc: 0.94596, Robust Acc: 0.05362 (sampled for test speed)
  + model3. Natural Acc: 0.80330, Robust acc: 0.19970 (fully run)
  + model4. Natural Acc: 0.86800, Robust Acc: 0.83200 (sampled for test speed)
  + model6. Natural Acc: 0.89800, Robust Acc: 0.86800 (sampled for test speed)
  + Obscured gradient gives false sense of security.
  + **Why is this, mathematically?**
- Analyzed activation values before ReLU
  + Robust models tend to have smaller activation values.
  + Obfuscated gradient model actually have very large activation values.
  + Before and after tweaking the ReLU (to be leaky) the trends of these activation values stay unchanged.
- Implemented Frank–Wolfe algorithm
  + model6. Natural Acc: 0.88250, Robust acc: 0.63790, distance: 0.03137. A bit better than PGD20, with same speed (DeepFool is a bit slower.)

## May.07
- Implemented DeepFool.
- Optimized DeepFool.
- Benchmarking model4
  + DeepFool. Natural Acc: 0.84920, Robust acc: 0.53810, distance: 0.03137
  + PGD20. Natural Acc: 0.84920, Robust acc: 0.56180, distance: 0.03137
- Baseline evaluation (PGD20)
  + model1. Natural Acc: 0.94290, Robust acc: 0.00040, distance: 0.03137
  + model2. Natural Acc: 0.83020, Robust acc: 0.51290, distance: 0.03137
  + model3. Natural Acc: 0.80330, Robust acc: 0.65150, distance: 0.03137
  + model4. Natural Acc: 0.84920, Robust acc: 0.56180, distance: 0.03137
  + model5. Natural Acc: 0.81430, Robust acc: 0.54820, distance: 0.03137
  + model6. Natural Acc: 0.88250, Robust acc: 0.64340, distance: 0.03137
- DeepFool evaluation
  + model1. Natural Acc: 0.94290, Robust acc: 0.00020, distance: 0.03137
  + model2. Natural Acc: 0.83020, Robust acc: 0.48050, distance: 0.03137
  + model3. Natural Acc: 0.80330, Robust acc: 0.31600, distance: 0.03137
  + model4. Natural Acc: 0.84920, Robust acc: 0.53810, distance: 0.03137
  + model5. Natural Acc: 0.81430, Robust acc: 0.52550, distance: 0.03137
  + model6. Natural Acc: 0.88250, Robust acc: 0.60990, distance: 0.03137
- A new idea
  + If you can not do something, prove that it is impossible.
  + We can use static program analysis techniques to make this happen.


## Apr.30
- [Thank you, Second Order!](./second_order_attack.py)

## Apr.16
- Performed comprehensive tests on model robustness
  1. model1: weak
  2. model2: normal. AT
  3. model3: gradient obfuscation at final cosine-linear layer. 
  4. model4: strong; pgd acc ~ 50; stochastic attack doesnt work. Trades loss
  5. model5: pgd acc ~ 50; stochastic attack partially work. Hypersphere embedding.
  6. model6: pgd acc ~ 60; . Adversarial Weight Pertubation.
- Implemented `SobolHappyAttack`
  - Enhanced stochastic black-box attack
- Defeated `model3`
  - By removing obfuscation in cosine-linear layer.

## Apr.09
- Investigated `model3`
  - Gradient outputs have many 0's, so in many cases gradient-based attacks will fail.
  - `model3` implements a `CosineLinear` layer
- Implemented `ChihaoHappyAttack`
  - Randomized black-box attack
  - Performs better than PGD on `model3`
  - `model3` may have some kind of gradient obfuscation