# DevLog

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