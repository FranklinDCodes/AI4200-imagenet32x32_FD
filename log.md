
### BASELINE — MNIST — 2026-10-27

**Result:** Acc: 0.9506


### BASELINE V2 — MNIST — 2026-10-28

**Change:** Baseline but with 30 epochs

**Result:** Acc: 0.9802


### BASELINE — cifar10 — 2026-10-28

**Result:** Acc: 48.61


### BASELINE — imagenet32 — 2026-10-28

**Result:** Acc: 0.0677



### MODEL_2 V1 — MNIST — 2026-10-27

**Change:** Used a weight initialization scheme I came up with that satisfies glorot but shifts the variance more to the matrix rows than columns

**Why:** Because I hope that this helps keep the neuron tasks from overlapping too much, and maybe they can learn unique tasks quicker

**Expected:** Pretty much a 30% chance this improves anything, more likely makes it worse I think.

**Result:** Acc: 0.2489. Huh.


### MODEL_2 V2 — MNIST — 2026-10-27

**Change:** I made the varince difference less extreme to see if that would help. From 0.1 and 2 to 1.0 and 2.

**Why:** Because clearly whatever happened in the last run didn't work, so I wanted to moderate it

**Expected:** I think just better than 24%

**Result:** Acc: went up to 90.59%. Seems like this weight init scheme is hurting if anything. I want to try one more time with more epochs to see if it can get more out of the model. To be fair though, I want to compare it to a baseline trained twice as long to see how it helps.


### MODEL_2 V3 — MNIST — 2026-10-28

**Change:** Same as V2 but training thrice as long

**Why:** Becuase I still suspect this weight init scheme could help reach a better accuracy, but I would be surprised if it would take longer to get there.

**Expected:** I think it will beat the .9059 from V2, but I don't know by how much. I would be pleasantly surprised if it beat either baseline.

**Result:** Acc: 0.9597. Not bad at all. Beats the original baseline, but doesn't beat just training the baseline longer for the same amount of time. This will lead me to give up on this weight init scheme I think.


### model_shift V1 — MNIST — 2026-10-28

**Change:** Trained for 30 epochs (seemed to work well before) and using a shift augmentation. This one makes 3 new copies of each instance with each row and col randomly shifted from -5 to +5 rows or columns by wrapping around.

**Why:** Working on learning spatial invariance.

**Expected:** I think it will beat the the 30 epoch baseline of 0.9802.

**Result:** Acc: 0.9777. I wonder if I made too big of shifts to be that useful, or even if I may be clipping out the number at times.


### model_shift V2 — MNIST — 2026-10-28

**Change:** Trained for 10 epochs.

**Why:** That took too long I need to make faster comparisons.

**Expected:** I think it will be less than the V1 acc of 0.9777.

**Result:** Acc: 


