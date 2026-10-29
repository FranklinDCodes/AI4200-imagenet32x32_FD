
### BASELINE — MNIST — 2026-10-27

**Result:** Acc: 0.9506


### BASELINE V2 — MNIST — 2026-10-28

**Change:** Baseline but with 30 epochs

**Result:** Acc: 0.9802


### BASELINE — cifar10 — 2026-10-28

**Result:** Acc: 0.4861


### BASELINE V2 — cifar10 — 2026-10-28

**Change:** Baseline but with 10 epochs

**Result:** Acc: 0.4535


### BASELINE — imagenet32 — 2026-10-28

**Result:** Acc: 0.0677



### MODEL_2 V1 — MNIST — 2026-10-27

**Change:** Used a weight initialization scheme I came up with that satisfies Glorot but shifts the variance more to the matrix rows than columns

**Why:** Because I hope that this helps keep the neuron tasks from overlapping too much, and maybe they can learn unique tasks quicker

**Expected:** Pretty much a 30% chance this improves anything, more likely makes it worse I think.

**Result:** Acc: 0.2489. Huh.


### MODEL_2 V2 — MNIST — 2026-10-27

**Change:** I made the variance difference less extreme to see if that would help. From 0.1 and 2 to 1.0 and 2.

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


### model_shift V2 (short) — MNIST — 2026-10-28

**Change:** Trained for 10 epochs.

**Why:** That took too long I need to make faster comparisons.

**Expected:** I think it will be less than the V1 acc of 0.9777.

**Result:** Acc: 0.9485


### model_shift V3 (lighter shift) — MNIST — 2026-10-28

**Change:** Shift max size 2.

**Why:** I think the shift was too much and I want it to not clip the numbers.

**Expected:** I think that it will beat certainly the V2 accuracy, and hopefully it will beat the 10 epoch baseline model accuracy of 0.9506.

**Result:** Acc: 0.9672. Hooray! Beat the baseline and the 5-shifted version.


### model_shift V4 (AdamW) — MNIST — 2026-10-28

**Change:** V3 but using AdamW and lr=0.003.

**Why:** I think AdamW is better than SGD generally and usually I've heard LR from [0.003, 0.0001] is good for Adam.

**Expected:** I think that this will beat the V3 accuracy of 0.9672. It possibly wont because it will converge slower, but I think it would be better in time.

**Result:** Acc: 0.9750. Yay! it beat it. If the dirty loss does better too, I'll combine them.


### model_shift V5 (dirty loss) — MNIST — 2026-10-28

**Change:** V3 but I am using a dirty cheater loss function. Basically the loss function is weighted more heavily towards classes that are more represented in the training data using the following formula:

            weights = 1 + z / 100 where z = (class_counts - class_counts.mean()) / class_counts.std()

**Why:** Trying to game the system basically.

**Expected:** I sort of hope that it will beat the V3 acc of 0.9672 but I think it's unlikely.

**Result:** Acc: 0.9661. Ok I am silly because I now realize that MNIST is perfectly balanced. I though maybe it was slightly unbalanced and was assuming that the train and test data would represent that slight unbalance in similar ways but I was poorly mistaken.


### model_shift V6 (reverse dirty loss) — MNIST — 2026-10-28

**Change:** Just like V3 or V5 but class weights as follows: weights = 1 - (class_counts - class_counts.mean()) / class_counts.mean(). This formula scales counts according to their distance from the mean.

**Why:** After I realized that MNIST is perfectly balanced, I realized I should be trying to rectify imbalances in the training data not exacerbate them.

**Expected:** Unless the weights are too strong, I thing they may show some small improvement over the 0.9672 acc for the shifted model with SGD. If it works better I will combine it with the AdamW version.

**Result:** Acc: 0.9696. This is a slight improvement


### model_shift V7 (AdamW, balanced, long) — MNIST — 2026-10-28

**Change:** Combining all the best models. AdamW with the lower lr (V4), class balance (V6), and 30 epochs (V1).

**Why:** I think these will all be helpful in different ways and together I expect this to be the best I have for MNIST.

**Expected:** I think this will beat both my first shifted model acc of .9777 and the baseline at 30 epochs of .9802

**Result:** Acc: 0.9833


### IMPORTANT NOTE

I just realized that all the shifted models up until this point have been using my goofy weight init scheme like model_2 used.
This is unfortunate because it pretty much showed to lower accuracy in the scenarios tested.


### model_shift_reg_init (normal init) — MNIST — 2026-10-28

**Change:** Like model_shift V7 **WITHOUT** the goofy weight init scheme I apparently have been using this whole time *facepalm.

**Why:** Because it didn't help either of the times I tried it

**Expected:** I would be very surprised if the accuracy doesn't surpass that of model_shift V7: 

**Result:** Acc: 0.9828. That's actually wild. My weight init scheme squeaked out a victory against pure Glorot. Shout out


### no_aug — cifar — 2026-10-28

**Change:** this model contains all the features that I found were best for model_shift_reg_init without the shifting augmentation. Basically baseline + AdamW, 0.003 LR, and balanced loss. To save time it's only for 10 epochs.

**Why:** Because I think the wrap-around shifts are too risky for cifar10. All the images are quite close to the edge. All the other features seemed to help

**Expected:** I would expect this to beat the 10 epoch baseline accuracy of 0.4535

**Result:** Acc: 0.4965. Nice


### no_aug V2 (SwiGLU) — cifar — 2026-10-28

**Change:** SwiGLU

**Why:** Because I want to see what happens

**Expected:** I would expect maybe slight improvement but I wouldn't be surprised with small change in either direction.

**Result:** Acc: 0.5447. That was so much more improvement than I expected to see. Very cool.


### model_shift V8 (SwiGLU) — MNIST — 2026-10-28

**Change:** I had to go back to MNIST to try SwiGLU after I saw how much it improved cifar10

**Why:** It worked for cifar10

**Expected:** Maybe a slight increase over my best model acc (model_shift V7) of 0.9833

**Result:** Acc: 


### no_aug V3 (wider blocks) — cifar — 2026-10-28

**Change:** Like V2 with SwiGLU but increased hidden state.

**Why:** I want to see if increased capacity in the width dimension helps.

**Expected:** I would expect maybe slight improvement but again, I wouldn't be surprised with small change in either direction.

**Result:** Acc: 0.5471. Very negligible increase. 


### flipper — cifar — 2026-10-28

**Change:** Made to be exactly like no_aug V2 (the miniscule gains of V3 I figured weren't worth the computation) but there are 3 copies of each original image: original, flipped horizontally, flipped vertically

**Why:** 

**Expected:** I would expect maybe slight improvement but again, I wouldn't be surprised with small change in either direction.

**Result:** Acc: 0.5471. Very negligible increase. 

