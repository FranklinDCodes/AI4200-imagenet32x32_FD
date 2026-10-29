
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


### BASELINE V2 — imagenet32 — 2026-10-28

**Change:** Baseline but with 5 epochs

**Result:** Acc: 0.0512


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


### model_shift_reg_init — MNIST — 2026-10-28

**Change:** Like model_shift V7 **WITHOUT** the goofy weight init scheme I apparently have been using this whole time *facepalm. Instead straight Glorot

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

**Change:** I had to go back to MNIST to try SwiGLU after I saw how much it improved cifar10. Exactly like model_shift V7 but SwiGLU blocks

**Why:** It worked for cifar10

**Expected:** Maybe a slight increase over my best model acc (model_shift V7) of 0.9833

**Result:** Acc: 0.9840. Nice!


### no_aug V3 (wider blocks) — cifar — 2026-10-28

**Change:** Like V2 with SwiGLU but increased hidden state.

**Why:** I want to see if increased capacity in the width dimension helps.

**Expected:** I would expect maybe slight improvement but again, I wouldn't be surprised with small change in either direction.

**Result:** Acc: 0.5471. Very negligible increase. 


### flipper — cifar — 2026-10-28

**Change:** Made to be exactly like no_aug V2 (the miniscule gains of V3 I figured weren't worth the computation) but there are 3 copies of each original image: original, flipped horizontally, flipped vertically.

**Why:** Flips I hope will help the network learn more the shapes of things, and not the location or angle.

**Expected:** I would expect either a few percent increase or a few percent decrease from no_aug V2. I could see this not helping if all images for a certain object are usually oriented a certain way in the test set. I would be most surprised if it had little to no effect.

**Result:** Acc: 0.5526. That is a noticeable improvement.


### flipper V2 (lower lr) — cifar — 2026-10-28

**Change:** I want to try a lower learning rate of 0.001.

**Why:** Given the increase in data I want it to learn less from individual instances to hopefully converge more smoothly.

**Expected:** I would expect either a couple percent increase on .5526, or if the lower lr is too slow I would expect a small decrease that could I imaging go away with more train time (like if it learns too slow).

**Result:** Acc: 0.5450. Much closer than I would've imagined. If I start training for 30 epochs, may need to try this again.


### flipper V3 (flipper fader) — cifar — 2026-10-28

**Change:** Just like og flipper but I'm copying items in the dataset 1 more time, this time with a 75% fade in all channels.

**Why:** I want to see if I can give the model image-softness invariance.

**Expected:** Genuinely no clue. Could help, could confuse the model, or at this point, since there is a lot of data, it could just have a very little effect.

**Result:** Acc: 0.5416. Color fading seemed to hurt.


### flipper V4 (flipper colorizer) — cifar — 2026-10-28

**Change:** Just like og flipper but I'm copying items in the dataset 2 more times, each time with a different random permutation of the color channels.

**Why:** I want to see if I can give the model some color invariance.

**Expected:** I would think it'll help a little.

**Result:** Acc: 0.500. Very interesting. Color augmentations seemed to be notably hurtful.


### flipper V5 (long) — cifar — 2026-10-28

**Change:** Like og flipper but now back to original 30 epoch training length

**Why:** The baseline benefited from the additional training.

**Expected:** I would expect 0.58-0.59 test accuracy.

**Result:** Acc: 0.5377. Very interesting. May be overfitting.


### flipper V6 (long, lower lr) — cifar — 2026-10-28

**Change:** Like V5 (long) flipper, but I wanted to try that lower learning rate again with more train time.

**Why:** Because the lower LR performed very similar to the higher one 10 epochs and I wonder if it overtakes it if you give them both more time.

**Expected:** Realistically maybe 1-2% less that V5 but I think it's possible that it's higher, that's why I want to try.

**Result:** Acc: 0.5338. Virtually the same as the long one with the higher lr. Long training is a bad idea clearly.


### flipper V7 (short) — cifar — 2026-10-28

**Change:** Like original flipper but 7 epochs of training.

**Why:** Because long flipper (V5) showed a decrease in test accuracy so maybe we are overfit.

**Expected:** Unsure, maybe few percent increase or decrease.

**Result:** Acc: 0.5494. Interesting. Very close to og flipper.


### flipper V8 (medium) — cifar — 2026-10-28

**Change:** Like original flipper but 13 epochs of training. I really should've built some validation but it's too late now.

**Why:** Because long flipper (V5) showed a decrease in test accuracy and short flipper (V7) showed a smaller one.

**Expected:** Unsure, maybe few percent increase or decrease.

**Result:** Acc: 0.5451. AND STILL, og flipper is the best.


### flipper V9 (wider) — cifar — 2026-10-28

**Change:** Like og flipper but I went ahead and made the blocks twice as wide again. That helped a small amount in no_aug V3.

**Why:** Figured if it added a little accuracy without augmentation it may help with flipping.

**Expected:** A miniscule increase or decrease over original 0.5526.

**Result:** Acc: 0.5466. Dang bro, really nothing beats the original


### flipper V10 (narrower) — cifar — 2026-10-28

**Change:** Like og flipper but I went ahead and made the blocks 192 in the hidden state instead of 256.

**Why:** The wider V9 did worse than the original.

**Expected:** A miniscule increase or decrease over original 0.5526.

**Result:** Acc: 0.5425. I guess I really struck gold with og Flipper.


### no_aug V4 — imagenet32 — 2026-10-28

**Change:** Like no_aug V2 from cifar, but shorter train time (5 epochs). Should one helpful baseline for imagenet32

**Why:** Because I want an unaugmented baseline.

**Expected:** I would expect maybe a couple percent improvement on the residual_mlp_baseline 5 epoch test acc: .0512.

**Result:** Acc: 0.0927. Good!


### no_aug V5 (deep) — imagenet32 — 2026-10-28

**Change:** Like V4, but 10 layers deeper.

**Why:** I want to see if more model capacity will help right now with this dataset.

**Expected:** I would expect maybe a couple percent improvement on V4's 0.927

**Result:** Acc: 0.0939. Miniscule improvement.


### flipper — imagenet32 — 2026-10-28

**Change:** Like original flipper used in cifar. Figured it would be a good starting point for imagenet32.

**Why:** Because it worked alright on cifar.

**Expected:** Realistically maybe a couple percent more than the no_aug?

**Result:** Acc: 0.092. It looks like using flip augmentation hurts performance.


### flipper V11 (only colorizer) — imagenet32 — 2026-10-28

**Change:** Like flipper V4 except it does not also flip. Augmenting with 2 random permutations of the color channels for each instance.

**Why:** I want to see if I can give the model some color invariance.

**Expected:** I would think it may help a little over the no aug 0.0927.

**Result:** Acc: 0.0813. Definitely hurts performance


### no_aug V6 (big batch) — imagenet32 — 2026-10-28

**Change:** 512 batches

**Why:** This dataset has so many classes I want to try and make descent less stochastic by hopefully having more uniform, larger batches

**Expected:** It could improve performance a percent or two or it could hurt because it slows training down, and more training would be needed to know how helpful it is for sure.

**Result:** Acc: 0.0972. Bigger batches helped


### no_aug V7 (4 epochs) — imagenet32 — 2026-10-28

**Change:** 512 batches, 4 epochs

**Why:** Random thought just want to make sure we're not overfitting

**Expected:** Likely it'll be a little worse but I want to be sure

**Result:** Acc: 0.0928. Yeah I guess it was not, in fact, overfitting.


### no_aug V8 (supa big batch) — imagenet32 — 2026-10-28

**Change:** 1024 batches

**Why:** Increasing batch size helped before, it may again

**Expected:** This may increase a bit more accuracy, or it may now slow down training too much

**Result:** Acc: 0.0986. The bigger the better i guess


### no_aug V9 (zAnY big batches) — imagenet32 — 2026-10-28

**Change:** 2048 batches

**Why:** Increasing batch size helped before, it may again

**Expected:** This may increase a bit more accuracy a tiny bit

**Result:** Acc: 0.0993


### no_aug V10 (zAnierrr big batches) — imagenet32 — 2026-10-28

**Change:** 4096 batches

**Why:** Increasing batch size helped before, it may again

**Expected:** This may increase a bit more accuracy a tiny bit more, and it definitely speeds up training

**Result:** Acc: 0.0948. hit the limit of batch sizes i guess


### no_aug V11 (long training) — imagenet32 — 2026-10-28

**Change:** 1024 batches, 20 epochs of training

**Why:** Seems like I can get more out this model, with the batches this big I am training less than ever

**Expected:** I'm guessing this will get me in the 0.1-0.12 range

**Result:** Acc: 0.0824


### no_aug V11 (med training) — imagenet32 — 2026-10-28

**Change:** 1024 batches, 10 epochs of training

**Why:** 20 batches seemed to overfit, but I'm not convinced 5 is enough

**Expected:** I'm guessing this will get me in the 0.1-0.12 range for real

**Result:** Acc: 0.0973


### no_aug_2 V12 (dropout) — imagenet32 — 2026-10-28

**Change:** 2048 batches, 10 epochs, dropout

**Why:** It may help generalization with longer training

**Expected:** This may increase a bit more accuracy a tiny bit

**Result:** Acc: 0.1102. Hooray!
