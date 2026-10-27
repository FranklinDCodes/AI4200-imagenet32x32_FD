### BASELINE — MNIST — 2026-10-27

**Result:** Acc: 0.9506


### MODEL_2 — MNIST — 2026-10-27

**Change:** Used a weight initialization scheme I came up with that satisfies glorot but shifts the variance more to the matrix rows than columns

**Why:** Because I hope that this helps keep the neuron tasks from overlapping too much, and maybe they can learn unique tasks quicker

**Expected:** Pretty much a 30% chance this improves anything, more likely makes it worse I think.

**Result:** 
