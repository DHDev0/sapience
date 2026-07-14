# Capability vs. fidelity — measured cost of each biological constraint

16k-neuron spiking cortex, 220 e-prop steps, identical seed/data. bits/byte (lower = more capable).
Cost = bits/byte above the plausible learned-feedback baseline.

| Configuration | bits/byte | cost vs plausible base | ms/step | spike rate |
|---|---|---|---|---|
| BPTT (non-plausible ceiling) | 2.424 | -3.775 | 12 | 0.0249 |
| e-prop + random feedback (DFA) | 6.206 | +0.007 | 12 | 0.0394 |
| + learned feedback (Kolen-Pollack) | 6.199 | +0.000 | 12 | 0.0396 |
| + Dale's law (E/I typing) | 5.584 | -0.615 | 12 | 0.2198 |
| + dendritic / burst error | 6.578 | +0.379 | 12 | 0.0273 |
| + bounded synapses (Fusi) | 6.199 | +0.000 | 12 | 0.0396 |
| + firing-rate homeostasis | 6.207 | +0.008 | 12 | 0.0397 |
| + BTSP long eligibility | 6.055 | -0.144 | 12 | 0.0451 |
| + unified two-compartment | 5.899 | -0.300 | 13 | 0.0525 |
| + differentiated neuromod | 5.963 | -0.236 | 12 | 0.0356 |
| + stochastic spiking | 6.186 | -0.013 | 12 | 0.0398 |
| + metabolic cost | 6.338 | +0.139 | 12 | 0.0332 |
| FULL faithful stack (all) | 5.098 | -1.101 | 17 | 0.2322 |
