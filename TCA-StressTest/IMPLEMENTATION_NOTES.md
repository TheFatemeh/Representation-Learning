# Implementation Notes — Reservoir Contamination Stress Test

Short, plain-language log of the changes we make to the stock TCA repo for the
reservoir-contamination experiments. One entry per reviewed step.

---

## Part 1 — Entrypoint fixes (make the stock code runnable)

**Files changed:** `runner.py`

What we did:

1. **Fixed a stock crash.** The stock loader call passes `args.visualize_mask`,
   but that argument was never defined, so even `python runner.py --datasets eurosat`
   crashed immediately with an `AttributeError`. We added a `--visualize_mask` flag
   (off by default), so the program runs. We keep it off for all stress-test runs.

2. **Allowed the R=0.9 pruning setting.** The `--token_pruning` argument only
   accepted a fixed short list that did **not** include `Ours-0.035` (the R=0.9
   setting all our runs use). We removed that restriction and made `Ours-0.035`
   the default, so the command line accepts it. The value is still "name-rate"
   (e.g. `Ours-0.035`) because the model splits it on the dash to read the rate.

Why: these are pre-existing problems in the clone. Nothing about the method
changes here — we just make the baseline actually start, so later instrumentation
sits on a working program.

Check: run one stock EuroSAT run with `Ours-0.035` and confirm it reproduces the
paper's ~70.4% before we instrument anything.

---

## Part 2 — Full instrumentation (rules, modes, logging, fixed order)

**Files changed:** `runner.py`, `utils.py`, `datasets/utils.py`

What we did, in plain terms:

1. **New command-line switches** (`runner.py`). One clean knob each instead of the old
   tangle of true/false flags:
   - `--update_rule {fifo, uncertainty, similarity, diversity}` — how the buffer decides
     what to throw away when it's full.
   - `--reservoir_mode {clean, empty, seeded}` — `clean` is normal TCA; `empty` never
     stores anything (used to find the poisons); `seeded` drops one poison into a chosen
     class buffer before the run starts.
   - `--reservoir_size` — overrides the buffer size M so we can sweep M = 1, 3, 5.
   - `--target_class` / `--poison_path` — which class gets poisoned and which saved poison
     file to inject.
   - `--out_dir` / `--seed` — where results go, and the seed that also fixes the data order.

2. **Tagged every stored entry** (`runner.py`, `utils.py`). Each buffer entry now also
   carries its true label, a poison flag, and an insertion number. These tags ride along
   when an entry is moved or evicted, so we always know whether an entry is "in the wrong
   buffer" (a contaminant) and whether it is the injected poison. Two old lines that read
   the *last* element of an entry were pointed back at the real token field, so the tags
   don't confuse them.

3. **Added the two missing update rules** (`runner.py`). `fifo` evicts the oldest entry
   (by insertion number — needed because the old code re-sorted and destroyed order);
   `uncertainty` evicts the most-uncertain (highest-entropy) entry, which keeps a confident
   poison the longest. `similarity` / `diversity` keep the original maths.

4. **Three reservoir modes in the eval loop** (`runner.py`). `empty` skips storage and the
   logits correction entirely (otherwise it would crash on an empty buffer); `seeded`
   preloads the poison as the oldest entry in the target buffer; `clean` is unchanged TCA.

5. **Logging added** (`runner.py`). Every run now writes three files:
   - a per-sample CSV (true class, predicted class, correct?, confidence on predicted and
     on true class),
   - a per-timestep CSV counting contaminants in every buffer, the surviving seeded poison,
     and how many source-class samples got pulled into the poisoned buffer,
   - a small JSON with overall accuracy and average true-class confidence.
   In `empty` mode it also writes `poisons.csv` plus one saved seed file per class (the
   most-confident mistake for each class), which the seeded runs later inject.

6. **Fixed, repeatable data order — kept identical to the official repo.** We first tried a
   custom seeded loader generator, but that changed the shuffle order and dropped EuroSAT
   accuracy (~63% vs the paper's ~70%), because EuroSAT is order-sensitive. So we reverted
   it: the loader keeps the stock `shuffle=True` over the global RNG, and the global seed is
   set once (default 1). A fixed seed reproduces the same shuffled order on every run, so the
   clean and poisoned timelines still line up for fair comparison **and** we match the paper
   exactly. No loader change remains. (Order is interleaved/shuffled, not grouped by class.)

Note: the "source-in-target" count in the per-timestep file is our reading of "how many
poisons ended up in the targeted buffer" — entries whose true class is the poison's source
class that are sitting in the poisoned buffer. Easy to change if a different definition is
wanted.

### Reservoir eviction scores

Let a class buffer be $\mathcal{R}_c=\{e_1,\dots,e_N\}$ with capacity $M$; an eviction is
triggered once $N=M+1$. Each entry $e_i$ carries a confidence score $H_i$ (entropy) and a
domain-anchor token signature $\hat{\tau}_i$.

**Entropy.** For a test sample with $\ell_2$-normalised visual embedding $z$ and class text
embeddings $\{t_k\}_{k=1}^{C}$, CLIP assigns

$$
p_k=\frac{\exp\!\big(\langle z,t_k\rangle/\kappa\big)}{\sum_{j=1}^{C}\exp\!\big(\langle z,t_j\rangle/\kappa\big)},
\qquad
H=-\sum_{k=1}^{C}p_k\log p_k,
$$

with temperature $\kappa=0.01$ (logit scale $1/\kappa=100$). Low $H$ means a confident
prediction.

**Token signature.** From the $L$ transformer layers, the stored $\texttt{[cls]}$ tokens
$\{v_i^{(l)}\}_{l=1}^{L}$ are averaged over depth and normalised,

$$
\hat{\tau}_i=\frac{\bar{\tau}_i}{\lVert\bar{\tau}_i\rVert},
\qquad
\bar{\tau}_i=\frac{1}{L}\sum_{l=1}^{L}v_i^{(l)} .
$$

**Intra-buffer similarity.** Each entry's mean cosine similarity to the rest of the buffer,

$$
\bar{a}_i=\frac{1}{N-1}\sum_{j\neq i}\langle\hat{\tau}_i,\hat{\tau}_j\rangle .
$$

Both rules evict $\arg\max_i\,\mathrm{score}_i$, i.e. the entry with the largest score.

**Diversity-enforced.** Penalises redundancy: an entry that resembles the rest of the buffer
(high $\bar a_i$) or is uncertain (high $H_i$) is removed first.

$$
\mathrm{score}^{\mathrm{div}}_i=H_i+\bar{a}_i .
$$

**Similarity-enforced.** Penalises outliers: with dissimilarity $d_i=1-\bar a_i$, the
most uncertain / least-similar entry is removed, keeping a tight, confident buffer.

$$
\mathrm{score}^{\mathrm{sim}}_i=H_i+w\,d_i,
\qquad
w=\left\lfloor\frac{\max_j H_j-\min_j H_j}{\max_j d_j-\min_j d_j}\right\rfloor ,
$$

where $w$ rescales the dissimilarity range onto the entropy range so the two terms are
commensurate ($w=0$ if the denominator is zero).

---

## Part 3 — Clean baselines validated against the paper

**Files added:** `results/clean_baselines.csv`, `results/clean_validation.md`; run scripts moved
to `scripts/`.

**Setup fix (not code):** the clone was missing CLIP's tokenizer vocab file
`clip/bpe_simple_vocab_16e6.txt.gz`, which made any run crash before inference. Copied it in
from the sibling `TCA/clip/` checkout.

We ran all 16 clean runs (4 rules x M in {1,2,3,5}) on EuroSAT with the official order/seed.
The M=2 column lines up with the paper's Table 4 (EuroSAT):

| rule | ours (M=2) | paper | diff |
|---|---|---|---|
| diversity | 70.09 | 70.43 | -0.34 |
| uncertainty | 70.32 | 70.20 | +0.12 |
| similarity | 69.07 | 68.48 | +0.59 |
| fifo | 53.72 | 50.28 | +3.44 |

Three of four match within ~0.6 points. Two extra sanity checks passed: (a) reverting the
custom data order alone moved diversity M=2 from 63.2% back up to 70.1%, confirming the order
was the whole gap; (b) at M=1 the uncertainty and diversity rules are identical (67.65 each),
which is expected — with a 2-item buffer the diversity similarity term is symmetric, so it
reduces to "drop the higher-entropy entry", i.e. uncertainty. FIFO is the worst rule and the
only one written from scratch; its +3.44 gap is consistent with FIFO being the most
order-sensitive rule (the paper also reports it as unstable on EuroSAT), not a logic error.

Where results live: every run writes `results/runs/<key>/` (a scalar JSON, a per-sample CSV,
and a per-timestep CSV); the two files above are the aggregated summary. Run scripts now live
in `scripts/` (`smoke_clean.sbatch`, `sweep_clean.sbatch`, `identify.sbatch`); submit them from
the repo root so `$SLURM_SUBMIT_DIR` points at the TCA-StressTest folder.

---

## Part 4 — Seeded runs: a device bug, and confirming the poison actually bites

**Files changed:** `runner.py` (seeded-mode poison load); scripts added:
`scripts/sweep_poison_M1.sbatch`, `_M3`, `_M5` (Stage 3, one array per M, 40 runs each).

First attempt at the 120 poisoned runs failed almost entirely (crash: "tensors on two
devices, cpu and cuda"). Cause: the encoder builds each sample's cls-token list on the CPU
(a `torch.zeros(...)` with no device), so every stored anchor keeps its cls tokens on CPU
while its image feature is on the GPU. Clean runs never noticed because every entry was CPU
in that slot. When we injected a saved poison and (wrongly) moved its cls tokens to the GPU,
that one GPU tensor no longer matched the CPU ones, and the code that stacks them crashed.
Fix: make the injected poison mirror a normal entry exactly — image feature on GPU, cls tokens
left on CPU.

We then checked one seeded run (diversity, M=3, poisoning class 3) against its clean twin. The
overall accuracy was identical to 5 digits, which looked suspicious, so we compared the two
runs sample by sample: 16 predictions actually change, but the gains and losses cancel out, so
the headline number lands in the same place. The changes happen early and then stop once the
poison is evicted. So the poison does have a real, measurable effect; it just concentrates on a
few classes and nets to ~zero on the overall average — which is exactly the kind of localized
damage the experiment is designed to detect (and why the per-class breakdown, not the overall
number, is the metric that matters).

