# Transformer-based machine translation on IWSLT14

We compared four optimizer implementations—AdamW, AdaBelief, EMOJI, and Lion—on the IWSLT14 German-to-English machine translation task using Fairseq. All runs used the `transformer_iwslt_de_en` architecture and the same dataset and shared training configuration. The experiment protocol used the same three random seeds (1993, 1402, and 2025) across optimizer configurations.

## Model architecture and training configuration

The models used Fairseq's `transformer_iwslt_de_en` architecture. The saved configuration inspected for the EMOJI run confirms six encoder layers and six decoder layers, 512-dimensional embeddings, feed-forward networks of dimension 1,024, and four attention heads in both the encoder and decoder. The model used ReLU activation, dropout of 0.3, zero attention and activation dropout, post-layer normalization, non-learned positional embeddings, and shared decoder input/output embeddings.

**Table 1. Model architecture and configuration.**

| Component | Configuration |
|---|---|
| Task | German-to-English machine translation |
| Dataset | IWSLT14 German-to-English |
| Fairseq task | `translation` |
| Fairseq architecture | `transformer_iwslt_de_en` |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Encoder embedding dimension | 512 |
| Decoder embedding dimension | 512 |
| Encoder attention heads | 4 |
| Decoder attention heads | 4 |
| Attention head dimension | 128 |
| Encoder feed-forward dimension | 1,024 |
| Decoder feed-forward dimension | 1,024 |
| Activation function | ReLU |
| Decoder input/output embeddings | Shared |
| Encoder layer normalization | Post-LayerNorm (Pre-LN disabled) |
| Decoder layer normalization | Post-LayerNorm (Pre-LN disabled) |
| Embedding layer normalization | Disabled |
| Learned positional embeddings | Disabled |
| Maximum source positions | 1,024 tokens |
| Maximum target positions | 1,024 tokens |
| Model dropout | 0.3 |
| Attention dropout | 0.0 |
| Activation dropout | 0.0 |

The shared training and evaluation configuration is summarized in Table 2. The listed architecture details are present in the inspected EMOJI checkpoint configuration; the five supplied commands use the same architecture name and common command-line settings.

**Table 2. Shared training configuration.**

| Training parameter | Value |
|---|---|
| Maximum epochs | 50 |
| Maximum tokens per batch | 4,096 |
| Gradient accumulation (`update_freq`) | 1 (Fairseq configuration) |
| Learning-rate scheduler | Inverse square root (`inverse_sqrt`) |
| Warm-up updates | 4,000 |
| Gradient clipping (`clip_norm`) | 0.0 (disabled) |
| Training objective | Label-smoothed cross-entropy |
| Label-smoothing factor | 0.1 |
| Logging format | JSON |
| Logging interval | 100 updates (stored checkpoint configuration) |
| Data-loader workers | 1 (stored checkpoint configuration) |
| Validation frequency | Every epoch (stored checkpoint configuration) |
| Maximum validation tokens per batch | 4,096 (stored checkpoint configuration) |
| Random seeds in the experiment protocol | 1993, 1402, 2025 |


All command examples use the inverse-square-root learning-rate scheduler with 4,000 warm-up updates, disabled gradient clipping, dropout of 0.3, label-smoothed cross-entropy with a smoothing factor of 0.1, and a maximum of 4,096 tokens per batch. Optimizer-specific learning rates, beta coefficients, and weight decay are listed separately in Table 3.

## Optimizer configurations

AdamW, AdaBelief, and EMOJI were evaluated using the same learning rate of `5e-4`, weight decay of `0.0001`, and moment coefficients β₁ = 0.9 and β₂ = 0.98. The choice of these coefficients follows the original Transformer paper, *Attention Is All You Need*, which used Adam with β₁ = 0.9, β₂ = 0.98. In our experiments, we retained the same beta coefficients and used ε = 1e-8, together with a common inverse-square-root learning-rate schedule with 4,000 warm-up updates. Applying the same settings to AdamW, AdaBelief, and EMOJI provides a controlled comparison of their optimization behavior under a shared configuration.

**Table 3. Optimizer-specific hyperparameters.**


| Optimizer | Learning rate | β₁ | β₂ | Epsilon | Weight decay |
|---|---|---|---|---|---|
| AdamW | `5e-4` | 0.9 | 0.98 | `1e-8` | 0.0001 |
| AdaBelief | `5e-4` | 0.9 | 0.98 | `1e-8` | 0.0001 |
| EMOJI | `5e-4` | 0.9 | 0.98 | `1e-8` | 0.0001 |
| Lion | `5e-5` | 0.95 | 0.98 | Not applicable | 0.001 |

Lion was configured according to recommendations in the Lion literature. Its learning rate was set to one-tenth of the AdamW learning rate, while its weight decay was increased tenfold. This adjustment is motivated by Lion's sign-based update, which can produce updates with larger norms than AdamW's and therefore commonly calls for a smaller learning rate and a larger weight-decay coefficient. The coefficients β₁ = 0.95 and β₂ = 0.98 are also consistent with settings investigated for Lion in language-modeling experiments. These values are optimizer-specific recommendations rather than the default beta values of Lion.

The Lion configuration therefore follows the recommended relative scaling of learning rate and weight decay, rather than enforcing identical hyperparameters across all optimizers. Consequently, the comparison evaluates the four methods under the specified configurations, not under a strict requirement that every optimizer use identical optimizer-specific hyperparameters.

## Reproducibility

Each optimizer was evaluated using the same three random seeds, with all other shared model and training settings held constant. This shared-seed protocol helps control for variability due to random initialization and training stochasticity when comparing results across optimizers.

**Harmonized continuation (aligned with the section above):**

## Statistical Convention

Throughout all analyses, the representative value for each seed is the **mean of the final 10 validation epochs**.  

When three seeds are available, the reported mean and sample standard deviation (ddof = 1) are computed **across these three seed-level final-10 means**. Raw per-epoch values are never averaged across seeds. This rule is applied uniformly to every table, figure, and CSV produced by the plotting scripts.

## EMOJI Naming Convention

Unless explicitly labeled **EMOJI (α = 0, Without Centering)**, every reference to **EMOJI** denotes the proposed configuration with centering enabled (α = 1).  

The parameter α is introduced solely for the ablation study and is **not** a tunable hyperparameter of the proposed optimizer. Consequently:

- `iwslt14_emoji` = EMOJI (α = 1, Proposed)  
- `iwslt14_emojiabl` = EMOJI (α = 0, Without Centering)

## Plotting & Analysis Scripts

All scripts reside in the `checkpoints/` directory and write outputs to `checkpoints_of_seeds/plots/`. Input logs are Fairseq JSON-line files located at `checkpoints_of_seeds/<seed>/<run_folder>/train.log`.

### 1. `plot_iwslt14_epsilon_sensitivity_combined.py`

Produces ε-sensitivity analysis of EMOJI (α = 1) versus AdaBelief for `valid_bleu`, `valid_ppl`, and `valid_loss`.

- Single-seed curves (seed 1993) over six ε values: 1e-24, 1e-16, 1e-12, 1e-10, 1e-8, 1e-6  
- Three-seed mean ± sample STD over ε ∈ {1e-16, 1e-12, 1e-8}  
- Combined view (mean markers + single-seed dotted curves)

Outputs (per metric): three SVG figures and two CSV files following the final-10 statistical convention.

### 2. `plot_iwslt14_train_metrics_mean_var_ablation.py`

Epoch-wise mean ± 1 sample STD across the three seeds for the ablation comparison:

- EMOJI (α = 1, Proposed)  
- AdaBelief  
- EMOJI (α = 0, Without Centering)

Metrics: `train_loss`, `valid_loss`, `valid_bleu`, `valid_ppl`.

### 3. `plot_iwslt14_train_metrics_mean_var_all.py`

Epoch-wise mean ± 1 sample STD across five configurations:

- EMOJI (α = 1, Proposed)  
- AdaBelief  
- AdamW  
- Lion  
- EMOJI (α = 0, Without Centering)

Files are timestamped and saved as both SVG and PNG on figure close.

### 4. `plot_iwslt14_train_metrics_mean_var_per_seed.py`

Per-seed epoch curves (no cross-seed averaging) for the same five configurations. Produces 12 figures (3 seeds × 4 metrics), each saved as SVG + PNG.

### 5. `plot_iwslt14_train_metrics_mean_var.py`

Bulk CSV export covering nine runs (base optimizers + selected ε variants). Main outputs:

- `iwslt14_all_metrics_mean_std_last10.csv` — cross-seed mean ± sample STD of final-10 means  
- `iwslt14_all_metrics_last10_per_seed.csv` — per-seed final-10 means  
- Epoch-wise mean/STD and raw per-seed comparative CSVs

### 6. `plot_iwslt14_train_metrics_sensitivity2.py`

Presentation-oriented ε-sensitivity plots:

- Four three-seed mean ± STD epoch plots  
- Four single-seed (seed 1993) epoch plots  
- Standalone legend  
- Two CSV files with final-10 statistics (single-seed and three-seed)

All figures use 300 dpi, tight bounding boxes, and follow the same final-10 / cross-seed statistical protocol.

## Experiment Launchers

### Base launcher — `run_IWSLT_GE_2_EN.ps1`

Launches the core 15 runs (3 seeds × 5 configurations):

- `iwslt14_adamw`  
- `iwslt14_emoji` (α = 1)  
- `iwslt14_adabelief`  
- `iwslt14_emojiabl` (α = 0)  
- `iwslt14_lion`

All runs share the architecture and training settings listed in Tables 1–2 and the optimizer hyperparameters of Table 3. Lion uses the distinct learning rate, weight decay, and β values shown in Table 3.

### ε-sensitivity launchers

- `run_IWSLT_Sensivity.ps1` — three-seed sweep for ε ∈ {1e-12, 1e-16, 1e-8 (default)}  
- `run_IWSLT_Sensivity_1seed.ps1` — single-seed (1993) sweep for ε ∈ {1e-24, 1e-10, 1e-6}

These complete the six-ε single-seed and three-ε multi-seed trajectories consumed by scripts 1 and 6.

Default ε = 1e-8 is obtained by omitting the `--emoji-eps` / `--adabelief-eps` flag. All EMOJI ε-variants use the proposed configuration (α = 1).
