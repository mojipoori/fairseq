# GPT-style Transformer Language Modeling on WikiText-2

We evaluated four optimization algorithms—AdamW, AdaBelief, EMOJI, and Lion—on autoregressive language modeling using WikiText-2. All experiments used the same GPT-style Transformer architecture, dataset splits, training configuration, and set of three shared random seeds to support a consistent comparison across optimizers.

## Model architecture and training configuration

The model was implemented using Fairseq's `transformer_lm_gpt` architecture. It comprised four Transformer decoder layers, an embedding dimension of 128, four attention heads per layer, and a feed-forward network dimension of 512. The model used GELU activation, dropout of 0.1, attention dropout of 0.1, shared input and output embeddings, and a maximum sequence length of 128 tokens.

**Table 1. Model architecture and configuration.**

| Component | Configuration |
|---|---|
| Task | Autoregressive language modeling |
| Dataset | WikiText-2 |
| Fairseq architecture | `transformer_lm_gpt` |
| Transformer decoder layers | 4 |
| Embedding dimension | 128 |
| Attention heads per layer | 4 |
| Attention head dimension | 32 |
| Feed-forward dimension | 512 |
| Activation function | GELU |
| Input/output embeddings | Shared |
| Maximum sequence length | 128 tokens |
| Model dropout | 0.1 |
| Attention dropout | 0.1 |

The shared training configuration is summarized in Table 2.

**Table 2. Shared training configuration.**

| Training parameter | Value |
|---|---|
| Maximum epochs | 50 |
| Maximum tokens per batch | 1,024 |
| Gradient accumulation (`update_freq`) | 4 |
| Learning-rate scheduler | Inverse square root |
| Warm-up updates | 1,000 |
| Warm-up initial learning rate | `1e-7` |
| Gradient clipping (`clip_norm`) | 0.0 (disabled) |
| Logging interval | 100 updates |
| Data-loader workers | 0 |
| Random seeds | Three shared seeds |

The inverse-square-root schedule increased the learning rate during warm-up and subsequently decayed it according to the inverse square root of the update count. All optimizers used the same training schedule and model configuration, while their optimizer-specific hyperparameters were set as described below.

## Optimizer configurations

AdamW, AdaBelief, and EMOJI were evaluated using the same learning rate of `5e-4`, weight decay of `0.01`, and moment coefficients β₁ = 0.9 and β₂ = 0.98. The choice of these coefficients follows the original Transformer paper, *Attention Is All You Need*, which used Adam with β₁ = 0.9, β₂ = 0.98. In our experiments, we retained the same beta coefficients and used ε = 1e-8 and a common inverse-square-root learning-rate schedule with 1,000 warm-up updates. Applying the same settings to AdamW, AdaBelief, and EMOJI provides a controlled comparison of their optimization behavior under a shared configuration.

**Table 3. Optimizer-specific hyperparameters.**

| Optimizer | Learning rate | β₁ | β₂ | Epsilon | Weight decay |
|---|---|---|---|---|---|
| AdamW | `5e-4` | 0.9 | 0.98 | `1e-8` | 0.01 |
| AdaBelief | `5e-4` | 0.9 | 0.98 | `1e-8` | 0.01 |
| EMOJI | `5e-4` | 0.9 | 0.98 | `1e-8` | 0.01 |
| Lion | `5e-5` | 0.95 | 0.98 | Not applicable | 0.1 |

Lion was configured according to recommendations in the Lion literature. Its learning rate was set to one-tenth of the AdamW learning rate, while its weight decay was increased tenfold. This adjustment is motivated by Lion's sign-based update, which can produce updates with larger norms than AdamW's and therefore commonly calls for a smaller learning rate and a larger weight-decay coefficient. The coefficients β₁ = 0.95 and β₂ = 0.98 are also consistent with settings investigated for Lion in language-modeling experiments. These values are optimizer-specific recommendations rather than the default beta values of Lion.

The Lion configuration therefore follows the recommended relative scaling of learning rate and weight decay, rather than enforcing identical hyperparameters across all optimizers. Consequently, the comparison evaluates the four methods under the specified configurations, not under a strict requirement that every optimizer use identical optimizer-specific hyperparameters.

## Reproducibility

Each optimizer was evaluated using the same three random seeds, with all other shared model and training settings held constant. This shared-seed protocol helps control for variability due to random initialization and training stochasticity when comparing results across optimizers.

# WikiText-2 GPT-style Transformer: Optimizer Comparison and Ablation Study

The project contains four Python analysis scripts for analyzing optimizer performance in autoregressive language modeling using a GPT-style Transformer and the WikiText-2 dataset with Fairseq.

The scripts read existing Fairseq `train.log` files, extract epoch-wise training and validation metrics, calculate statistics across random seeds, export numerical results to CSV files, and generate plots in SVG and PNG formats. The per-seed plotting script additionally generates separate plots for each random seed and exports the same three timestamped CSV files.

These scripts are intended for post-training analysis. **They do not start training or modify checkpoints.**

All file paths below are relative to the project root unless otherwise specified.

## Scripts

### Comprehensive optimizer comparison

**Script:** `plot_wiki2_gpt_abl_all.py`

**Relative path:** `checkpoints\plot_wiki2_gpt_abl_all.py`

This script compares five optimizer configurations:

- **AdamW**
- **[AdaBelief](https://proceedings.neurips.cc/paper_files/paper/2020/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf)**
- **Lion**
- **EMOJI (Proposed; α = 1)**
- **EMOJI (α = 0, Without Centering)**

The experiment compares the proposed EMOJI optimizer against three baseline optimizers—AdamW, AdaBelief, and Lion—and also includes the EMOJI variant without centering.

The script extracts the available training and validation metrics from the corresponding Fairseq logs and generates plots and CSV files for all five configurations across the specified random seeds.

### Sequential ablation study

**Script:** `plot_wiki2_gpt_abl.py`

**Relative path:** `checkpoints\plot_wiki2_gpt_abl.py`

This script focuses on three configurations:

- **[AdaBelief](https://proceedings.neurips.cc/paper_files/paper/2020/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf)**
- **EMOJI (α = 0, Without Centering)**, which removes ε from the recursive second-moment state, and does not center the prediction error.
- **EMOJI (α = 1, Proposed)**, which incorporates two modifications: removing ε from the recursive second-moment state and centering the prediction error.

For clarity, α is not a hyperparameter of EMOJI and is introduced only in the ablation study to represent the presence or absence of the centering operation. The second-moment state in the ablation formulation is expressed as

$$
j_t = \operatorname{EMA}\left((d_t-\alpha v_t)^2\right).
$$

The EMOJI (α = 1, Proposed) configuration uses the centered gradient prediction error, whereas the No Centering variant omits this centering operation. Setting α = 0 therefore removes the centering term, making the resulting update theoretically equivalent to AdaBelief without ε in the recursive second-moment state. The proposed EMOJI configuration corresponds to the centered formulation.

### Four-optimizer comparison

**Script:** `plot_wiki2_gpt.py`

**Relative path:** `checkpoints\plot_wiki2_gpt.py`

This script compares four optimizer configurations:

- **AdamW**
- **[AdaBelief](https://proceedings.neurips.cc/paper_files/paper/2020/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf)**
- **Lion**
- **EMOJI (Proposed; α = 1)**

Unlike the comprehensive comparison script, this script excludes the EMOJI no-centering variant from the active optimizer list. Its configuration remains commented out in the code.

The script processes the recorded training and validation metrics for the four active configurations, calculates statistics across the available seeds, and exports the same three types of CSV files and up to four SVG and four PNG figures.

### Per-seed plotting and CSV exports

**Script:** `plot_wikitext2_gpt_metrics_per_seed.py`

**Relative path:** `fairseq/plot_wikitext2_gpt_metrics_per_seed.py`

This script analyses the existing Fairseq `train.log` files for all five optimizer configurations and the three random seeds (`1993`, `1402`, and `2025`). It does not start training or modify checkpoints.

Unlike the aggregate plotting scripts, this script generates separate plots for each seed. Each plot compares the five optimizer configurations using the same seed, without averaging across seeds or displaying cross-seed standard-deviation bands:

- AdamW
- AdaBelief
- Lion
- EMOJI (α = 0, without centering)
- EMOJI (α = 1, proposed)

For each seed, the script plots training loss, validation loss, training perplexity, and validation perplexity against epoch. Thus, it can generate up to 12 plots in total: four metrics for each of the three seeds.

#### Run the per-seed plotting script

From the project root, run:

```powershell
python fairseq/plot_wikitext2_gpt_metrics_per_seed.py
```

The script first exports the CSV files and then opens the available plot previews. Since `SHOW_PREVIEW = True` and `WAIT_FOR_PREVIEW_CLOSE = True`, close each preview window to save its SVG and PNG files and proceed to the next plot. If a metric has no available data for a seed, that plot is skipped.

#### Per-seed plot files

All plot files are written to:

```text
checkpoints/wikitext2_gpt_plots/
```

For each available metric and seed, the script saves both an SVG and a PNG file at 300 DPI. Filenames include the metric, seed, and run timestamp, for example:

```text
valid_ppl_vs_epoch_seed_1993_<RUN_TIMESTAMP>.svg
valid_ppl_vs_epoch_seed_1993_<RUN_TIMESTAMP>.png
```

The four plot filename prefixes are:

- `train_loss_vs_epoch`
- `valid_loss_vs_epoch`
- `train_ppl_vs_epoch`
- `valid_ppl_vs_epoch`

The per-seed plots compare optimizer curves directly within each seed; they do not show a mean curve or standard-deviation band across seeds.

#### CSV files

The per-seed script writes three timestamped CSV files to `checkpoints/wikitext2_gpt_plots/`:

```text
wikitext2_gpt_epochwise_metrics_<RUN_TIMESTAMP>.csv
wikitext2_gpt_final10_summary_<RUN_TIMESTAMP>.csv
wikitext2_gpt_table_final10_<RUN_TIMESTAMP>.csv
```

- **Epoch-wise metrics:** contains the available epoch-level values for every optimizer, seed, and metric, along with the cross-seed mean, sample standard deviation, and number of seeds contributing to each epoch-level statistic.
- **Final-10 summary:** selects the last 10 available epoch values separately for each optimizer, seed, and metric. It reports each seed's mean and sample standard deviation over the selected epochs, followed by the mean and sample standard deviation of the available seed-level means. If fewer than 10 epochs are available for a run, the summary uses the available epochs.
- **Fixed final-10 table:** averages epochs 41–50 separately within each seed. Cross-seed mean and sample standard deviation are reported only when all ten epochs are available for all three seeds for that optimizer and metric; otherwise, the aggregate fields are left blank and a warning is printed.

The final-10 summary and the fixed final-10 table are therefore not necessarily identical when logs have missing epochs: the summary uses the last 10 available epochs, whereas the table specifically uses epochs 41–50.

### EMOJI naming convention

Unless explicitly labeled **EMOJI (α = 0, Without Centering)**, every reference to EMOJI in these experiments denotes the **proposed configuration with α = 1**, which uses the centered gradient prediction error.

The value α is not an EMOJI hyperparameter. It is introduced exclusively for the ablation study to represent whether the centering operation is enabled or disabled. Consequently, labels such as `EMOJI`, `Emoji (Proposed)`, and `EMOJI (α = 1, Proposed)` refer to the same proposed configuration.

## Experimental configuration

The experiments use Fairseq's `transformer_lm_gpt` architecture for autoregressive language modeling on WikiText-2.

The shared model and training configuration is summarized below.

| Setting | Value |
|---|---|
| Dataset | WikiText-2 |
| Model architecture | `transformer_lm_gpt` |
| Transformer layers | 4 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Feed-forward network dimension | 512 |
| Dropout | 0.1 |
| Attention dropout | 0.1 |
| Random seeds | 1993, 1402, 2025 |
| Learning-rate scheduler | `inverse_sqrt` |
| Warm-up updates | 1,000 |
| Tokens per sample | 128 |
| Maximum tokens per batch | 1,024 |
| Update frequency | 4 |

The optimizer-specific learning rates and weight decay values used in the comprehensive and four-optimizer comparisons are:

| Optimizer configuration | Learning rate | Weight decay |
|---|---:|---:|
| AdamW | `5e-4` | `0.01` |
| AdaBelief | `5e-4` | `0.01` |
| Lion | `5e-5` | `0.1` |
| EMOJI (Proposed; α = 1) | `5e-4` | `0.01` |
| EMOJI (α = 0, Without Centering) | `5e-4` | `0.01` |

The ablation study compares AdaBelief and the two EMOJI configurations using the settings specified in its experiment configuration.

## Metrics and aggregation across seeds

The scripts extract four metrics from the Fairseq JSON-formatted training logs:

- `train_loss` — training loss.
- `valid_loss` — validation loss.
- `train_ppl` — training perplexity.
- `valid_ppl` — validation perplexity.

### Epoch-wise statistics

For each optimizer and metric, the aggregate scripts calculate the mean and sample standard deviation across the available seeds at each epoch.

The aggregate plots display the epoch-wise mean as a curve, with a shaded region representing one sample standard deviation above and below the mean.

These plots allow the training and validation trajectories of the optimizer configurations to be compared. If a log or metric is unavailable, the scripts process the available data and report missing log files or unavailable metrics.

### Final-10-epoch summary

The final-epoch summary CSV calculates a representative value for each optimizer, seed, and metric by averaging the **final 10 available epoch values separately for each seed**.

It also records the sample standard deviation within those selected epochs. The resulting seed-level means are then aggregated across the available seeds, producing the mean and sample standard deviation across seeds.

This preserves individual-run results while providing aggregate statistics for each optimizer configuration.

### Compact final-10-epoch comparison

The compact table CSV specifically uses **epochs 41–50**.

For each optimizer and metric, it reports the mean over these ten epochs separately for seeds 1993, 1402, and 2025. It then calculates the mean and sample standard deviation across the three seed-level means.

Aggregate statistics are reported only when all ten required epochs are available for all three seeds. Otherwise, the corresponding aggregate fields remain empty.

### Per-seed plotting

In addition to the aggregate plots, `plot_wikitext2_gpt_metrics_per_seed.py` generates separate plots for each seed. Each per-seed plot compares the five optimizer configurations using the same seed and does not average across seeds or display cross-seed standard-deviation bands. For each seed, the script plots training loss, validation loss, training perplexity, and validation perplexity against epoch, producing up to 12 plots in total.

## Generated files

The aggregate scripts save their outputs to the following directory, relative to the project root identified by the script:

`checkpoints/wikitext2_gpt_plots/`

The per-seed plotting script writes its plots and CSV exports to the same directory.

Each analysis script generates three CSV files. The aggregate scripts also generate four SVG figures and four PNG figures when all four metrics have data available for plotting. The per-seed script generates up to 12 per-seed SVG and PNG figures, four metrics for each of the three seeds, when the corresponding data are available.

The output filenames include a run timestamp in the `YYYYMMDD_HHMMSS` format. In the filename patterns below, `{RUN_TIMESTAMP}` represents the actual timestamp generated when the script starts.

### CSV files

| Exact filename pattern | Description |
|---|---|
| `wikitext2_gpt_epochwise_metrics_{RUN_TIMESTAMP}.csv` | Contains epoch-wise metric values for each optimizer, seed, and metric. Each row also includes the cross-seed mean, sample standard deviation, and number of available seeds for the corresponding optimizer–metric–epoch combination. |
| `wikitext2_gpt_final10_summary_{RUN_TIMESTAMP}.csv` | Contains the mean and sample standard deviation over the final 10 available epochs for each individual optimizer–seed–metric combination. It also includes `MEAN_ACROSS_SEEDS` rows containing the mean and sample standard deviation across the available seed-level means. If fewer than 10 epochs are available for a run, the summary uses the available epochs. |
| `wikitext2_gpt_table_final10_{RUN_TIMESTAMP}.csv` | Provides a compact table with one row per optimizer. For each metric, it includes individual seed means over epochs 41–50, the mean and sample standard deviation across seeds, the epochs used, and the number of included seeds. Aggregate statistics require complete data for all three seeds and all ten specified epochs; otherwise, the aggregate fields are left blank and a warning is printed. |

The per-seed plotting script writes the same three timestamped CSV files to `checkpoints/wikitext2_gpt_plots/`:

```text
wikitext2_gpt_epochwise_metrics_<RUN_TIMESTAMP>.csv
wikitext2_gpt_final10_summary_<RUN_TIMESTAMP>.csv
wikitext2_gpt_table_final10_<RUN_TIMESTAMP>.csv
```

The three CSV files serve complementary purposes:

- **Epoch-wise metrics:** detailed numerical data for examining training and validation trajectories and comparing individual seeds.
- **Final-10-epoch summary:** per-seed statistics over the final 10 available epochs, together with aggregate statistics across seeds.
- **Compact final-10-epoch table:** a concise comparison based specifically on epochs 41–50.

The final-10 summary and the fixed final-10 table are therefore not necessarily identical when logs have missing epochs: the summary uses the last 10 available epochs, whereas the table specifically uses epochs 41–50.

The CSV exports can be opened in spreadsheet software or processed further with Python for additional analysis.

### Comparing individual seeds in the CSV files

The CSV files support both **optimizer comparisons within the same seed** and **seed comparisons for the same optimizer**.

| Comparison objective | CSV file | How to use it |
|---|---|---|
| Compare different optimizers for the same seed at each epoch | `wikitext2_gpt_epochwise_metrics_{RUN_TIMESTAMP}.csv` | Filter by a fixed `seed` and `metric`, then compare the `value` column across optimizers at matching epochs. |
| Compare different seeds for the same optimizer across epochs | `wikitext2_gpt_epochwise_metrics_{RUN_TIMESTAMP}.csv` | Filter by a fixed `optimizer` and `metric`, then compare the records for seeds 1993, 1402, and 2025. |
| Compare optimizers using one representative value per seed over the final 10 available epochs | `wikitext2_gpt_final10_summary_{RUN_TIMESTAMP}.csv` | Filter by a fixed `seed` and `metric`, then compare `mean_over_last_epochs` across optimizers. |
| Compare seed-level final-10 means for different optimizers in a compact table | `wikitext2_gpt_table_final10_{RUN_TIMESTAMP}.csv` | Compare columns such as `valid_ppl_seed_1993_last10_mean`, `valid_ppl_seed_1402_last10_mean`, and `valid_ppl_seed_2025_last10_mean` across optimizer rows. |

For a **seed-by-seed comparison between optimizers**, use the same seed for every optimizer. For example, compare AdamW, AdaBelief, Lion, and EMOJI for seed `1993`, then repeat for seeds `1402` and `2025`.

For a **comparison of variability across seeds**, keep the optimizer fixed and compare its results for seeds `1993`, `1402`, and `2025`.

In the final-10-epoch summary, rows labeled `MEAN_ACROSS_SEEDS` contain aggregate statistics rather than individual-seed results. In the compact table, each seed's final-10 mean is stored in its own column.

### SVG and PNG figures

The aggregate scripts generate plots for four metrics.

| Metric | SVG filename | PNG filename |
|---|---|---|
| Training loss | `train_loss_vs_epoch_{RUN_TIMESTAMP}.svg` | `train_loss_vs_epoch_{RUN_TIMESTAMP}.png` |
| Validation loss | `valid_loss_vs_epoch_{RUN_TIMESTAMP}.svg` | `valid_loss_vs_epoch_{RUN_TIMESTAMP}.png` |
| Training perplexity | `train_ppl_vs_epoch_{RUN_TIMESTAMP}.svg` | `train_ppl_vs_epoch_{RUN_TIMESTAMP}.png` |
| Validation perplexity | `valid_ppl_vs_epoch_{RUN_TIMESTAMP}.svg` | `valid_ppl_vs_epoch_{RUN_TIMESTAMP}.png` |

Each aggregate figure displays the epoch-wise mean metric values for the active optimizer configurations. The shaded region represents plus or minus one sample standard deviation across the available seeds.

- **SVG:** a scalable vector format suitable for research papers, publications, and documentation.
- **PNG:** a raster format convenient for previews, presentations, and sharing.

With the current configuration, each plot is displayed in a preview window and saved in both formats when the window is closed. The CSV files are exported before the plotting windows open. If no data are available for a metric, the corresponding figure is not generated.

#### Per-seed figures

The per-seed script saves both an SVG and a PNG file at 300 DPI for each available metric and seed. Filenames include the metric, seed, and run timestamp, for example:

```text
valid_ppl_vs_epoch_seed_1993_<RUN_TIMESTAMP>.svg
valid_ppl_vs_epoch_seed_1993_<RUN_TIMESTAMP>.png
```

The four plot filename prefixes are:

- `train_loss_vs_epoch`
- `valid_loss_vs_epoch`
- `train_ppl_vs_epoch`
- `valid_ppl_vs_epoch`

The per-seed plots compare optimizer curves directly within each seed; they do not show a mean curve or standard-deviation band across seeds.

## TensorBoard logging

TensorBoard logging was enabled in the Fairseq training commands for the WikiText-2 GPT-style Transformer experiments using the `--tensorboard-logdir` argument.

Each training run is associated with a particular optimizer configuration and random seed. The TensorBoard event files are written to the directory specified by that run's `--tensorboard-logdir` argument. When the run directories are organized under the common `checkpoints` directory, TensorBoard can display the logged runs together for interactive comparison.

To launch TensorBoard from the appropriate project environment, run:

```bash
tensorboard --logdir checkpoints
```

Then open the local URL printed in the terminal, typically:

`http://localhost:6006`

The directory supplied to `--logdir` must contain the TensorBoard event files. If the event files are stored in a different location, replace `checkpoints` with the appropriate directory.

### Comparing individual seeds in TensorBoard

TensorBoard allows users to compare **different optimizers using the same seed** and **different seeds using the same optimizer**, provided the corresponding runs have logged event files under the selected log directory.

- **Compare optimizers for the same seed:** select the runs for the desired seed across the optimizer configurations. For example, compare AdamW/seed 1993, AdaBelief/seed 1993, Lion/seed 1993, and EMOJI/seed 1993. This helps compare their recorded training and validation trajectories under the same seed.
- **Compare seeds for the same optimizer:** select the runs for seeds 1993, 1402, and 2025 for one optimizer. This helps inspect how the recorded trajectory varies across random seeds.
- **Repeat for other seeds and optimizers:** use the same procedure for the remaining seed values and optimizer configurations. In the ablation study, compare the three configurations included in that experiment.

For a fair seed-by-seed comparison, select runs with the **same seed across all optimizers**. To inspect seed-to-seed variability, select the **same optimizer across the three seeds**.

The available metrics and run names depend on the event files generated during training and the directory structure used by `--tensorboard-logdir`.

TensorBoard complements the CSV exports and figures by providing an interactive view of the recorded training metrics, while the CSV files provide numerical values for independent analysis.

## Running the analysis scripts

Run the desired script from the configured project environment. The following paths are relative to the project root.

For the comprehensive five-configuration optimizer comparison:

```bash
python checkpoints/plot_wiki2_gpt_abl_all.py
```

For the sequential ablation study:

```bash
python checkpoints/plot_wiki2_gpt_abl.py
```

For the four-optimizer comparison:

```bash
python checkpoints/plot_wiki2_gpt.py
```

For the per-seed plotting and CSV export script:

```powershell
python fairseq/plot_wikitext2_gpt_metrics_per_seed.py
```

The scripts expect their corresponding training logs to exist under the configured checkpoint directory structure. Missing log files are reported, and the scripts continue processing the available runs.

The generated CSV files are written before the plotting windows open. Close each plot window to save its SVG and PNG figures. For the per-seed script, close each preview window to save its SVG and PNG files and proceed to the next plot. If a metric has no available data for a seed, that plot is skipped.

## Summary

The four scripts provide complementary analyses of EMOJI on WikiText-2 using a GPT-style Transformer.

| Script | Relative path | Purpose | Active optimizer configurations |
|---|---|---|---|
| `plot_wiki2_gpt_abl_all.py` | `checkpoints\plot_wiki2_gpt_abl_all.py` | Comprehensive optimizer comparison | AdamW, AdaBelief, Lion, EMOJI (Proposed; α = 1), and EMOJI (α = 0, Without Centering) |
| `plot_wiki2_gpt_abl.py` | `checkpoints\plot_wiki2_gpt_abl.py` | Sequential ablation study | AdaBelief, EMOJI (α = 0, Without Centering), and EMOJI (α = 1, Proposed) |
| `plot_wiki2_gpt.py` | `checkpoints\plot_wiki2_gpt.py` | Four-optimizer comparison | AdamW, AdaBelief, Lion, and EMOJI (Proposed; α = 1) |
| `plot_wikitext2_gpt_metrics_per_seed.py` | `fairseq/plot_wikitext2_gpt_metrics_per_seed.py` | Per-seed plotting and CSV exports | AdamW, AdaBelief, Lion, EMOJI (α = 0, Without Centering), and EMOJI (α = 1, Proposed) |

Each analysis script exports three CSV files containing epoch-wise metrics, final-10-epoch summaries, and a compact table based on epochs 41–50. When the required metric data are available, the aggregate scripts also generate SVG and PNG figures for the four training and validation metrics. The per-seed script generates up to 12 per-seed SVG and PNG figures, four metrics for each of the three seeds.

Together with the TensorBoard logs recorded during Fairseq training, these outputs support detailed examination of optimizer behavior, seed-level comparisons, and reproducible reporting of the WikiText-2 experiments.

## Running the WikiText-2 GPT-style Transformer experiments

**Training script:** `run_wiki_tr_gpt_loop.ps1`

**Relative path:** `fairseq\run_wiki_tr_gpt_loop.ps1`

The PowerShell script `run_wiki_tr_gpt_loop.ps1` automates the WikiText-2 training experiments using Fairseq's GPT-style Transformer architecture. It launches the training runs sequentially, records the training logs, saves checkpoints, and configures TensorBoard logging separately for each optimizer and random seed.

Unlike the plotting scripts described above, this script **runs the training experiments** rather than analyzing existing logs.

### Experiment configurations and execution order

The training script includes **15 runs**:

- **12 standard optimizer runs:** AdamW, EMOJI (Proposed; α = 1), AdaBelief, and Lion, evaluated using three random seeds.
- **3 additional ablation runs:** EMOJI (α = 0, Without Centering), evaluated using the same three seeds.

The standard runs follow this order:

| Random seed | Execution order |
|---|---|
| `1993` | AdamW → EMOJI → AdaBelief → Lion |
| `1402` | AdamW → EMOJI → AdaBelief → Lion |
| `2025` | AdamW → EMOJI → AdaBelief → Lion |

After the 12 standard runs, the script executes the three additional EMOJI no-centering runs for seeds `1993`, `1402`, and `2025`.

Each run is configured for a maximum of 50 epochs.

### Model and training configuration

All runs use the following GPT-style Transformer configuration:

| Setting | Value |
|---|---|
| Dataset | WikiText-2 |
| Architecture | `transformer_lm_gpt` |
| Transformer layers | 4 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Feed-forward network dimension | 512 |
| Activation function | `gelu` |
| Dropout | 0.1 |
| Attention dropout | 0.1 |
| Learning-rate scheduler | `inverse_sqrt` |
| Warm-up updates | 1,000 |
| Warm-up initial learning rate | `1e-7` |
| Tokens per sample | 128 |
| Maximum tokens per batch | 1,024 |
| Update frequency | 4 |
| Maximum epochs | 50 |
| Random seeds | 1993, 1402, 2025 |
| Fixed validation seed | Same as the run's training seed |
| Gradient clipping norm | 0.0 |
| Number of data-loader workers | 0 |

### Optimizer-specific settings

The standard optimizer configurations use the following hyperparameters:

| Optimizer | Learning rate | Weight decay | Betas | Epsilon |
|---|---:|---:|---|---:|
| AdamW | `5e-4` | `0.01` | `(0.9, 0.98)` | `1e-8` |
| EMOJI (Proposed; α = 1) | `5e-4` | `0.01` | `(0.9, 0.98)` | `1e-8` |
| AdaBelief | `5e-4` | `0.01` | `(0.9, 0.98)` | `1e-8` |
| Lion | `5e-5` | `0.1` | `(0.95, 0.98)` | Not applicable |

The additional ablation runs use the following configuration:

| Setting | Value |
|---|---|
| Optimizer | `emojiabl` |
| Centering configuration | α = 0 |
| Learning rate | `5e-4` |
| Weight decay | `0.01` |
| Betas | `(0.9, 0.98)` |
| Epsilon | `1e-8` |
| Random seeds | 1993, 1402, 2025 |

Unless explicitly labeled **EMOJI (α = 0, Without Centering)**, EMOJI refers to the proposed configuration with α = 1. The value α is not an EMOJI hyperparameter; it is used only in the ablation study to indicate whether gradient prediction error centering is enabled or disabled.

### Run directories and saved files

Each training run has a separate output directory under `checkpoints/`. The directory structure distinguishes optimizer configurations, their training settings, and individual random seeds.

For the four standard optimizers, the general directory structure is:

```text
checkpoints/
└── wikitext-2_transformer_lm_gpt_<optimizer>/
    └── Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD<weight_decay>_WU1000_LR<lr>_Schedinverse_sqrt_TPS128_MT1024_UF4/
        └── seed_<seed>/
            ├── train.log
            ├── tensorboard/
            └── Fairseq checkpoint files
```

The additional EMOJI no-centering runs are stored separately:

```text
checkpoints/
└── wikitext-2_transformer_lm_gpt_emojiabl/
    └── Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD0.01_WU1000_LR5e-4_Schedinverse_sqrt_TPS128_MT1024_UF4_Alpha0/
        ├── seed_1993/
        ├── seed_1402/
        └── seed_2025/
```

Each seed-specific directory contains the corresponding `train.log`, a `tensorboard/` directory, and the checkpoint files produced by Fairseq.

The training logs use JSON-formatted output and record the metrics needed by the plotting scripts. The separate directory structure also allows each training run to be inspected independently.

### TensorBoard logging and seed-by-seed comparison

The script passes `--tensorboard-logdir` to Fairseq, assigning a separate TensorBoard event directory to each optimizer–seed combination.

After training, launch TensorBoard from the project environment:

```bash
tensorboard --logdir checkpoints
```

Then open the local URL printed in the terminal, typically:

`http://localhost:6006`

Because the event directories are nested under `checkpoints/`, TensorBoard can display the runs together when their event files are present under that directory.

TensorBoard supports two complementary comparisons:

- **Compare optimizers using the same seed:** select AdamW, EMOJI, AdaBelief, and Lion for seed `1993`, for example, and compare their recorded training and validation trajectories. Repeat for seeds `1402` and `2025`. In the ablation study, include the corresponding EMOJI (α = 0) run.
- **Compare seeds using the same optimizer:** select seeds `1993`, `1402`, and `2025` for one optimizer to inspect how its training and validation trajectories vary across random seeds.

For a seed-matched comparison, use the same seed across all optimizer configurations. For a seed-variability comparison, keep the optimizer fixed and inspect all three seeds.

The event files provide the interactive training curves, while the plotting scripts generate the CSV summaries and publication-ready figures from the recorded `train.log` files.

#### Filtering runs and metrics

To compare validation metrics across all five optimizer configurations for a specific random seed, use the following regular expression in TensorBoard:

```regex
^(?=.*\*seed_1993)(?=.\**\bvalid\b).\*$
```

This expression is intended to select the validation metrics associated with seed `1993` across all five configurations:

- AdamW
- AdaBelief
- Lion
- EMOJI (α = 0, without centering)
- EMOJI (α = 1, proposed)

To perform the same comparison for another seed, replace `1993` in the expression with `1402` or `2025`, respectively.

This allows a seed-matched comparison of the validation metrics across all five configurations while keeping the random seed consistent.

### Running the training script

Open PowerShell in the Fairseq project directory containing `data-bin/wikitext-2`, make sure the configured Python environment and Fairseq installation are available, and run:

```powershell
.\run_wiki_tr_gpt_loop.ps1
```

The relative path from the project root is `fairseq\run_wiki_tr_gpt_loop.ps1`.

The script launches the standard optimizer runs in the specified order and subsequently executes the three additional EMOJI no-centering runs.

The 12 standard runs have exit-code checks and are recorded in the `$failedRuns` summary. In the supplied script, the three standalone `emojiabl` commands are outside that loop and are not included in its final failure summary.

The script uses the existing WikiText-2 binarized dataset and does not perform data preprocessing. It saves each run's logs and checkpoints to its corresponding output directory, allowing the resulting experiments to be analyzed with the plotting scripts documented above.