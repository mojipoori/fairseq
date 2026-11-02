# Medium-sized GPT-style Transformer on WikiText-2
# Execution order:
# Seed 1993: AdamW, Emoji, AdaBelief, Lion
# Seed 1402: AdamW, Emoji, AdaBelief, Lion
# Seed 2025: AdamW, Emoji, AdaBelief, Lion

$modelArch = "transformer_lm_gpt"

# Architecture
$layers = 4
$embedDim = 128
$heads = 4
$ffnDim = 512
$dropout = 0.1
$attentionDropout = 0.1
$activationFn = "gelu"

# Shared training configuration
$warmupUpdates = 1000
$warmupInitLr = "1e-7"
$scheduler = "inverse_sqrt"
$tokensPerSample = 128
$maxTokens = 1024
$updateFreq = 4
$maxEpoch = 50

# Requested seed order
$seeds = @(1993, 1402, 2025)

# Optimizer-specific configuration
# Preserve the LR, weight decay, and betas from your message.
$optimizerConfigs = @(
    [PSCustomObject]@{
        Name = "adamw"
        LR = "5e-4"
        WeightDecay = "0.01"
        Betas = "(0.9, 0.98)"
        Eps = "1e-8"
    },
    [PSCustomObject]@{
        Name = "emoji"
        LR = "5e-4"
        WeightDecay = "0.01"
        Betas = "(0.9, 0.98)"
        Eps = "1e-8"
    },
    [PSCustomObject]@{
        Name = "adabelief"
        LR = "5e-4"
        WeightDecay = "0.01"
        Betas = "(0.9, 0.98)"
        Eps = "1e-8"
    },
    [PSCustomObject]@{
        Name = "lion"
        LR = "5e-5"
        WeightDecay = "0.1"
        Betas = "(0.95, 0.98)"
        Eps = $null
    }
)

$failedRuns = @()

foreach ($seed in $seeds) {
    foreach ($opt in $optimizerConfigs) {

        $optimizer = $opt.Name
        $lr = $opt.LR
        $weightDecay = $opt.WeightDecay

        # Build optimizer-specific arguments.
        # AdamW uses the Adam argument names exposed by your CLI help.
        switch ($optimizer) {
            "adamw" {
                $optimizerArgs = @(
                    "--adam-betas"
                    $opt.Betas
                    "--adam-eps"
                    $opt.Eps
                )
            }
            "emoji" {
                $optimizerArgs = @(
                    "--emoji-betas"
                    $opt.Betas
                    "--emoji-eps"
                    $opt.Eps
                )
            }
            "adabelief" {
                $optimizerArgs = @(
                    "--adabelief-betas"
                    $opt.Betas
                    "--adabelief-eps"
                    $opt.Eps
                )
            }
            "lion" {
                $optimizerArgs = @(
                    "--lion-betas"
                    $opt.Betas
                )
            }
        }

        # Configuration-specific output directory.
        $config = "Arch${modelArch}_L${layers}_E${embedDim}_H${heads}_F${ffnDim}_D${dropout}_AD${attentionDropout}_WD${weightDecay}_WU${warmupUpdates}_LR${lr}_Sched${scheduler}_TPS${tokensPerSample}_MT${maxTokens}_UF${updateFreq}"

        $runDir = "checkpoints/wikitext-2_${modelArch}_${optimizer}/${config}/seed_${seed}"

        New-Item -ItemType Directory -Force $runDir | Out-Null

        Write-Host ""
        Write-Host "=================================================="
        Write-Host "Seed: $seed"
        Write-Host "Optimizer: $optimizer"
        Write-Host "Learning rate: $lr"
        Write-Host "Weight decay: $weightDecay"
        Write-Host "Output: $runDir"
        Write-Host "=================================================="

        fairseq-train data-bin/wikitext-2 `
            --task language_modeling `
            --arch $modelArch `
            --decoder-layers $layers `
            --decoder-embed-dim $embedDim `
            --decoder-attention-heads $heads `
            --decoder-ffn-embed-dim $ffnDim `
            --activation-fn $activationFn `
            --dropout $dropout `
            --attention-dropout $attentionDropout `
            --share-decoder-input-output-embed `
            --optimizer $optimizer `
            @optimizerArgs `
            --clip-norm 0.0 `
            --lr-scheduler $scheduler `
            --lr $lr `
            --warmup-updates $warmupUpdates `
            --warmup-init-lr $warmupInitLr `
            --weight-decay $weightDecay `
            --tokens-per-sample $tokensPerSample `
            --sample-break-mode none `
            --max-tokens $maxTokens `
            --update-freq $updateFreq `
            --log-interval 100 `
            --log-format json `
            --log-file "$runDir/train.log" `
            --save-dir $runDir `
            --seed $seed `
            --fixed-validation-seed $seed `
            --max-epoch $maxEpoch `
            --num-workers 0 `
            --tensorboard-logdir "$runDir/tensorboard"

        $exitCode = $LASTEXITCODE

        if ($exitCode -ne 0) {
            Write-Error "Training failed: optimizer=$optimizer, seed=$seed, exit_code=$exitCode"
            $failedRuns += "$optimizer / seed $seed (exit code $exitCode)"
        }
        else {
            Write-Host "Training command finished successfully: $optimizer / seed $seed"
        }
    }
}

Write-Host ""
Write-Host "================ Final summary ================"

if ($failedRuns.Count -eq 0) {
    Write-Host "All 12 training runs completed successfully."
}
else {
    Write-Host "$($failedRuns.Count) run(s) failed:"
    foreach ($failedRun in $failedRuns) {
        Write-Host " - $failedRun"
    }
}


#################################### emojiabl
$pythonExe = (Get-Command python.exe -ErrorAction Stop).Source

& $pythonExe -m fairseq_cli.train data-bin/wikitext-2 `
  --task language_modeling `
  --arch transformer_lm_gpt `
  --optimizer emojiabl `
  --emojiabl-betas "(0.9, 0.98)" `
  --emojiabl-eps 1e-8 `
  --emojiabl-alphaa 0 `
  --lr 5e-4 `
  --weight-decay 0.01 `
  --lr-scheduler inverse_sqrt `
  --warmup-updates 1000 `
  --warmup-init-lr 1e-7 `
  --max-epoch 50 `
  --max-tokens 1024 `
  --update-freq 4 `
  --tokens-per-sample 128 `
  --sample-break-mode none `
  --decoder-layers 4 `
  --decoder-embed-dim 128 `
  --decoder-attention-heads 4 `
  --decoder-ffn-embed-dim 512 `
  --dropout 0.1 `
  --attention-dropout 0.1 `
  --activation-fn gelu `
  --share-decoder-input-output-embed `
  --clip-norm 0.0 `
  --seed 1993 `
  --fixed-validation-seed 1993 `
  --num-workers 0 `
  --log-format json `
  --log-interval 100 `
  --log-file "checkpoints/wikitext-2_transformer_lm_gpt_emojiabl/Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD0.01_WU1000_LR5e-4_Schedinverse_sqrt_TPS128_MT1024_UF4_Alpha0/seed_1993/train.log" `
  --tensorboard-logdir "checkpoints/wikitext-2_transformer_lm_gpt_emojiabl/Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD0.01_WU1000_LR5e-4_Schedinverse_sqrt_TPS128_MT1024_UF4_Alpha0/seed_1993/tensorboard" `
  --save-dir "checkpoints/wikitext-2_transformer_lm_gpt_emojiabl/Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD0.01_WU1000_LR5e-4_Schedinverse_sqrt_TPS128_MT1024_UF4_Alpha0/seed_1993"

  #########################
  $pythonExe = (Get-Command python.exe -ErrorAction Stop).Source

& $pythonExe -m fairseq_cli.train data-bin/wikitext-2 `
  --task language_modeling `
  --arch transformer_lm_gpt `
  --optimizer emojiabl `
  --emojiabl-betas "(0.9, 0.98)" `
  --emojiabl-eps 1e-8 `
  --emojiabl-alphaa 0 `
  --lr 5e-4 `
  --weight-decay 0.01 `
  --lr-scheduler inverse_sqrt `
  --warmup-updates 1000 `
  --warmup-init-lr 1e-7 `
  --max-epoch 50 `
  --max-tokens 1024 `
  --update-freq 4 `
  --tokens-per-sample 128 `
  --sample-break-mode none `
  --decoder-layers 4 `
  --decoder-embed-dim 128 `
  --decoder-attention-heads 4 `
  --decoder-ffn-embed-dim 512 `
  --dropout 0.1 `
  --attention-dropout 0.1 `
  --activation-fn gelu `
  --share-decoder-input-output-embed `
  --clip-norm 0.0 `
  --seed 1402 `
  --fixed-validation-seed 1402 `
  --num-workers 0 `
  --log-format json `
  --log-interval 100 `
  --log-file "checkpoints/wikitext-2_transformer_lm_gpt_emojiabl/Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD0.01_WU1000_LR5e-4_Schedinverse_sqrt_TPS128_MT1024_UF4_Alpha0/seed_1402/train.log" `
  --tensorboard-logdir "checkpoints/wikitext-2_transformer_lm_gpt_emojiabl/Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD0.01_WU1000_LR5e-4_Schedinverse_sqrt_TPS128_MT1024_UF4_Alpha0/seed_1402/tensorboard" `
  --save-dir "checkpoints/wikitext-2_transformer_lm_gpt_emojiabl/Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD0.01_WU1000_LR5e-4_Schedinverse_sqrt_TPS128_MT1024_UF4_Alpha0/seed_1402"

  #########################################

  $pythonExe = (Get-Command python.exe -ErrorAction Stop).Source

& $pythonExe -m fairseq_cli.train data-bin/wikitext-2 `
  --task language_modeling `
  --arch transformer_lm_gpt `
  --optimizer emojiabl `
  --emojiabl-betas "(0.9, 0.98)" `
  --emojiabl-eps 1e-8 `
  --emojiabl-alphaa 0 `
  --lr 5e-4 `
  --weight-decay 0.01 `
  --lr-scheduler inverse_sqrt `
  --warmup-updates 1000 `
  --warmup-init-lr 1e-7 `
  --max-epoch 50 `
  --max-tokens 1024 `
  --update-freq 4 `
  --tokens-per-sample 128 `
  --sample-break-mode none `
  --decoder-layers 4 `
  --decoder-embed-dim 128 `
  --decoder-attention-heads 4 `
  --decoder-ffn-embed-dim 512 `
  --dropout 0.1 `
  --attention-dropout 0.1 `
  --activation-fn gelu `
  --share-decoder-input-output-embed `
  --clip-norm 0.0 `
  --seed 2025 `
  --fixed-validation-seed 2025 `
  --num-workers 0 `
  --log-format json `
  --log-interval 100 `
  --log-file "checkpoints/wikitext-2_transformer_lm_gpt_emojiabl/Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD0.01_WU1000_LR5e-4_Schedinverse_sqrt_TPS128_MT1024_UF4_Alpha0/seed_2025/train.log" `
  --tensorboard-logdir "checkpoints/wikitext-2_transformer_lm_gpt_emojiabl/Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD0.01_WU1000_LR5e-4_Schedinverse_sqrt_TPS128_MT1024_UF4_Alpha0/seed_2025/tensorboard" `
  --save-dir "checkpoints/wikitext-2_transformer_lm_gpt_emojiabl/Archtransformer_lm_gpt_L4_E128_H4_F512_D0.1_AD0.1_WD0.01_WU1000_LR5e-4_Schedinverse_sqrt_TPS128_MT1024_UF4_Alpha0/seed_2025"