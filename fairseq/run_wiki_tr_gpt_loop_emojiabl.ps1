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