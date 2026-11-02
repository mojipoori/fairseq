

   
    fairseq-train data-bin/iwslt14.tokenized.de-en `
    --arch transformer_iwslt_de_en `
    --share-decoder-input-output-embed `
    --optimizer adamw `
    --adam-betas '(0.9, 0.98)' `
    --clip-norm 0.0 `
    --lr 5e-4 `
    --lr-scheduler inverse_sqrt `
    --warmup-updates 4000 `
    --dropout 0.3 `
    --weight-decay 0.0001 `
    --criterion label_smoothed_cross_entropy `
    --label-smoothing 0.1 `
    --max-tokens 4096 `
    --eval-bleu `
    --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
    --eval-bleu-detok moses `
    --eval-bleu-remove-bpe `
    --eval-bleu-print-samples `
    --best-checkpoint-metric bleu `
    --maximize-best-checkpoint-metric `
    --log-format json `
    --log-file checkpoints_of_seeds/seed2025/iwslt14_adamw/train.log `
    --save-dir checkpoints_of_seeds/seed2025/iwslt14_adamw `
    --seed 2025 `
    --fixed-validation-seed 2025 `
    --max-epoch 50        

    
    fairseq-train data-bin/iwslt14.tokenized.de-en `
    --arch transformer_iwslt_de_en `
    --share-decoder-input-output-embed `
    --optimizer emoji `
    --emoji-betas '(0.9, 0.98)' `
    --clip-norm 0.0 `
    --lr 5e-4 `
    --lr-scheduler inverse_sqrt `
    --warmup-updates 4000 `
    --dropout 0.3 `
    --weight-decay 0.0001 `
    --criterion label_smoothed_cross_entropy `
    --label-smoothing 0.1 `
    --max-tokens 4096 `
    --eval-bleu `
    --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
    --eval-bleu-detok moses `
    --eval-bleu-remove-bpe `
    --eval-bleu-print-samples `
    --best-checkpoint-metric bleu `
    --maximize-best-checkpoint-metric `
    --log-format json `
    --log-file checkpoints_of_seeds/seed2025/iwslt14_emoji/train.log `
    --save-dir checkpoints_of_seeds/seed2025/iwslt14_emoji `
    --seed 2025 `
    --fixed-validation-seed 2025 `
    --max-epoch 50    


    fairseq-train data-bin/iwslt14.tokenized.de-en `
    --arch transformer_iwslt_de_en `
    --share-decoder-input-output-embed `
    --optimizer adabelief `
    --adabelief-betas '(0.9, 0.98)' `
    --clip-norm 0.0 `
    --lr 5e-4 `
    --lr-scheduler inverse_sqrt `
    --warmup-updates 4000 `
    --dropout 0.3 `
    --weight-decay 0.0001 `
    --criterion label_smoothed_cross_entropy `
    --label-smoothing 0.1 `
    --max-tokens 4096 `
    --eval-bleu `
    --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
    --eval-bleu-detok moses `
    --eval-bleu-remove-bpe `
    --eval-bleu-print-samples `
    --best-checkpoint-metric bleu `
    --maximize-best-checkpoint-metric `
    --log-format json `
    --log-file checkpoints_of_seeds/seed2025/iwslt14_adabelief/train.log `
    --save-dir checkpoints_of_seeds/seed2025/iwslt14_adabelief `
    --seed 2025 `
    --fixed-validation-seed 2025 `
    --max-epoch 50 

    fairseq-train data-bin/iwslt14.tokenized.de-en `
    --arch transformer_iwslt_de_en `
    --share-decoder-input-output-embed `
    --optimizer emojiabl `
    --emojiabl-alphaa 0 `
    --emojiabl-betas '(0.9, 0.98)' `
    --clip-norm 0.0 `
    --lr 5e-4 `
    --lr-scheduler inverse_sqrt `
    --warmup-updates 4000 `
    --dropout 0.3 `
    --weight-decay 0.0001 `
    --criterion label_smoothed_cross_entropy `
    --label-smoothing 0.1 `
    --max-tokens 4096 `
    --eval-bleu `
    --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
    --eval-bleu-detok moses `
    --eval-bleu-remove-bpe `
    --eval-bleu-print-samples `
    --best-checkpoint-metric bleu `
    --maximize-best-checkpoint-metric `
    --log-format json `
    --log-file checkpoints_of_seeds/seed2025/iwslt14_emojiabl/train.log `
    --save-dir checkpoints_of_seeds/seed2025/iwslt14_emojiabl `
    --seed 2025 `
    --fixed-validation-seed 2025 `
    --max-epoch 50 

    fairseq-train data-bin/iwslt14.tokenized.de-en `
    --arch transformer_iwslt_de_en `
    --share-decoder-input-output-embed `
    --optimizer lion `
    --lion-betas "(0.95,0.98)" `
    --clip-norm 0.0 `
    --lr 5e-5 `
    --lr-scheduler inverse_sqrt `
    --warmup-updates 4000 `
    --dropout 0.3 `
    --weight-decay 0.001 `
    --criterion label_smoothed_cross_entropy `
    --label-smoothing 0.1 `
    --max-tokens 4096 `
    --eval-bleu `
    --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
    --eval-bleu-detok moses `
    --eval-bleu-remove-bpe `
    --eval-bleu-print-samples `
    --best-checkpoint-metric bleu `
    --maximize-best-checkpoint-metric `
    --log-format json `
    --log-file checkpoints_of_seeds/seed2025/iwslt14_lion/train.log `
    --save-dir checkpoints_of_seeds/seed2025/iwslt14_lion `
    --seed 2025 `
    --fixed-validation-seed 2025 `
    --max-epoch 50

###############################################################################################


   
fairseq-train data-bin/iwslt14.tokenized.de-en `
--arch transformer_iwslt_de_en `
--share-decoder-input-output-embed `
--optimizer adamw `
--adam-betas '(0.9, 0.98)' `
--clip-norm 0.0 `
--lr 5e-4 `
--lr-scheduler inverse_sqrt `
--warmup-updates 4000 `
--dropout 0.3 `
--weight-decay 0.0001 `
--criterion label_smoothed_cross_entropy `
--label-smoothing 0.1 `
--max-tokens 4096 `
--eval-bleu `
--eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
--eval-bleu-detok moses `
--eval-bleu-remove-bpe `
--eval-bleu-print-samples `
--best-checkpoint-metric bleu `
--maximize-best-checkpoint-metric `
--log-format json `
--log-file checkpoints_of_seeds/seed1993/iwslt14_adamw/train.log `
--save-dir checkpoints_of_seeds/seed1993/iwslt14_adamw `
--seed 1993 `
--fixed-validation-seed 1993 `
--max-epoch 50        


fairseq-train data-bin/iwslt14.tokenized.de-en `
--arch transformer_iwslt_de_en `
--share-decoder-input-output-embed `
--optimizer emoji `
--emoji-betas '(0.9, 0.98)' `
--clip-norm 0.0 `
--lr 5e-4 `
--lr-scheduler inverse_sqrt `
--warmup-updates 4000 `
--dropout 0.3 `
--weight-decay 0.0001 `
--criterion label_smoothed_cross_entropy `
--label-smoothing 0.1 `
--max-tokens 4096 `
--eval-bleu `
--eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
--eval-bleu-detok moses `
--eval-bleu-remove-bpe `
--eval-bleu-print-samples `
--best-checkpoint-metric bleu `
--maximize-best-checkpoint-metric `
--log-format json `
--log-file checkpoints_of_seeds/seed1993/iwslt14_emoji/train.log `
--save-dir checkpoints_of_seeds/seed1993/iwslt14_emoji `
--seed 1993 `
--fixed-validation-seed 1993 `
--max-epoch 50    


fairseq-train data-bin/iwslt14.tokenized.de-en `
--arch transformer_iwslt_de_en `
--share-decoder-input-output-embed `
--optimizer adabelief `
--adabelief-betas '(0.9, 0.98)' `
--clip-norm 0.0 `
--lr 5e-4 `
--lr-scheduler inverse_sqrt `
--warmup-updates 4000 `
--dropout 0.3 `
--weight-decay 0.0001 `
--criterion label_smoothed_cross_entropy `
--label-smoothing 0.1 `
--max-tokens 4096 `
--eval-bleu `
--eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
--eval-bleu-detok moses `
--eval-bleu-remove-bpe `
--eval-bleu-print-samples `
--best-checkpoint-metric bleu `
--maximize-best-checkpoint-metric `
--log-format json `
--log-file checkpoints_of_seeds/seed1993/iwslt14_adabelief/train.log `
--save-dir checkpoints_of_seeds/seed1993/iwslt14_adabelief `
--seed 1993 `
--fixed-validation-seed 1993 `
--max-epoch 50 

fairseq-train data-bin/iwslt14.tokenized.de-en `
--arch transformer_iwslt_de_en `
--share-decoder-input-output-embed `
--optimizer emojiabl `
--emojiabl-alphaa 0 `
--emojiabl-betas '(0.9, 0.98)' `
--clip-norm 0.0 `
--lr 5e-4 `
--lr-scheduler inverse_sqrt `
--warmup-updates 4000 `
--dropout 0.3 `
--weight-decay 0.0001 `
--criterion label_smoothed_cross_entropy `
--label-smoothing 0.1 `
--max-tokens 4096 `
--eval-bleu `
--eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
--eval-bleu-detok moses `
--eval-bleu-remove-bpe `
--eval-bleu-print-samples `
--best-checkpoint-metric bleu `
--maximize-best-checkpoint-metric `
--log-format json `
--log-file checkpoints_of_seeds/seed1993/iwslt14_emojiabl/train.log `
--save-dir checkpoints_of_seeds/seed1993/iwslt14_emojiabl `
--seed 1993 `
--fixed-validation-seed 1993 `
--max-epoch 50 

fairseq-train data-bin/iwslt14.tokenized.de-en `
--arch transformer_iwslt_de_en `
--share-decoder-input-output-embed `
--optimizer lion `
--lion-betas "(0.95,0.98)" `
--clip-norm 0.0 `
--lr 5e-5 `
--lr-scheduler inverse_sqrt `
--warmup-updates 4000 `
--dropout 0.3 `
--weight-decay 0.001 `
--criterion label_smoothed_cross_entropy `
--label-smoothing 0.1 `
--max-tokens 4096 `
--eval-bleu `
--eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
--eval-bleu-detok moses `
--eval-bleu-remove-bpe `
--eval-bleu-print-samples `
--best-checkpoint-metric bleu `
--maximize-best-checkpoint-metric `
--log-format json `
--log-file checkpoints_of_seeds/seed1993/iwslt14_lion/train.log `
--save-dir checkpoints_of_seeds/seed1993/iwslt14_lion `
--seed 1993 `
--fixed-validation-seed 1993 `
--max-epoch 50

######################################################


   
fairseq-train data-bin/iwslt14.tokenized.de-en `
--arch transformer_iwslt_de_en `
--share-decoder-input-output-embed `
--optimizer adamw `
--adam-betas '(0.9, 0.98)' `
--clip-norm 0.0 `
--lr 5e-4 `
--lr-scheduler inverse_sqrt `
--warmup-updates 4000 `
--dropout 0.3 `
--weight-decay 0.0001 `
--criterion label_smoothed_cross_entropy `
--label-smoothing 0.1 `
--max-tokens 4096 `
--eval-bleu `
--eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
--eval-bleu-detok moses `
--eval-bleu-remove-bpe `
--eval-bleu-print-samples `
--best-checkpoint-metric bleu `
--maximize-best-checkpoint-metric `
--log-format json `
--log-file checkpoints_of_seeds/seed1402/iwslt14_adamw/train.log `
--save-dir checkpoints_of_seeds/seed1402/iwslt14_adamw `
--seed 1402 `
--fixed-validation-seed 1402 `
--max-epoch 50        


fairseq-train data-bin/iwslt14.tokenized.de-en `
--arch transformer_iwslt_de_en `
--share-decoder-input-output-embed `
--optimizer emoji `
--emoji-betas '(0.9, 0.98)' `
--clip-norm 0.0 `
--lr 5e-4 `
--lr-scheduler inverse_sqrt `
--warmup-updates 4000 `
--dropout 0.3 `
--weight-decay 0.0001 `
--criterion label_smoothed_cross_entropy `
--label-smoothing 0.1 `
--max-tokens 4096 `
--eval-bleu `
--eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
--eval-bleu-detok moses `
--eval-bleu-remove-bpe `
--eval-bleu-print-samples `
--best-checkpoint-metric bleu `
--maximize-best-checkpoint-metric `
--log-format json `
--log-file checkpoints_of_seeds/seed1402/iwslt14_emoji/train.log `
--save-dir checkpoints_of_seeds/seed1402/iwslt14_emoji `
--seed 1402 `
--fixed-validation-seed 1402 `
--max-epoch 50    


fairseq-train data-bin/iwslt14.tokenized.de-en `
--arch transformer_iwslt_de_en `
--share-decoder-input-output-embed `
--optimizer adabelief `
--adabelief-betas '(0.9, 0.98)' `
--clip-norm 0.0 `
--lr 5e-4 `
--lr-scheduler inverse_sqrt `
--warmup-updates 4000 `
--dropout 0.3 `
--weight-decay 0.0001 `
--criterion label_smoothed_cross_entropy `
--label-smoothing 0.1 `
--max-tokens 4096 `
--eval-bleu `
--eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
--eval-bleu-detok moses `
--eval-bleu-remove-bpe `
--eval-bleu-print-samples `
--best-checkpoint-metric bleu `
--maximize-best-checkpoint-metric `
--log-format json `
--log-file checkpoints_of_seeds/seed1402/iwslt14_adabelief/train.log `
--save-dir checkpoints_of_seeds/seed1402/iwslt14_adabelief `
--seed 1402 `
--fixed-validation-seed 1402 `
--max-epoch 50 

fairseq-train data-bin/iwslt14.tokenized.de-en `
--arch transformer_iwslt_de_en `
--share-decoder-input-output-embed `
--optimizer emojiabl `
--emojiabl-alphaa 0 `
--emojiabl-betas '(0.9, 0.98)' `
--clip-norm 0.0 `
--lr 5e-4 `
--lr-scheduler inverse_sqrt `
--warmup-updates 4000 `
--dropout 0.3 `
--weight-decay 0.0001 `
--criterion label_smoothed_cross_entropy `
--label-smoothing 0.1 `
--max-tokens 4096 `
--eval-bleu `
--eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
--eval-bleu-detok moses `
--eval-bleu-remove-bpe `
--eval-bleu-print-samples `
--best-checkpoint-metric bleu `
--maximize-best-checkpoint-metric `
--log-format json `
--log-file checkpoints_of_seeds/seed1402/iwslt14_emojiabl/train.log `
--save-dir checkpoints_of_seeds/seed1402/iwslt14_emojiabl `
--seed 1402 `
--fixed-validation-seed 1402 `
--max-epoch 50 

fairseq-train data-bin/iwslt14.tokenized.de-en `
--arch transformer_iwslt_de_en `
--share-decoder-input-output-embed `
--optimizer lion `
--lion-betas "(0.95,0.98)" `
--clip-norm 0.0 `
--lr 5e-5 `
--lr-scheduler inverse_sqrt `
--warmup-updates 4000 `
--dropout 0.3 `
--weight-decay 0.001 `
--criterion label_smoothed_cross_entropy `
--label-smoothing 0.1 `
--max-tokens 4096 `
--eval-bleu `
--eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' `
--eval-bleu-detok moses `
--eval-bleu-remove-bpe `
--eval-bleu-print-samples `
--best-checkpoint-metric bleu `
--maximize-best-checkpoint-metric `
--log-format json `
--log-file checkpoints_of_seeds/seed1402/iwslt14_lion/train.log `
--save-dir checkpoints_of_seeds/seed1402/iwslt14_lion `
--seed 1402 `
--fixed-validation-seed 1402 `
--max-epoch 50
