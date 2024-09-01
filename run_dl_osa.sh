for m in "dbn" "gcn" "rnn" "gru" 
do
    for t in "Severity" "AHI_5" "AHI_15" "AHI_30"
    do
        #python DL_OSA_Eval.py --model $m --batch_size 256 --epochs 10  --target_col $t     
        python DL_OSA_EVAL_WO_training.py --model $m --batch_size 256 --epochs 10  --target_col $t     

    done
done
