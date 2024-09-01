
for m in  "gru"
do
    for t in "AHI_30"
    do        
            python DL_OSA_interpretebility_SHAP.py --model $m   --target_col $t
    done
  
done

