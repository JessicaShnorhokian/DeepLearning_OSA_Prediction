#!/bin/bash


for m in "dbn" "gcn" "rnn"  "cnn" "gru" "lstm" 
do
    for t in "Severity" "AHI_5" "AHI_15" "AHI_30"
    do
        for f in "Wu" "Mencar" "Ustun" "Huang" "Rodruiges" "KBest Chi2" "RF Impurity" "RF Permutation" "KBest Fclass" "Correlation" "Kruskall Chi" "CatBoost" "SHAP_RF"

        do
            for i in "SMOTE" "BorderlineSMOTE" "SMOTETomek" "ADASYN"
            do
                python DL_OSA_feature_selection.py --model $m --batch_size 1024 --epochs 10  --target_col $t --imb $i --features_name $f
            done
        done
    done
done

# for m in   "rnn"  "cnn" "gru" "lstm" 
# do
#     for t in "Severity" "AHI_5" "AHI_15" "AHI_30"
#     do
#         for f in "Demographic" "Measurements" "Symptoms" "Questionnaires" "Comorbidities" "Wu" "Mencar" "Ustun" "Huang" "Rodruiges" "KBest Chi2" "RF Impurity" "RF Permutation" "KBest Fclass" "Correlation" "Kruskall Chi" "CatBoost" "SHAP_RF"
#         do
#             for i in "SMOTE" "BorderlineSMOTE" "SMOTETomek" "ADASYN"
#             do
#                 python DL_OSA_feature_selection.py --model $m --batch_size 1024 --epochs 10  --target_col $t --imb $i --features_name $f
#             done
#         done
#     done
# done

#rnn
# python DL_OSA_imb_today.py --model rnn --epochs 10 --target_col AHI_5 --imb ADASYN

#gru
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col AHI_5 --imb SMOTE
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col AHI_15 --imb SMOTE
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col AHI_30 --imb SMOTE

#gru
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col Severity --imb SMOTE &

# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col AHI_5 --imb BorderlineSMOTE &
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col AHI_15 --imb BorderlineSMOTE &
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col AHI_30 --imb BorderlineSMOTE &
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col Severity --imb BorderlineSMOTE & 

# wait

# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col AHI_5 --imb SMOTETomek  &
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col AHI_15 --imb SMOTETomek  &
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col AHI_30 --imb SMOTETomek  &
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col Severity --imb SMOTETomek &

# wait

# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col AHI_5 --imb ADASYN  &
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col AHI_15 --imb ADASYN  &
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col AHI_30 --imb ADASYN  &
# python DL_OSA_imb_today.py --model gru --epochs 10 --target_col Severity --imb ADASYN  &

# wait


# #cnn
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col AHI_5 --imb SMOTE  &
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col AHI_15 --imb SMOTE  &
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col AHI_30 --imb SMOTE  &
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col Severity --imb SMOTE  &

# wait

# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col AHI_5 --imb BorderlineSMOTE  &
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col AHI_15 --imb BorderlineSMOTE  &
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col AHI_30 --imb BorderlineSMOTE  &
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col Severity --imb BorderlineSMOTE  &

# wait

# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col AHI_5 --imb SMOTETomek  &
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col AHI_15 --imb SMOTETomek  &
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col AHI_30 --imb SMOTETomek  &
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col Severity --imb SMOTETomek  &

# wait

# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col AHI_5 --imb ADASYN  &
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col AHI_15 --imb ADASYN  &
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col AHI_30 --imb ADASYN  &
# python DL_OSA_imb_today.py --model cnn --epochs 10 --target_col Severity --imb ADASYN  &

# wait


#lstm
# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col AHI_5 --imb SMOTE  &
# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col AHI_15 --imb SMOTE  &

# wait

# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col AHI_30 --imb SMOTE  &
# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col Severity --imb SMOTE  &

# wait

# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col AHI_5 --imb BorderlineSMOTE  &
# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col AHI_15 --imb BorderlineSMOTE  &

# wait

# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col AHI_30 --imb BorderlineSMOTE  &
# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col Severity --imb BorderlineSMOTE  &

# wait

# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col AHI_5 --imb SMOTETomek  &
# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col AHI_15 --imb SMOTETomek  &

# wait

# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col AHI_30 --imb SMOTETomek  &
# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col Severity --imb SMOTETomek  &

# wait

# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col AHI_5 --imb ADASYN  &
# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col AHI_15 --imb ADASYN  &

# wait

# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col AHI_30 --imb ADASYN  &
# python DL_OSA_imb_today.py --model lstm --epochs 10 --target_col Severity --imb ADASYN  &


