# for m in "dbn" "gcn" "rnn" "gru"
# do
#     for t in "Severity" "AHI_5" "AHI_15" "AHI_30"
#     do
#         for i in "SMOTE" "BorderlineSMOTE" "SMOTETomek" "ADASYN" "none"
#         do
#             for f in "roc_curve" "precision_recall_curve" "confusion_matrix" "scores"
#             do
#             python visuals.py --model $m --target_col $t --figure_type $f --imb $i
#             done
#         done
#     done
# done
for m in "gru"
do
    for t in "AHI_15" "AHI_30" 
    do
        for i in "SMOTE" "BorderlineSMOTE" "SMOTETomek" "ADASYN" 
        do
            for f in  "confusion_matrix" 
            do
            python visuals.py --model $m --target_col $t --figure_type $f --imb $i
            done
        done
    done
done

#for m in "dbn" 
# do
#     for t in "Severity" "AHI_5" "AHI_15" "AHI_30"
#     do
#         for i in "SMOTE" "BorderlineSMOTE" "SMOTETomek" "ADASYN" "none"
#         do
#             for f in "scores"
#             do
#             python visuals.py --model $m --target_col $t --figure_type $f --imb $i
#             done
#         done
#     done
# done