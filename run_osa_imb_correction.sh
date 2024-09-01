

#python DL_OSA_imb_WO_training.py --model "gru" --batch_size 256 --epochs 10  --target_col "Severity" --imb "SMOTETomek"
python DL_OSA_imb_WO_training.py --model "gru" --batch_size 256 --epochs 10  --target_col "Severity" --imb "ADASYN"

# for m in  "gru"
# do
#     for t in "AHI_30"
#     do

#         for i in "SMOTE"
#         do
#             python DL_OSA_imb_today.py --model $m --batch_size 256 --epochs 10  --target_col $t --imb $i 
#         done
        
#     done
# done


# for m in  "gru"
# do
#     for t in "AHI_15" 
#     do

#         for i in  "ADASYN"
#         do
#             python DL_OSA_imb_today.py --model $m --batch_size 256 --epochs 10  --target_col $t --imb $i 
#         done
        
#     done
# done

# for m in  "gru"
# do
#     for t in "AHI_30"
#     do

#         for i in  "BorderlineSMOTE" "SMOTETomek" "ADASYN"
#         do
#             python DL_OSA_imb_today.py --model $m --batch_size 256 --epochs 10  --target_col $t --imb $i 
#         done
        
#     done
# done