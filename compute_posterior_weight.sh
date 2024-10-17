for method in 'zeroshot' 'mle'
do
  for which_model in '0' '1' '2' '3' '4' '5' '6' '7'
  do
    if method=='zeroshot':
      for dnm in 'imagenet' 'imagenet_a' 'imagenet_r'
        if dnm=='imagenet_a'
          dnmar = 'imagenet_a'
        if dnm=='imagenet_r'
          dnmar = 'imagenet_r'




        python3 compute_Hessian_zeroshot_mle.py \
        --device 'cpu' \
        --method ${method}\
        --which-model-to-use ${which_model}\
        --data-name-model ${dnm}\
        --data-name-marginal ${dnmar}
    done
  done
done