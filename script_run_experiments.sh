resnet_experiments=("resnet50" "resnet101" "resnet152")

for method in "${resnet_experiments[@]}"; do
    python run.py --method ${method} --output "${method}.png" --batch_n 64
done

efficientnet_experiments=("efficientnet_v2_s" "efficientnet_v2_m" "efficientnet_v2_l")

for method in "${efficientnet_experiments[@]}"; do
    python run.py --method ${method} --output "${method}.png" --batch_n 64
done

vit_experiments=("vit_b_16")

for method in "${vit_experiments[@]}"; do
    python run.py --method ${method} --output "${method}.png" --batch_n 64
done

# vgg_experiments=("vgg16" "vgg19")

# for method in "${vgg_experiments[@]}"; do
#     python run.py --method ${method} --output "${method}.png" --batch_n 32
# done
