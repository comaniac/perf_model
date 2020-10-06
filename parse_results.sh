BASE_DIR_PATH=$1
FILES=(
InceptionV3.txt
MobileNet1.0.txt
MobileNetV2_1.0.txt
ResNet18_v1.txt
ResNet18_v2.txt
ResNet50_v1.txt
ResNet50_v2.txt
SqueezeNet1.0.txt
SqueezeNet1.1.txt
VGG19.txt
VGG19_bn.txt
)

for filename in ${FILES[@]};
do
  cat ${BASE_DIR_PATH}/${filename} | sed -n 's/Median inference time: \(.*\) ms/\1/p'
done;
