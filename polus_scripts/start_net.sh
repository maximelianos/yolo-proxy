rel_path=$(dirname "$0")                # relative
abs_path=$(cd "$rel_path" && pwd)       # absolutized and normalized

cd "$rel_path/.." && python train-vgg.py > >(tee net.log) 2>&1