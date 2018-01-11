# export INPUT=../data/dev.de
export INPUT=../data/devtest.de
rm -rf ../work
mkdir ../work

python3 translate.py ../data/train.de-en ../data/train.en $INPUT devtest

mv devtest* ../work
