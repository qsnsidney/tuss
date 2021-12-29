#!/bin/bash

echo "> Hello Sidney, are you ready to continue?"
echo "No, you are not ready. To be ready, check this script"
exit

echo "> You are not supposed to see this line. If you see this, you should be sure you modified the next part of the script"
cd ~
echo "< $(pwd)"

echo "> Preparing repo.."
mkdir -p ece1782proj
cd ece1782proj
mkdir -p exe_repo
cd exe_repo
echo "< $(pwd)"
rm -rv ./*
git clone https://github.com/qsnsidney/tuss.git
cd ..
mkdir -p repo
cd repo
echo "< $(pwd)"
rm -rv ./*
git clone https://github.com/qsnsidney/tuss.git

echo "> Making"
cd ../../exe_repo/tuss
make
echo "< $(pwd)"

echo "> Running benchmark script for CPU-GPU Cross-Comparison"
cd ../../repo/tuss
echo "! Results will be in $(pwd)/tmp !"

echo "> You can now add in more benchmark scripts after here in the bash script"
# I am the fen ge xian, please notice me
# Yo, here ==============================|
#                                        v
python3 -m scripts.benchmark cpu --version=0 --exe=../../exe_repo/tuss/build/cpusim/cpusim_exe








#                                        ^
# Above this line =======================|
# Not here!