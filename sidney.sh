#!/bin/bash

exe_root_repo="$HOME/ece1782proj/exe_repo"
# Use this variable
# =============================
exe_repo="$exe_root_repo/tuss"
# =============================
echo "exe_repo(git) is $exe_repo"

w_root_repo="$HOME/ece1782proj/repo"
# Use this variable
# =============================
w_repo="$w_root_repo/tuss"
# =============================
echo "w_repo(git) is $w_repo"

echo "> Hello Sidney, are you ready to continue?"
echo "No, you are not ready. To be ready, check this script"
exit

echo "> You are not supposed to see this line. If you see this, you should be sure you modified the next part of the script"
cd ~
echo "< $(pwd)"

echo "> Preparing repo.."
mkdir -p $exe_root_repo
cd $exe_root_repo
echo "< $(pwd)"
rm -rv ./*
git clone https://github.com/qsnsidney/tuss.git
mkdir -p $w_root_repo
cd $w_root_repo
echo "< $(pwd)"
rm -rv ./*
git clone https://github.com/qsnsidney/tuss.git

# Here is how you can quick rebuild 
echo "> Building exe_repo"
make -C $exe_repo -j8

echo "> Running benchmark script for CPU-GPU Cross-Comparison"
echo "! Results will be in $w_repo/tmp"

echo "> You can now add in more benchmark scripts after here in the bash script"
# I am the fen ge xian, please notice me
# Yo, here ==============================|
#                                        v
cd $w_repo
python3 -m scripts.benchmark cpu --version=0 --exe="$exe_repo/build/cpusim/cpusim_exe"








#                                        ^
# Above this line =======================|
# Not here!