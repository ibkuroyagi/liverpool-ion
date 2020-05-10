#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace/virtualenvs/py36/bin/activate
# 作成したPythonスクリプトを実行 
export OMP_NUM_THREADS=8
python -u quantize.py
