## 研究室PCの初期設定
```
# 仮想環境の作成
conda create -n pytorch-playground python=3.10 -y
conda activate pytorch-playground

# PyTorchのインストール（GPUが新しいので下げると動かないよ．）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# リポジトリを置くディレクトリへ移動
cd <WORKSPACE_DIR>
# リポジトリのクローン（PATを使用）
git clone https://<TOKEN>@github.com/<USERNAME>/<REPO>.git
cd <REPO>

# 残りの手書きリストからインストール
pip install -r requirements.txt

# 作成した仮想環境にipykernelをインストール
conda install -c anaconda ipykernel
ipython kernel install --user --name pytorch-playground
# 確認
jupyter kernelspec list
```

## 研究室PCの運用フロー
```
conda activate pytorch-playground
cd <WORKSPACE_DIR>/<REPO>
git pull
jupyter notebook
```
