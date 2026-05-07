# PyTorchのTips
## 平均値の求め方
- 悪い例：`acc_train = (pred == labels_train).sum() / len(labels_train)`

    「PyTorchのテンソル÷Pythonの整数」という異種交配の計算が発生している．

- 良い例：`acc_train = (pred == labels_train).float().mean()`

    全てがPyTorchのテンソル演算内で完結している．演算が全てGPUで処理されるため，オーバーヘッドが減る．`float()`は，ブール型を`float32`へ変換している．

## `flatten()`
- `flatten()`：全部つぶして1次元にする．
```python
x.shape # (10, 1, 5)

x.flatten().shape # (50, )
```

## `squeeze()`と`unsqueeze()`
- `squeeze()`：サイズ1の余計な次元だけ消す．
```python
x.shape # (10, 1, 5)

x.squeeze().shape # (10, 5)
```
- `unsqueeze()`：サイズ1の次元を1つ追加する．モデルに入力するデータの次元を調整する時などに使うよ．
```python
x.shape # (30, )

x.unsqueeze(0).shape # (1, 30)
x.unsqueeze(1).shape # (30, 1)
```

## softmax関数に指数関数が使われている理由
$$
y_i = \frac{exp(x_i)}{\sum\limits_{k=1}^{n} exp(x_k)}
$$
- softmax関数の目的：スコアを確率に変換したい，かつ，最も有力な候補をはっきり選びたい．
- 直感的な疑問：確率を求めるなら，スコア$x_i$をスコアの総和で割ればよいのではないか？
- スコアを指数関数に通す理由：
    - スコアは負の値を取る事があり，スコア$x_i$をスコアの総和で割っても確率$[0, 1]$へ変換できないため．
    - 指数関数$e^x$が正の実数を返す単調増加関数であるため．大小の比率は保たれないが，大小関係は維持される．
    - また，指数関数によって大きい値がより強調されるため，「最も大きいスコアを持つクラス」を際立たせやすい．
- 余談：なのでsoftmaxの世界では，負のスコアたちはかなり厳しい扱いを受けます…．🥲
