# ノート
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

## PyTorchのCrossEntropyLoss関数がsoftmax関数と対数関数をまとめている理由
予測関数と損失関数の境界がわかりづらくなるのに，指数関数と対数関数をまとめるのには数値計算の観点による動機がある．

予測モデルが自信満々で，あるクラスに対して$x_i = 1000$という大きなロジットを出力したとする．
- softmax関数と対数関数を別々に適用する場合

    softmax関数の内，$e^{1000}$の計算がオーバーフローを起こして無限大（`inf`）となる．
    $$
    y_i = \frac{exp(x_i)}{\sum\limits_{k=1}^{n} exp(x_k)} \approx \frac{\infty}{\infty} = nan
    $$
    この時，交差エントロピー関数の中の対数関数を適用すると，
    $$
    \log(nan) = nan
    $$
    となる．この`nan`が逆伝播全体に伝染して，勾配・重みが`nan`となり学習ができなくなる．
- softmax関数と対数関数をまとめる場合
    
    詳細な式変形は省略するが，"Log-Sum-Exp Trick"を使う．`x_i`から最大値`m`を引く工夫を行う．
    $$
    \begin{align*}
        m = max(x_i) \\
        x_i - m \leq 0 \\
        exp(x_i - m) \in (0, 1]
    \end{align*}
    $$
    定義域が良いので，指数関数でオーバーフローを起こさない．

$x_i = -1000$の場合は，$e^{-1000} \approx 0.0$となる．もし，softmax関数の分母全体も$0.0$になった場合は，softmax関数が不定形で`nan`を吐き，以降の議論は同様．

## 重み行列の添字
通常，重み行列の要素$w_{ij}$は「j番目の入力からi番目の出力への重み」を指す．ニューラルネットの図を想像してね．

## `nn.BCELoss()`と`nn.CrossEntropyLoss()`で渡す正解ラベルの次元が違う理由
どうしてBCELossの正解ラベルを作る時は，わざわざ1次元テンソルを2次元に拡張したのに，CrossEntropyLossの正解ラベルは1次元テンソルじゃないといけない仕様になっているの？
```python
# nn.BCELoss()
inputs_train = torch.tensor(x_train, dtype=torch.float32)
labels_train = torch.tensor(y_train, dtype=torch.float32).view((-1, 1)) # torch.Size([N, 1])

criterion = nn.BCELoss() # 2値交差エントロピー
loss = criterion(outputs, labels_train) # 損失

# nn.CrossEntropyLoss()
inputs_train = torch.tensor(x_train, dtype=torch.float32)
labels_train = torch.tensor(y_train, dtype=torch.long) # torch.Size([N])

criterion = nn.CrossEntropyLoss() # 交差エントロピー
loss = criterion(outputs, labels_train) # 損失
```

- BCELossの数式
    $$ L_{BCE} = -\frac{1}{N} \sum_{i=1}^N \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right) $$
    - $N$：サンプル数（バッチサイズ）（スカラー）
    - $y_i$：$i$番目のサンプルの真のラベル（$0$または$1$のスカラー）
    - $\hat{y}_i$：$i$番目のサンプルに対して予測モデルが出した確率（$0$から$1$のスカラー）

- CrossEntropyLossの数式
    $$ L_{CE} = -\frac{1}{N} \sum_{i=1}^N \log\left( \frac{\exp(x_{i, y_i})}{\sum_{j=1}^C \exp(x_{i, j})} \right) $$
    - $C$：クラスの総数（スカラー）
    - $x_{i, j}$：$i$番目のサンプルの$j$番目のクラスに対する予測ロジット（スカラー）
    - $y_i$：$i$番目のサンプルの正解クラスのインデックス（$0$から$C-1$の整数スカラー）

- 回答：BCEとCEでは，正解ラベルの使い方が違うから．
    - BCE：正解ラベル$y_i$は，予測モデルの出力$\hat{y}_i$との積を直接計算される．よって，`labels`と`outputs`では，同じインデックスの要素が同じ意味を持つ必要がある．（例：[N]と[N]，[N, 1]と[N, 1]）次元が不一致な場合，お節介なブロードキャストによって意味の対応がズレる事故が起きる．
    - CE：正解ラベル$y_i$は，$i$番目のサンプル中から正解クラスの出力をピックアップするための「インデックス」として機能する．テンソル演算に参加しない「選択キー」である．ベーシックな学習においては，インデックスに2次元テンソルは冗長であり，1次元テンソルが最適である．（※正解を確率分布として直接教えたい高度なケースに限っては，CEでも2次元の正解ラベルが許容される．）

## numpy.ndarrayとtorch.Tensorの相互変換
### Tensor → NumPy
- `tensor.numpy()`：`requires_grad=False` のときのみ可（`True`ならエラー）
- `tensor.detach().numpy()`：計算グラフから切り離して変換（推奨）
- `tensor.data.numpy()`：使わないで！安全チェック無しで危険！非人道的！

### NumPy → Tensor
- `torch.from_numpy(array)`：メモリ共有（高速・コピーなし）
- `torch.tensor(array)`：コピーして新規作成

### 注意
- `.detach().numpy()` はコピーではなくメモリ共有
- 完全コピーしたい場合：`tensor.detach().numpy().copy()`
- GPUテンソルは：`tensor.cpu().detach().numpy()`

## PyTorchにおけるGPU利用のルール
- テンソル変数は，データがCPU・GPU上のどちらにあるのかを属性として持っている．
- CPUとGPU間でデータを転送する時は，to関数を使う．
- 2つの変数が両方ともGPU上にある場合，演算はGPU上で行われる．
- 変数の片方がCPU，もう一方がGPUの場合，演算はエラーになる．
