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
- 「モデル本体」「入力データ」「正解データ」の3つを送る事をお忘れなきよう．

## 組み込みデータセットの分割とTransform（データ拡張）の罠
- まず，`torchvision.datasets`モジュールの組み込みデータセット（MNISTやCIFAR10など）を使用する時は，引数`train`が存在する．
- `train=False`は，テストデータである．これを検証データとみなすと，テストデータが存在しなくなってしまう．よって，検証データは，`train=True`のデータセットから切り出す必要がある．
- ここで，データ拡張は一般的に学習データにのみ適用し，検証や評価データには含めない事を思い出しておく．検証や評価データのTransformにデータ拡張を混ぜてしまうと，エポックごとにデータ拡張結果が変化してしまい，指標の変化が「モデル性能の変化」に依るものか「入力データの変化」に依るものか区別できなくなってしまう．（※PyTorchのTransformは，データセットの`__getitem__`メソッドによって毎エポック実行される．）
- では，`train=True`のデータセットを訓練データ・検証データに分割してから，それぞれ異なるTransformを適用するだけの話ではないか？
- ところが，`torchvision.datasets`モジュールのインスタンスがTransformを保持するAttribute（クラスの属性）名は，データセット毎に異なる．つまり，`train_set.transform = new_transform`などと書けば，エラーも出ないのにTransformが書き換わらない場合があるという恐ろしい罠が存在する．
    > Manipulating the internal .transform attribute assumes that self.transform is indeed used to apply the transformations. While this might be the case for e.g. MNIST other datasets could use other attributes (e.g. self.image_fransform) and you would need to add this manipulation according to the real implementation (which could of course also change between releases). The right approach is thus to set the transformations once during the initialization of the Dataset and allow the Dataset to handle the transformations internally without depending on its actual implementation. (https://discuss.pytorch.org/t/how-to-apply-another-transform-to-an-existing-dataset/85416/7)
- じゃあどうすればいいのか．解決策は「元のデータセットを2つ作り，それぞれに別々のTransformを適用した上で，同じインデックス配列を使って`Subset`で抽出する」事である．
    ```python
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import Subset

    # 1. 訓練用と検証用で，元のデータセットを2つ作る．
    train_dataset_full = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform) # 訓練用フィルタ（データ拡張あり）

    valid_dataset_full = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=test_transform) # 検証用フィルタ（データ拡張なし）

    # 2. 訓練と検証のデータ数を決める．
    train_size = int(len(train_dataset_full) * 0.9)
    valid_size = len(train_dataset_full) - train_size

    # 3. データを分割するための「ランダムなインデックス」を生成
    # torch.randperm(n)：0からn-1までのランダムな整数の順列を返す．
    torch.manual_seed(42)
    indices = torch.randperm(len(train_dataset_full)).tolist()

    # 4. random_splitを使わず，Subsetを使って訓練データ・検証データを分ける．
    train_dataset = Subset(train_dataset_full, indices[:train_size])
    valid_dataset = Subset(valid_dataset_full, indices[train_size:])
    ```
    > My preference would be to create three difference datasets using the desired transformations for the training, validation, and test sets. This approach makes it clear that the train_dataset also uses the train_transform only. Once this is done, create the training, validation, and test indices via any kind of splitting (sklearn.model_selection.train_test_split is quite popular) and wrap the datasets into a Subset with the corresponding indices. (https://discuss.pytorch.org/t/custom-dataset-best-practices-for-transformations-on-training-set/155761/2)
- 「Transformは，最初に固定せよ！」

## バッチサイズは訓練・検証・テストデータ数の公約数？
- 結論：バッチサイズは公約数でなくてOK．最後の端数バッチも問題なく処理される．
- 理由：損失はバッチサイズの違いを吸収するために，平均で設計するから．
- 実務：
    - `drop_last=True` で端数を捨てる．（端数が1の場合，バッチ正規化がエラーになるため．）
    - バッチサイズを決める基準：
        - GPUに収まる最大
        - 2の累乗（128, 256, 512, …）が基本

## nn.ReLU(inplace=True)のinplaceは何のため？
### 結論
`inplace=True`は，中間テンソルを上書きしてメモリを節約するためのオプション．

### inplaceの動き
通常：
```python
a = x * 2
b = a ** 2
```
→ `a`と`b`は，別のメモリに保存される．

inplace：
```python
a = x * 2
a.pow_(2) # aを上書き
```
→ 元の`a`の値が消える．

### 中間テンソルってなんやねん（誤差逆伝播のイメージ）
計算グラフ：
```
x → f → g → h → L (loss)
```

順伝播：
```python
f = x * 2
g = f ** 2
h = g + 1
```

誤差逆伝播法で求めたい：
```
∂L/∂x
```

連鎖律：
```
∂L/∂x
= ∂L/∂g * ∂g/∂x
= ∂L/∂g * ∂g/∂f * ∂f/∂x
```

各項：
```
∂L/∂g = ∂(g + 1)/∂g = 1
∂g/∂f = ∂(f ** 2)/∂f = 2f # ここで中間テンソルfが必要！
∂f/∂x = ∂(x * 2)/∂x = 2
```
中間テンソルfは，いつの値？

→ 学習ループ中に，今のパラメータ・今の入力を使ってforwardした時の中間テンソル！

→ だから中間テンソルをキャッシュしておかなければならない．

### ReLUで`inplace=True`がOKな理由
ReLU：
```
y = max(0, x)
```

微分：
```
1 (x > 0)
0 (x ≤ 0)
```

- x > 0 ⇔ y > 0
- x ≤ 0 ⇔ y = 0

**出力yを見れば判定できる．**

### まとめ
- 中間テンソルは「逆伝播のための記録」
- 勾配は「そのときの forward に対するもの」
- inplace はその記録を壊す可能性がある
- ただし ReLU は例外的に安全

## CNNモデル
### 畳み込み層のカーネルの画素数を奇数にする理由
1.  偶数だと，入力のi+0.5（またはi-0.5）ピクセル目を重心とする情報が，出力のiピクセル目に移され，画像の情報がドリフトしてしまうから．
    - 別にドリフトしても良くね？
    
        画像分類タスクなら大きな影響は無いかもしれない．しかし，セグメンテーションや物体検出などでは，入力の座標と出力の座標が完全に一致していなければならない．「出力座標のiピクセル目に歩行者がいるよ．」と言って，それが入力座標のiから10ピクセルずれていたら，自動運転車は交通事故を起こす．
    - ドリフトの具体的イメージ

        ```python
        import torch
        import torch.nn as nn

        # 長さ10の1次元画像．インデックス5に「歩行者（1.0）」がいる
        x = torch.zeros(1, 1, 10)
        x[0, 0, 5] = 1.0 

        # 【奇数カーネル (3)】：左右対称な重み（平均化）
        conv_odd = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        conv_odd.weight.data = torch.tensor([[[1/3, 1/3, 1/3]]]) # 均等な重み

        # 【偶数カーネル (2)】：左右対称な重み（平均化）
        conv_even = nn.Conv1d(1, 1, kernel_size=2, padding=1, bias=False)
        conv_even.weight.data = torch.tensor([[[1/2, 1/2]]]) # 均等な重み

        # 伝言ゲーム（20層連続で通過させるシミュレーション）
        out_odd = x.clone()
        out_even = x.clone()

        # ※偶数の場合はサイズ維持のため，右側の余分なパディングを削る処理を手動で挟む
        for _ in range(20):
            out_odd = conv_odd(out_odd)
            out_even = conv_even(out_even)[..., :-1] # サイズ維持のための苦肉の策

        print(f"奇数(K=3)通過後のピーク位置: {torch.argmax(out_odd).item()}") # 出力: 元の位置から1ミリも動いていない！
        print(out_odd.detach().numpy())
        print(out_odd.shape)
        print()

        print(f"偶数(K=2)通過後のピーク位置: {torch.argmax(out_even).item()}") # 出力: 配列の端に追いやられて消滅する（完全にズレる！）
        print(out_even.detach().numpy())
        print(out_even.shape)
        print()

        """
        奇数(K=3)通過後のピーク位置: 5
        [[[0.02570434 0.0504282  0.07277296 0.09079538 0.10229285 0.1053796
           0.09907098 0.08360145 0.06036919 0.03160813]]]
        torch.Size([1, 1, 10])

        偶数(K=2)通過後のピーク位置: 9
        [[[0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
           9.5367432e-07 1.9073486e-05 1.8119812e-04 1.0871887e-03 4.6205521e-03]]]
        torch.Size([1, 1, 10])
        """
        ```

2.  偶数だと，入力のサイズを維持するためのパディングが非対称になってしまい，PyTorchで実装するのが面倒になるため．

### プーリング層
- プーリング層の役割：

    「局所領域内の値の要約」と「ダウンサンプリング」の2つを順に行う事．元信号に高い周波数の成分が含まれる時，不十分なサンプリングレートでダウンサンプリングすると，本来存在しない偽信号が生じるエイリアシングが発生する．エイリアシングの主な対策は，「ナイキスト周波数を超える高周波成分をカットしてしまう事」である．「局所領域内の値の要約」は，入力信号の高周波成分を除去するローパスフィルタを役割を果たし，「ダウンサンプリング」におけるエイリアシングを防いでいる．

- そもそもダウンサンプリングをしなきゃいけない理由：
    - 受容野（Receptive Field）の拡大：ネットワークの深い層にある1つのピクセルが，元の入力画像の「どれくらい広い範囲（森）」を見ているかを示す指標．空間を圧縮せずに畳み込みを続けると局所的な特徴しか捉えられないが，ダウンサンプリングを行うことで，少ない層数で画像全体のコンテキスト（意味）を捉えられるようになる．
    - 計算リソースの節約と高次元化：空間解像度（縦・横）を圧縮し，浮いたメモリ容量を使って特徴量の数（チャンネル数）を増やす．これにより，「物理的な位置情報」から「意味的な特徴情報」への変換を効率的に行うため．

- プーリング方法の選び方：
    - 最大プーリング（Max Pooling）：
        - 局所領域の最大値をとる．畳み込み（画像パッチとカーネルの内積）で得られた「最も強い特徴」だけを次へ伝える．
        - メリット：特徴が数ピクセルずれても出力が変わらない「移動不変性（Translation Invariance）」を獲得できる．画像認識の序盤〜中盤で「どこにあるか」より「何があるか（エッジやパーツの存在）」を際立たせたい時に選ぶ．
        - 余談：カーネルはなぜ「見つけたい特徴」と同じ形に育つのか？

            CNNの学習において，カーネル（重み）は無数のエポックを経た後，探したい画像パターンと全く同じ形に変形していく．正確に言えば，「正例（役に立つパッチ）に対しては内積が最大化され，負例（役立たずのノイズ）に対しては内積が最小化されるように」カーネルの向きが最適化される．なぜそんな魔法のようなことが起きるのか？

            それは，勾配降下法における更新式が，結果的に「入力画像パッチが $\mathbf{x}$ の時，損失の微分が負なら，$\mathbf{x}$ そのものをカーネルに**足す**」「微分が正なら，$\mathbf{x}$ をカーネルから**引く**」というベクトル合成の挙動をするからだ．

            つまり直感的には，ランダムだったカーネル $\mathbf{w}$ は無数の更新を経た後，以下のようなベクトルの線形結合に化ける．
            $$ \mathbf{w} \approx \sum (\text{役に立つ画像パッチ}) - \sum (\text{役立たずの画像パッチ}) $$

            より数学的に厳密に証明しよう．勾配降下法の場合，
            $T$ 回のパラメータ更新を行った後のカーネル $\mathbf{w}^{(T)}$ は，初期状態 $\mathbf{w}^{(0)}$ から微小な更新を $T$ 回分蓄積したものとして，以下のように展開できる．

            $$
            \begin{align*}
                \mathbf{w}^{(T)} &= \mathbf{w}^{(0)} - \sum_{t=1}^{T} \left( \eta \frac{\partial L^{(t)}}{\partial \mathbf{w}^{(t)}} \right) \\
                &= \mathbf{w}^{(0)} - \sum_{t=1}^{T} \left( \eta \frac{\partial L^{(t)}}{\partial y^{(t)}} \frac{\partial y^{(t)}}{\partial \mathbf{w}^{(t)}} \right) \\
                &= \mathbf{w}^{(0)} - \sum_{t=1}^{T} \left( \eta \frac{\partial L^{(t)}}{\partial y^{(t)}} \frac{\partial (\mathbf{w}^{(t)} \cdot \mathbf{x}^{(t)})}{\partial \mathbf{w}^{(t)}} \right) \\
                &= \mathbf{w}^{(0)} - \sum_{t=1}^{T} \left( \eta \frac{\partial L^{(t)}}{\partial y^{(t)}} \mathbf{x}^{(t)} \right)
            \end{align*}
            $$

            【記号の定義とShape】
            - $\mathbf{w}^{(t)}$：ステップ $t$ におけるカーネル（重み）ベクトル．Shape: $(K^2, 1)$
            - $\mathbf{w}^{(0)}$：初期化されたランダムなカーネル．
            - $\eta$：学習率（正のスカラー定数）．
            - $L^{(t)}$：ステップ $t$ における損失（スカラー）．
            - $y^{(t)}$：内積による予測スコア（スカラー）．
            - $\mathbf{x}^{(t)}$：ステップ $t$ で入力された画像パッチのベクトル．Shape: $(K^2, 1)$

            ここで最も重要なのが，スカラー係数となる $\frac{\partial L}{\partial y}$ （予測誤差）の符号である．

            - 「役に立つ（正例）」＝ $\frac{\partial L}{\partial y} < 0$ となる．
                予測スコアが足りていないため，マイナスが掛かって結果的に $\mathbf{x}$ が $\mathbf{w}$ に**足し算**される．
            - 「役立たず（負例）」＝ $\frac{\partial L}{\partial y} > 0$ となる．
                過剰反応してしまっているため，プラスが掛かって結果的に $\mathbf{x}$ が $\mathbf{w}$ から**引き算**される．

            【直感的な例え：モンタージュ写真】
            警察が「犯人のモンタージュ写真」を作るシーンを想像するとわかりやすい．
            - 初期状態：最初は，白紙に適当な線を引いたようなデタラメな似顔絵（$\mathbf{w}^{(0)}$）を持っている．
            - 正例（目撃情報）：「犯人はこんな目をしていました！」という写真 $\mathbf{x}$ が来た．損失関数は「似顔絵にこの特徴を足せ（微分が負）」と指示する．似顔絵にその目を書き足す（ベクトル加算）．
            - 負例（無関係な人）：「この人は無実の一般人です」という写真 $\mathbf{x}$ が来た．損失関数は「似顔絵からこの特徴を引け（微分が正）」と指示する．似顔絵から，一般人と共通する特徴を消しゴムで消す（ベクトル減算）．

            これを何万枚もの写真で繰り返すと，デタラメだった似顔絵は「無実の人々の特徴が削ぎ落とされ，犯人の特徴だけが塗り重ねられた，完璧なモンタージュ写真」に化ける．これが，ニューラルネットワーク（ここでは畳み込み層）が特徴を「学習」するメカニズムの正体だ．

    - 平均プーリング（Average Pooling）：
        - 局所領域の平均（期待値）をとる．
        - メリット：情報を尖らせず，全体的な傾向を要約する．現代では，ネットワークの最終層で空間情報を一気に1ピクセルに潰す「Global Average Pooling (GAP)」として使われ，「画像全体としてその特徴がどれくらい含まれるか」の期待値を算出するために選ばれる．

    - 【発展】ストライド畳み込み（Strided Convolution）：
        - 現代の最先端モデルにおける第3の選択肢．プーリング層を使わず，畳み込み層のストライド$s$（歩幅）を$s>1$に設定することで空間を圧縮する．
        - メリット：プーリングのような「固定の圧縮ルール」ではなく，「どう圧縮すれば損失が最小になるか」というダウンサンプリングの方法自体をネットワークに学習させることができるため，表現力が高い．

## テンソルの第0次元（一般にバッチサイズ）を取得する方法
1.  `len(inputs)`：Python標準
2.  `inputs.shape[0]`：NumPy由来
3.  `inputs.size(0)`：PyTorchネイティブ（NumPyとの区別のためにコレを使おうね．）
