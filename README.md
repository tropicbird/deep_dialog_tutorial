# このレポジトリについて
このレポジトリでは、TransformerをTensorFlow 1系を用いて手作りされた [@halhorn](https://github.com/halhorn) さんのコードに少し手を加えて、TensorFlow 2系で動作するコードに書き換えた [コード](https://github.com/tropicbird/deep_dialog_tutorial/tree/master/deepdialog/transformer) を公開しています。なお、あくまで2系でもコードが動くようにしただけですので、必ずしも2系で推奨されている記述形式に従っている訳ではありません。 [@halhorn](https://github.com/halhorn) さんの元コードは以下のリンクでご参照下さい。Transformerを学習する上で非常に重宝しました。

## 元コード
- [GitHub (※このレポジトリのフォーク元)](https://github.com/halhorn/deep_dialog_tutorial)
- [Qiita記事](https://qiita.com/halhorn/items/c91497522be27bde17ce)

## 動作確認済の環境
- Python 3.7
- tensorflow-gpu 2.1.0

<!-- 
# Deep Dialog Tutorial
会話モデルネタでいろいろ追加していくリポジトリ

- [Transformer](https://github.com/halhorn/deep_dialog_tutorial/tree/master/deepdialog/transformer)
- [RNNLM](https://github.com/halhorn/deep_dialog_tutorial/tree/master/deepdialog/rnnlm)

# Install
python は python3 を想定してます。

```zsh
git clone git@github.com:halhorn/deep_dialog_tutorial.git
cd deep_dialog_tutorial
pip install pipenv
pipenv install

# 起動
pipenv run jupyter lab
```

# Transformer
[コード](https://github.com/halhorn/deep_dialog_tutorial/tree/master/deepdialog/transformer)
[作って理解する Transformer / Attention](https://qiita.com/halhorn/private/c91497522be27bde17ce)

# RNNLM
rnnlm.ipynb

RNN の言語モデル。
たくさんの文章集合から、それっぽい文章を生成するモデルです。

- 学習時：上から順に Train のセクションまで実行してください
- 生成時：Train 以外のそれより上と、 Restore, Generate を実行してください。
    - Restore 時のモデルのパスは適宜変えてください。

# Testing
```py3
./test/run
# or
./test/run deepdialog/transformer/transformer.py
```
--> 