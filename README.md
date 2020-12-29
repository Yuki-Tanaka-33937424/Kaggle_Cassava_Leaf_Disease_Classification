![0A63BFAC-7C55-49F6-A6BA-111F1DF9F6AD_4_5005_c](https://user-images.githubusercontent.com/70050531/102084360-65bf1d80-3e58-11eb-82de-647a2e845ed7.jpeg)
# Kaggle_Cassava_Leaf_Disease_Classification<br>
Cassava Leaf Disease Classification コンペのリポジトリ  
タスク管理ボード: [リンク](https://github.com/Yuki-Tanaka-33937424/Kaggle_Cassava_Leaf_Disease_Classification/projects/2) <- 大事!!!  

## 有用リンク
[optimizer](https://github.com/jettify/pytorch-optimizer)  
[bi_tempered_loss の pytorch実装](https://github.com/mlpanda/bi-tempered-loss-pytorch)  

## 方針(暫定)
- Kaggle上で全てを完結させる
- Version名は常にVersion○に統一して、READMEに簡単に内容を書き込む
- 詳しい実験などを行った際には、その旨をREADMEに書き込み、詳しい実験結果はそのNoteBookの一番上にoverviewとして書き込む

## Basics
### Overview(DeepL)
アフリカ第二の炭水化物供給源であるキャッサバは、過酷な条件に耐えることができるため、零細農家によって栽培されている重要な食糧安全保障作物である。サハラ以南のアフリカでは、少なくとも80％の家庭用農場がこのでんぷん質の根を栽培していますが、ウイルス性の病気が収量不良の主な原因となっています。データサイエンスの助けを借りれば、一般的な病気を特定して治療することが可能になるかもしれません。

既存の病気の検出方法では、農家は政府が出資する農業専門家の協力を得て、植物を目視で検査して診断しなければならない。この方法は、労働集約的で、供給量が少なく、コストがかかるという問題があります。加えて、アフリカの農家は低帯域幅のモバイル品質のカメラしか利用できない可能性があるため、農家のための効果的なソリューションは、大きな制約の下でうまく機能しなければなりません。

このコンテストでは、ウガンダの定期調査で収集された21,367枚のラベル付き画像のデータセットを紹介します。ほとんどの画像は、農家が庭の写真を撮影しているところからクラウドソーシングされ、カンパラのマケレレレ大学のAI研究室と共同で国立作物資源研究所（NaCRRI）の専門家が注釈を付けたものだ。これは、農家が実際の生活の中で診断する必要があるものを最も現実的に表現したフォーマットです。

あなたの課題は、キャッサバの画像を4つの病気のカテゴリに分類するか、健康な葉を示す5つ目のカテゴリに分類することです。あなたの助けがあれば、農家は病気にかかった植物を迅速に特定することができ、取り返しのつかないダメージを与える前に作物を救うことができるかもしれません。

### data(DeepL)
比較的安価なカメラの写真を使って、キャッサバの植物の問題点を特定できますか？このコンテストでは、多くのアフリカ諸国の食糧供給に重大な被害をもたらすいくつかの病気を区別することに挑戦します。いくつかのケースでは、主な救済策は、それ以上の広がりを防ぐために感染した植物を燃やすことですが、これは、迅速な自動化されたターンアラウンドを農家にとって非常に有用なものにすることができます。

ファイル  
train/test]_images: 画像ファイルです。テスト画像のフルセットは、スコアリングのために提出されたときにのみノートブックに表示されます。テストセットには約15,000枚の画像が含まれています。  

**train.csv**  

image_id: 画像ファイル名。  

label: 病気のIDコード。  

**sample_submission.csv** 開示されたテストセットの内容に応じて適切にフォーマットされたサンプル提出物。  

image_id: 画像ファイル名  

label: 病気の予測されるIDコード。  

train/test]_tfrecords: tfrecord形式の画像ファイル。  

label_num_to_disease_map.json。各疾患コードと実際の疾患名とのマッピング。  

## Log

### 20201214
- join!  
- 管理方法についてひたすら悩む。  
- 明日はコードを書かなければ...  
- GitHubの使い方を掴むためにひたすら動かしてみた。issue管理うまくいきそう。

### 20201215  
- とりあえずEDA(cassava_EDA)  
- train画像は21397枚で、test画像はsubmit時に裏で与えられるみたい。  
- ベースラインのモデルに悩む。とりあえずEfficient Net B0でいいかな。軽いし。  

### 20201216  
- MoAメダリストの反省会を聞いてモチベが上がった。全部試す精神は本当に大切。  
- nb001(ver1)。とりあえずDatasetまで作った。早めにサブしたい。  

### 20201217
- nb001  
  - Dataloaderまで作った。  
  - モデルの定義の仕方がわからず、結局次からベースラインを見ることにした。 
  
### 20201218  
- nb001  
  - EDAの後に全ての画像のラベルを揃えてsubしてみた。trainとpublicの分布はほぼ同じとみていい。privateもほぼ同じだろう。  
  
### 20201219  
- nb 001  
  - LBの結果を追記。  
- nb 003  
  - ver2  
    - Y.Nakamaさんのベースラインを写経。Resnext50_32x4dのpretrain有り。  
    - CV=0.87321, LB=0.880  
  - ver3  
    - optimizerをAdamからAdaBeliefに変えてみる。  
    - CV=0.87321, LB=0.879  ほぼ変わらん。なぜ？？？ <- Schedulerがlrをいい感じに変えているためにoptimizerの差が打ち消されている可能性が大きい。  
- nb 005   
  - ver1  
    - EfficientNet-B0を実装。実行時間が30分ぐらい減った!!  
    - AdaBliefはそのまま。  
    - CV=0.85288, LB=0.861  
    - モデルは軽い方がいいため、多少精度は落ちるかもしれないがEfficientNet_B0で実験をしていく。  
CVよりLBスコアの方が高いのはなぜ？若干違和感がある。 <- CVのラベリングがおかしい(Noisyな)可能性がある。

### 20201220  
- nb_005  
  - ver2  
    - n_foldを4に落とした。  
    - CV=0.85129, LB=0.855  
    - LBが予想より低い。これは誤差と捉えていいのかは検討の余地がありそう。  
  - ver3  
    - batch_sizeを大きくすれば大きな時間短縮につながるのでは？  
    - どうやら、batch_sizeを大きくするならば学習率も大きくしないといけないらしい。  
    - [ここ](https://qiita.com/koshian2/items/8d8f0197aab1779e096b)では、「バッチサイズと学習率を同様に大きくすれば高速化できる」とある。  
    - というわけで、とりあえずbatch_size: 32->64, lr: 1e-4->2e-4としてみる。  
    - CV=0.85507, LB=0.862  
    - 精度は若干上がってるけど、5分しか実行時間が短縮されてない。なぜ？？？ <= そりゃ倍速にはならんだろ...  
  - ver5(ver4は失敗)
    - batch_size: 64->128, lr: 2e-4->4e-4としてみる。  
    - さらに実行時間が4分程度減った。8100秒ぐらい。  
    - 128がメモリの限界らしい。256にしたらエラーになった。  
    - CV=0.85657, LB=0.862だった。伸び幅は小さかった...  
  - ver6  
    - epoch３までは最終層以外凍結し、lr=4e-3で学習させ、epoch4から全て解凍し、lr=4e-4で学習させてみる。  
    - epoch3がほぼ無駄になっていたので止めた  
  - ver7  
    - 凍結するのをepoch2までにした。  

### 20201221  
- nb_005  
  - ver7  
    - epoch2はほとんど学習していなかった。最初の学習率を10倍にしたからか？  
    - CV=0.85619, LB=0.867  
    - CVはほとんど変わっていないが、LBスコアは改善された。いいとは思うんだけど理由がよくわからん。特徴量抽出器がいい感じに残ったからか？  
    - 最初のlrも4e-4で揃えていいかもしれない。ver5はepoch8~は過学習してるみたいだし。  
    - あと、あんまり関係ないけどGPUがもう切れそう。早い...  
  - ver8  
    - パラメータの解凍をepoch2に早めた。
    - なせか実行時間が6分ぐらい減った。なぜ？？？
    - CV=0.85718, LB=0.868  
    - スコアがよくなってる！！  
- [カエルさんのDiscussion](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/202017)よると、データセットのラベルがNoisyらしい。外れ値に対するpenaltyを和らげた**tempered loss**というものが有効らしい。それについてのGoogleの論文->[リンク](https://arxiv.org/abs/1906.03361)
- GPUが明らかに足りてないので、とりあえずcolabに移行してみて、ダメだったらGCPを使いたい。    

### 20201222  
- Google Colabに移行しようとしたら、Kaggle APIのversionが古いらしくうまくいかなかった。[ここ](https://qiita.com/RIRIh/items/6c8495a190e3c978a48f)を参考にして強制的にupgradeしたら一応解決した。  
- google colabの設定に死ぬほど時間がかかった!!!!!  
- google colabだとbatch_sizeが64で限界っぽい。  
- google colabだとフルで実験を回せない...  やはりGCPがいいか。<br>
- 画像コンペでの心得をいくつか聞いたのでメモ。<br>
  - モデル選択が非常に重要。確かにそうなので、他のモデルに移行する時期も決めた方が良さげ。<br>
  - 最初のfoldだけで実験するといい。だいぶ時間の節約になりそう。<br>
- nb 005  
  - ver8.5  
    - batch_sizeを64に落として学習してみた。fold1でCV=0.86019。これはver7とほぼ同じ。  
    - ここまでの実験では、batch_sizeは32よりは64がいいが、128にしても大差はなさそう、といった感じ。  

### 20201227  
- GCPへの移行とE資格対策にかなりの時間を費やしてしまった。  
- nb 005  
  - ver 9
    - bi_tempered_lossをとりあえず実装し、動くことを確認。t1とt2が1の時にうまく動作しない...  
- nb 007 (E資格のプロダクト課題) nb 005 ver8 のfork　
  - ver 3 (ver1とver2は失敗)  
    - optimizerをAdam, AdaBound, RAdam, Adabeliefにして、fold1のみで実験  
    - まあ恐らく結果はほとんど変わらない。schedulerがいい働きをしてoptimizer間の違いを吸収してしまうはず。  
  - ver 4<br>
    - schedulerをNoneにしてみた。
    - 結果がver3と全く同じになってしまった。なぜ？？？
    - 全結合層以外を凍結する操作を入れた時から、optimizerのインスタンスを新しく生成しているせいで、schedulerの管理下から外れていることが発覚。ということは、nb_007のver7から全てそうなってしまっている。<br>
    - じゃあなんで実行時間が伸びたんだよ...<br>
  - ver 5<br>
    - ver3において、schedulerをしっかり動かした。<br>
    - AdamがCV=0.86449(fold1)でトップ。どうやらschedulerを使うと逆転するらしい。<br>
    - 今後はAdamに戻そうと思う。<br>

### 20201228<br>
- nb 005<br>
  - ver 10<br>
    - ver8からの更新。(bi tempered lossは一旦消している)<br>
    - E資格のプロダクト課題から、schedulerを使う時にはAdamが一番性能が良いことがわかっているため、schedulerをオンにして、optimizerをAdamに変更する。<br>
    - CV=0.86449(fold1)。かなりスコアがよくなった。<br>
  - ver 11<br>
    - [公開Kernel](https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-train-amp-aug)のAugmentationが良い感じだったので真似してみた。<br>
    - CV=0.86523(fold1)。スコアが改善された。<br>
  - ver 13(12は失敗)<br>
    - 画像のsizeを512にしようとしたらメモリが足りなくなってしまった。その場合はbatchsizeを32にしないといけないらしい。そこで、まずはsizeを256のままbatch_sizeを32に落としてみる。<br>
    - CV=0.86224。当たり前だけど悪化した。<br>
  - ver 14
    - sizeを512に変更。
    - CV=0.88692。こんなに上がるかよおい。
    
- cassavaコンペはこれまで何回かにわたり実施されていたらしい。そこで用いられたデータは使って良いそうだ。[このDiscussion](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/198131)に前回のcassavaコンペのリンクと1位から3位までのsolutionが載ってる。
- Snapmixという手法が有用らしい。[Discussionのリンク](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/204631)と[GitHubのリンク](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/198131)<br>
  
### 20201229<br>
- nb 005<br>
  - ver 16(15は失敗)<br>
    - bi_tempered_loss(今後はbi_lossと呼ぶ)を実装するにあたり実験の数が増えるので、一旦ver11に戻した。<br>
    - bi_lossを実装するにあたり、labelをone-hotにした。validの時だけ戻してる。<br>
    - num_iterの説明は論文のAppendixAにメモしておいた。pytorch実装ではt2を1とするとdimensionがおかしくなるみたい。<br>
    - t1=1, t2=1.0001, label_smoothing=0で実験(ver11とほぼ同じ結果になるか試したい)。<br>
    - CV=0.86467(fold1)で、lossはほぼver11と変わらないので、しっかり再現できてる。<br>
  - ver 17<br>
    - t1=0.9に変更。<br>
    - CV=0.86841(fold1)。スコアが上がっている。<br>
    - ただし、train_lossの減りがさらに早くなっている。計算方法が違うとは思えないため、学習が早くなっていると言えそう。<br>
  - ver 18<br>
    - t1を0.5, 0.6, 0.7, 0.8にして実験。<br>
    - t1=0.5: CV=0.86710, 0.6: CV=0.86411. 0.7: 0.87196, 0.8:0.86224<br>
    - CVだけ見るとt1=0.7が一番いいが、lossの計算方法が違うせいで過学習してるかどうかの判定かできない。<br>
  - ver 19<br>
    - よく考えたら、t2も同時に変えないといけないことがわかったので、セルフgrid searchをして見ることにした。lossも、表示は以前のものに直した。(backwardはbi_loss)<br>
- [カエルさんのディスカッション](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/202017)で、gambler's lossというものが紹介されていた。Noisyなデータの中でもしっかり学習できるらしい。[paper](https://arxiv.org/pdf/2002.06541v1.pdf)はこれ。pytorch実装は一応[ここ](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/205424)で議題には上がっているが、誰も反応はしてないみたい。
