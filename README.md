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
    - 実験結果は以下の通り。
    - t1 | t2 | CV | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: | :-----: 
      0.7 | 1.2 | 0.86766 | 0.3439 | 0.4727
      0.7 | 1.5 | 0.86561 | 0.5245 | 0.7046
      0.8 | 1.2 | **0.86953** | 0.3374 | 0.4707
      0.8 | 1.5 | 0.86299 | 0.5297 | 0.7387
      0.9 | 1.2 | 0.86561 | 0.3284 | 0.4813
      0.9 | 1.5 | **0.87028** | 0.5139 | 0.7602 <br>
    - t2がlossに強く影響してるようだ。t1=0.9, t2=1.5の時がいいようなので、今後は暫定的にこの値を使用する。<br>
      
- [カエルさんのディスカッション](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/202017)で、gambler's lossというものが紹介されていた。Noisyなデータの中でもしっかり学習できるらしい。[paper](https://arxiv.org/pdf/2002.06541v1.pdf)はこれ。pytorch実装は一応[ここ](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/205424)で議題には上がっているが、誰も反応はしてないみたい。<br>

### 20201230<br>
- nb 005<br>
  - ver 20<br>
    - label smoothingを導入する実験。実験結果は以下の通り。(t1=0.9, t2=1.5) <br>
    - smooth | CV | train_loss | valid_loss
      :-----: | :-----: | :-----: | :-----: 
      1e-5 | 0.86467 | 0.5129 | 0.7503
      1e-4 | 0.86355 | 0.5168 | 0.7533
      1e-3 | **0.86897** | 0.5176 | 0.7526
      1e-2 | **0.86822** | 0.5066 | 0.7063
      1e-1 | **0.86953** | 0.3667 | 0.5263 <br>
    - 1e-1の結果が一番いいが、lossが異常に低いのが気になる。1e-3以下に比べれば1e-2もかなり低いので、正常値と見ていいか？？正直、CVの0.0006の違いなんて誤差だろう。<br>
    - LBスコアも見た方がいいと思うので、上位三つに関しては再実験する。<br>
    - 実験の再現性がないことに気づいた。恐らく、train_loopの中でseedを固定し忘れたこと、またはtorch.backends.cudnn.benchmark = Falseを忘れていたことが原因だろう。<br>
  - ver 22 ~ 25<br>
    - 再現性を確保してから、sub込みでsmoothingをもう一度実験。結果は次の通り。
      smooth | CV | LB | train_loss | valid_loss
      :-----: | :-----: | :-----: | :-----: | :-----:
      0 | 0.86972 | 0.859 | 0.5706 | 0.7400
      1e-3 | 0.86692 | 0.860 | 0.5666 | 0.7686
      1e-2 | **0.87028** | **0.868** | 0.4946 | 0.6984
      1e-1 | 0.86972 | 0.864 | 0.3887 | 0.5093 <br>
    - 結果から、1e-2が一番良さそうなので、とりあえずこれでいく。<br>
- SAMというoptimizerが強いらしい。[解説記事](https://qiita.com/omiita/items/f24e4f06ae89115d248e)、[原論文](https://arxiv.org/abs/2010.01412)、[pytorch実装](https://github.com/davda54/sam)。<br>
  - ver 26<br>
    - 最新のoptimizer,SAMを入れてみる。base_optimizerはAdam。<br>
    - CV | LB | train_loss | valid_loss
      :-----: | :-----: | :-----: | :-----:
      0.87234 | 0.867 | 0.4754 | 0.6090 <br>
    - LBはほぼ変わらないけどCVはよくなっているので、SGDも試してみてから採用を考える。<br>
  - ver 28(27は失敗)<br>
    - SAMのbase_optimizerをSGDにした。<br>
    - CV=0.69813。明らかにダメなので却下する。Adamの場合も、LBが変わらないのに学習時間が30%ほど増えているため、Adamのままにしておく。<br>

### 20210102<br>
- nb 008(EfficientNet_B4)<br>
  - ver 1<br>
    - EfficientNetB4にモデルを変更。<br>
    - 画像サイズを512にしたところ、batchsizeが8じゃないとメモリに乗り切らないため、accumulation_stepを4にして、実質batch_sizeを32のままに保つことにした。<br>
    - num_foldを5にしたが、時間がかかりすぎるため、fold0のみを訓練する。<br>
    - CV | LB | train_loss | valid_loss
      :-----: | :-----: | :-----: | :-----:
      0.88411 | 0.886 | 0.6157 | 0.6243 <br>
    - 期待よりスコアが低い。なぜ。<br>
    
- nb 005<br>
  - ver 29<br>
    - nb 008との比較のため、ver23あたりの、SAMをいれる前の状態でnum_foldを5にした。<br>
    - CV | LB | train_loss | valid_loss
      :-----: | :-----: | :-----: | :-----:
      0.86565 | 0.866 | 0.5139 | 0.7346 <br>
    - LBはほぼ変わらないので、CVの悪化は誤差とみなす。<br>
    - Logをよく見ると、valid_lossはepoch7が一番低かった。valid_lossでEarlystoppingをかけたら、LBスコアが改善する可能性があると考えられる。<br>
  - ver 30<br>
    - 画像サイズを512にして、batchsizeを32に下げ、さらにSAMを入れた。<br>
    - CV | LB | train_loss | valid_loss
      :-----: | :-----: | :-----: | :-----:
      0.88949 | 0.882 | 0.4715 | 0.5076 <br>
    - LBが振るわなくてショック...<br>
- AdamよりSAMの方がlossの下がり方が圧倒的に安定していた。なんとかしてgradient_accumulationとSAMを組み合わせたい。<br>

### 20210103<br>
- nb 005<br>
  - ver 31<br>
    - ver26から、num_foldを5にして再実験。<br>
    - CV | LB | train_loss | valid_loss
      :-----: | :-----: | :-----: | :-----:
      0.87103 | 0.874 | 0.4764 | 0.6150 <br>
    - LBが一気によくなって少し疑問が残るが、変化はnum_foldだけだから偶然とみなしておく。<br>
    - さらに、TTAを入れる実験を行った。<br>
    - tta | LB 
      :-----: | :-----:
      なし | 0.874 
      cropのみ | 0.881
      crop&flip | 0.883 <br>
    - すごい上がり方だけど本当かこれ。TTAはcropもflipも入れた方がいいみたい。(時系列的には、nb008 ver2の実験よりも後に行っている。)<br>
  - ver 32<br>
    - ver31からaugmentationを少し削った。<br>
    - CV | LB | train_loss | valid_loss
      :-----: | :-----: | :-----: | :-----:
      0.86893 | 0.871 | 0.4861 | 0.6049 <br>
    - やはり、augmentationは増やしたままがいいらしい。<br>
    
- nb 008<br>
  - ver 2<br>
    - batch_sizeを8にした上で、accumulation_stepを1に戻してSAMを入れた。<br>
    - CV | LB | train_loss | valid_loss
      :-----: | :-----: | :-----: | :-----:
      0.89159 | 0.886 | 0.5131 | 0.4894 <br>
    - CVはよくなってるがLBは変わらない。Noiseに引っ張られている可能性が高い。改善策としてはEarly stoppingや、画像の解像度を原論文通りに少し下げることが考えられる。<br>
    - TTAは本質的な改善には繋がらない気がするが、手っ取り早いのですぐ実装する。<br>
    - tta | LB 
      :-----: | :-----:
      なし | 0.886
      cropのみ | 0.885
      crop&flip | 0.887 <br>
    - 今回はclop&flipが一番いい結果を残しているが、[このdiscussion](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/206489)では、flipは効かないという意見が出ている。trainのaugmentationの複雑さと関係している可能性もあるため、trainのaugmentationを緩くすれば、結果が逆転する可能性もある。<br>

- [このdiscussion](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/206220)では、resizeよりcenter clopの方がいい結果が出てるというコメントがある。今のところRandomResizedcropを使っているため、試す価値はありそう。<br>

### 20210104<br>
- nb005<br>
  - ver 33<br>
    - epochごとに全てモデルを保存しておいて後からアンサンブルしようとしたが、間違えてepoch10のみを保存してしまった。<br>
  - ver 34<br>
    - epoch6, 7, 8, 9でAveragingしてみた。アンサンブルなしとの比較は次の通り。<br>
    - Averaging | LB
      :-----: | :-----: 
      なし(ver31) | 0.883
      あり(ver34) | 0.888<br>
    - かなりスコアが上がってる。Early stopiingの効果とアンサンブルの効果がどっちも反映されてるっぽい。<br>

- discussionで議論されているlossがほぼ全て[このNotebook](https://www.kaggle.com/piantic/train-cassava-starter-using-various-loss-funcs)で実装されてる。すごい。<br>

### 20210105<br>
- [このDiscussion](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/208239)でSymetric Cross Entropy LossというのがBi-tempered-lossよりいいスコアを出しているという報告がされている。<br>

- EfficientNetは、画像の解像度も含めて最適化されているから、画像サイズをそれに合わせた方がいいかもしれない。<br>
- [このディスカッション](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/205491)でCutMixについての言及がある。有効っぽい。pytorchの実装は[ここ](https://github.com/hysts/pytorch_cutmix)にある。

- nb005<br>
  - ver34(続き)<br>
    - ttaの回数を変えてみた。比較の結果は次の通り。<br>
    - tta | LB 
      :-----: | :-----: 
      3回 | 0.888
      5回 | 0.889
      8回 | 0.887 <br>
    - 多ければいいというわけではないらしい。今後は5回で固定する。<br>
  - ver35<br>
    - 様々なLossが実装されているNotebookからloss functionをコピペした。BiTemperedLossにして、これまでと同じ結果が正しく再現できるかをチェックする。<br>
    - 完全に再現できていた！<br>
  - ver37(ver36失敗)<br>
    - LossをSymmetricCrossEntropyLossにして実験

- nb010(create_model_EfficientNet_B0ns)<br>
  - ver1<br>
    - nb005_ver34から、モデルをEfficientNetB0(Noisy Student)に変更。<br>
    - CV | LB | train_loss | valid_loss
      :-----: | :-----: | :-----: | :-----:
      0.86799 | 0.886 | 0.4871 | 0.6274 <br>
    - 予想に反して、Noisy Studentの方がスコアが低い。誤差の範囲とも言えるが...<br>

- nb012(inference_model_EfficientNet_B0&B0ns)<br>
  - ver1<br>
    - nb005_ver34とnb010_ver1のアンサンブル。結果は次の通り。<br<
    - model | LB 
      :-----: | :-----: 
      B0 only | 0.889
      B0ns only | 0.886
      B0 & B0ns | 0.890 <br>
    - 改善はしたが、思ったより伸び幅が小さい。epochごとのモデルを混ぜたり、TTAをしたりしているうちに多様性による伸び幅を使い尽くしてしまったのか...？<br>

### 20210106<br>
- AutoAugmentationが[このディスカッション](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/208887)で議論されている。EfficientNetのSOTAのスコアにも貢献してるらしい。試す価値がありそう。<br>

- nb005<br>
  - ver37~39<br>
    - Loss Functionを変えて実験。結果は次の通り。<br>
    - Loss fn (ver) | CV | LB 
      :----- | :-----: | :-----: 
      BiTemperedLogisticLoss (ver35) | 0.87103 | 0.889
      SymmetricCrossEntropyLoss (ver37) | 0.86729 | 0.886
      FocalLoss (ver38) | 0.86706 | 0.883
      FocalCosineLoss (ver39) | 0.87313 | 0.887 <br>
    - Bi-Tempered-Lossが一番いいことがわかったので、今後もそれを用いる。<br>
      
### 20210107<br>
  - nb005<br>
    - ver40~<br>
      - 画像サイズを変える実験を行った。結果は以下の通り。<br> 
      - 画像サイズ(ver) | CV | LB | train_loss | valid_loss | time(s)
        :-----: | :-----: | :-----: | :-----: | :-----: | :-----:
        224(ver40) | 0.86168 | 0.881 | 0.4712 | 0.6547 | 2244.0
        240(ver41) | 0.86776 | 0.888 | 0.5158 | 0.6307 | 2372.8
        256(ver35) | 0.87103 | 0.889 | 0.4764 | 0.6150 | 2669.4
        300(ver44) | 0.87266 | 0.886 | 0.4838 | 0.5771 | 2848.3
        380(ver45) | 0.88154 | 0.892 | 0.4913 | 0.5183 | 4406.8
        456(ver46) | 0.88832 | 0.892 | 0.4570 | 0.5153 | 5580.8
        512(ver47) | 0.88949 | 0.893 | 0.4715 | 0.5076 | 6375.2 <br>
      - 結果を見るに、380が一番コスパがいいように見える。EfficientnetB0の元々のサイズは224だが、今後は380で固定する。<br>
      - 途中で気づいたが、推論の時には画像サイズを少し大きく取るとスコアが上がるらしい。その実験もしようと思う。<br>
- nb008<br>
  - ver3<br>
    - nb005とLossなどを揃えた。(quick save)<br>
- nb010<br>
  - ver3<br>
    - nb005とLossなどを揃えた。(quick save)<br>

### 20210108<br>
- PytorchでもTPUが使えるらしい。TPUも週30時間使えるので、実装できれば計算資料の制限がかなり緩和されそう。腎コンペの[このNotebook](https://www.kaggle.com/joshi98kishan/foldtraining-pytorch-tpu-8-cores)にわかりやすくまとめられている。<br>
- PANDAコンペでは、trainデータのみがNoisyだったらしい。1thの解法は、学習時にdenoiseしてしまってから再学習するというものだった。今回はtestデータもNoisyであるため機能するかは不透明だが、試す価値はあるのかもしれない。<br>
- メラノーマコンペでは、CVが安定しなかったらしいが、外部データも含めたCVと含めないCVの両方をモニターすることでCVが安定したらしい。cassavaにおいても、外部データ(過去コンペのデータ)を用いるときは今回のtrainデータだけのCVスコアも出した方が良さそう。<br>
- GPUの節約とモデルの多様性を両立させるために、例えば、B0:fold0, B1:fold1, ... ,B4:fold4, B0ns:fold4, B1ns:fold3, ..., B4ns:fold0みたいなことをしたい。複数のepochを用いてttaも行えばsingle foldでもモデルの潜在能力をかなり引き出せるため、モデルごとにfoldを跨げばそれぞれはsingle foldの学習のみで済むためGPUの節約にもなり、学習データを全て使えて、subの時間も抑えられて色々都合がいいように思える。 
- nb005<br>
  - ver45<br>
    - ver34でttaは5回が最適という結論が出たが、これはepochを複数にわたり用いる前の話なので、もう一度ttaをに下げる実験を行う。<br>
    - tta | LB
      :-----: | :-----: 
      3回 | 0.890
      5回 | 0.892 <br>
    - 5回がベターであることは、epochを複数用いても変わらなかった。今後も5回のままにする。<br>
    - 前日に気づいた、「推論時のみ画像サイズを少し上げるとスコアが改善する可能性がある」という案を実験する。ttaは5回、trainの画像サイズは380で固定する。<br>
    - 画像サイズ(inference) | LB 
      :-----: | :-----:
      380(そのまま) | 0.892
      400 | 0.888
      456 | 0.888 <br>
    - スコアが悪化した。改善する可能性としては、(1)元の画像サイズが小さい(224->256など), (2)変化が小さい(380->385)などが挙げられるが、これ以上追求しても無駄な気がするので、画像サイズを揃えるオーソドックスな方法を取る。<br>
  - ver48<br>
    - 2019年のデータも入れて実験。(パラメータはver45のまま。)<br>
    - data | CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: | :-----:
      2020 only | 0.88154 | 0.892 | 0.4913 | 0.5183 
      2020&2019 | 0.87946 | **0.893** | 0.4843 | 0.5694 <br>
    - データの中身が変わってるので、CVは単純には比較はできない。次は揃えたい。<br>
    
### 20210109<br>
    
- nb010<br>
  - ver3~4<br>
    - B0とB0nsをアンサンブルするために作った(nb005_ver48と同じ)。ver3はfold0のみで、ver4はfold1のみで学習を行った。<br>
    - 仮説では、B0_fold0&B0ns_fold0よりB0_fold0&B0_fold1の方がスコアが高くなる。(使ってるデータが若干ずれるためモデル同士の相関が低くなるから。)<br>
    - それぞれのスコアは次の通り。<br>
    - fold | CV | LB | train_loss | valid_loss | LB(with B0_fold0)
      :-----: | :-----: | :-----: | :-----: | :-----: | :-----: 
      0(ver3) | 0.88908 | 0.889 | 0.5177 | 0.5551 | 0.890 
      1(ver4) | 0.88459 | 0.889 | 0.5050 | 0.5442 | 0.892 <br> 
    - 仮説通り後者の方がスコアが高いが、B0のスコア(0.893)を下回っている。原因としては、そもそものB0nsのスコアが低いことが挙げられる。しかしそもそもなぜB0nsの方がスコアが低いかがいまいちわからないのが怖い。他のサイズだとnsの方が高い場合もあるので、discussionの通りlabelのnoiseに引っ張られている可能性が高い。
- nb013(create_model_EfficietNet_B1ns)<br>
  - ver3<br>
    - EfficientNetB1nsを画像サイズ410にして学習させる(ver1は380だが、使わない。ver2はただのミス)。
    - スコアは次の通り。
    - CV | LB | train_loss | valid_loss | time(s) 
      :-----: | :-----: | :-----: | :-----: |:-----:
      0.89066 | 0.896 | 0.4565 | 0.5380 | 7682.9 <br>
      
### 20210110<br>
- nb015(create_model_EfficientNet_B1)<br>
  - ver1<br>
    - 条件等はnb013_ver3と全く同じ。modelがNoisy studentではなくなっただけ。<br>
    - CV | LB | train_loss | valid_loss | time(s) 
      :-----: | :-----: | :-----: | :-----: | :-----: 
      0.88819 | 0.893 | 0.4469 | 0.5338 | 8063.4 <br>
    
- nb017(create_model_EfficientNet_B2)<br>
  - ver1<br>
    - nb015_ver1のfork。画像サイズを440にした。メモリに乗り切らなかったため、batch_sizeを16に下げて、学習率も半分ずつにした。<br>
    - CV | LB | train_loss | valid_loss | time(s) 
      :-----: | :-----: | :-----: | :-----: | :-----: 
      0.88269 | 0.893 | 0.4883 | 0.5276 | 10213.2 <br>
      
- nb019(create_model_EfficientNet_B2ns)<br>
  - ver1<br>
    - nb017_ver1のfork。モデルがNoisy studentになっている。<br>
    - CV | LB | train_loss | valid_loss | time(s) 
      :-----: | :-----: | :-----: | :-----: | :-----: 
      0.88459 | 0.895 | 0.5168 | 0.5032 | 9888.3 <br>
- nb021(create_model_EfficientNet_B3)<br>
  - ver1<br>
    - nb019_ver1のfork。画像サイズが470になっている。<br>
    - CV | LB | train_loss | valid_loss | time(s) 
      :-----: | :-----: | :-----: | :-----: | :-----: 
      0.88819 | 0.892 | 0.4585 | 0.5088 | 13689.6 <br>
- nb023(create_model_EfficientNet_B3ns)<br>
  - ver1<br>
    - nb021_ver1のfork。モデル以外は変更なし。
    - CV | LB | train_loss | valid_loss | time(s) 
      :-----: | :-----: | :-----: | :-----: | :-----: 
      0.89427 | 0.899 | 0.4740 | 0.4891 | 13755.4 <br>
- nb008(create_moodel_EfficientNet_B4)<br>
  - ver4<br>
    - nb023_ver1のfork。画像サイズを500に変えた。メモリに乗り切らなかったため、batch_sizeを8に変えて、学習率を半分に落とした。<br>
    - CV | LB | train_loss | valid_loss | time(s) 
      :-----: | :-----: | :-----: | :-----: | :-----: 
      0.88952 | 0.893 | 0.5055 | 0.4547 | 22356.8 <br>
    - valid_lossはepoch10が一番よかった。他のモデルにもepoch10が一番いい場合が見られるので、混ぜるepochは[6, 7, 8, 9]ではなく[7, 8, 9, 10]でもいいかもしれない。
- nb025(create_model_EfficientNet_B4ns)<br>
  - ver1<br>
    - nb008_ver4のfork。モデル以外は変更なし。<br>
    - CV | LB | train_loss | valid_loss | time(s) 
      :-----: | :-----: | :-----: | :-----: | :-----: 
      0.88933 | 0.897 | 0.5233 | 0.5013 | 22328.9 <br>
    - nb008_ver4と同じように、epoch10の時が一番valid_lossは小さかった。
    - trainではbatch_sizeは8でないとダメだったが、inferenceでは16でも大丈夫だった。もしかしたら32でも大丈夫かもしれない。<br>
    - epochを[7, 8, 9, 10]にしてもLBスコアは同じだった。batch_sizeは32でも大丈夫だった。(ver1_2)<br>
    - batch_sizeを64にしても大丈夫だった。それどころかいきなりLBスコアが0.900までのびた。これは、epochの変更が原因ではない。もしそうならver1_2の時点で伸びているはず。(ver1_3)<br>
    - batch_sizeを128にしても大丈夫だった。スコアは0.898だった。推論時間は2時間弱ぐらい。この変動は何なのか。原因の１番の候補は乱数の影響でTTAの画像が変わったことか。(ver1_4)<br>
    - batch_sizeを256にするとエラーになった。ここまでの実験をまとめる。<br>
    - batch_size | epochs | LB 
      :-----: | :-----: | :-----: 
      16 | [6, 7, 8, 9] | 0.897 
      32 | [7, 8, 9, 10] | 0.897 
      64 | [7, 8, 9, 10] | **0.900** 
      128 | [6, 7, 8, 9] | 0.898
      128 | [7, 8, 9, 10] | 0.898 
      256 | [7, 8, 9, 10] | error <br>
    - これを見る限り、epochsを変えても結果は変わらない。batch_sizeの変更によるスコアの上下は本質的な改善ではないため、気にしないほうがいいはず。
      
### 20210112<br>
- nb027(inference_EfficientNet_Averaging)<br>
  - ver1<br> 
    - nb013_ver3(Effinet_B1ns), nb019_ver1(Effinet_B2ns), nb023_ver1(Effinet_B3ns), nb025_ver1(Effinet_B4ns)のAveraging。9時間をオーバーしそうなので一旦B0はのぞいてある。<br>
    - 間違えて、画像サイズを380にしてしまった。batch_sizeも32で固定されてしまっている。<br>
    - もしB4nsがbatch_size32でも通るのであれば、推論時間の短縮になるのでそうしたい。<br>

### 20210115<br>
  - この三日間でひたすらbatch_sizeを変えて実験をしていたので、それをまとめる。
  - batch_size | B0 | B1ns | B2ns | B3ns | B4ns 
    :-----: | :-----: | :-----: | :-----: | :-----: | :-----: 
    16 | - | - | - | 0.899 | 0.897 
    32 | 0.893 | 0.896 | 0.895 | 0.900 | 0.897
    64 | 0.891 | 0.897 | 0.895 | 0.898 | 0.900
    128 | 0.891 | 0.893 | 0.898 | 0.899 | 0.898 
    256 | - | - | 0.894 | Error | Error <br>
  - B4は128まで上げられたので、小さいモデルに関してはもっと大きくすることもできそう。<br>
  - 何度も書いている通り、これらのスコアの変動は、TTAの変動によるもので、public dataのラベルのノイズに引っ張られてる可能性が非常に高いため気にするのはよくない。とりあえず推論時間の都合上64にするが今後圧迫されるようなら128や256に上げてもいいはず。<br>

- nb027(Averaging)<br>
  - ver2<br>
    - 得られて実験データを元に、batch_sizeを64に固定して、画像サイズをモデルごとに変更するように修正した。<br>
    - LBが0.872だった。これはあり得ないだろ...<br>
    - 途中のpredictionsの変数名を書き間違えていたことに気がついた。これのせいか。<br>
  - ver3<br>
    - LBは0.898だった。単体のモデルの精度を超えられない。DiscussionではTTAは2~3回がいいという意見が多数派だった。もしかするとアンサンブルをする段階においてはそうなのかもしれない。しかし、試してみる優先順位としては低い。<br>
    - モデルごとにdataloaderを用意して推論をしていたが、よく考えたらstatesに複数のモデルの分まで全て一つに押し込めばコードも短くなるし時間も大幅に短縮できる。これならアンサンブルの幅が大きく広がる。<br>

### 20210116<br>
- nb019<br>
  - ver1(再掲)<br>
    - 推論の段階で、epochごとにttaを5回やっていたのを、statesに全てのモデルを入れてからttaを回すように変更した。これでloaderを回す回数がttaごとに一回で済む。<br>
    - 失敗したのでver2でやってみる。<br>
  - ver2<br>
    - fold0とfold1の両方のモデルを作った。<br>
    - これまで通りにサブするとLBが0.896, 時間節約の方法でサブすると0.894だった。若干LBスコアは下降しているが、この方法ならモデルをどれだけ作ろうとsubにかかる時間は大して増えないため、アンサンブルをする際にはこれを検討したい。(ただし、そもそもfold0だけの時よりもスコアが悪化していることに注意)<br>
    
### 20210128 <br>
- テストから解放されたのでkaggleを再開する。LBがほとんど動いてないので、ラベルがNoisyなことに対する
解決策はまだ出てないみたい。<br>
- 今日はこれまでの日記の振り返りと、学習を回すところまで。サブなどの記録はまた明日。<br>

### 20210129<br>
- nb013(EffinetB1ns)<br>
  - ver4<br>
    - アンサンブルの実験をするためにfold0に加えてfold1も学習させる。<br>
- nb019(B2ns)<br>
  - ver4(ver3は失敗)<br>
    - fold0とfold2を学習させる。<br>
    - ver4_1はfold2のみの短縮サブ、ver4_2はfold2のみの通常サブ、ver4_3はfold0と2の通常サブ。<br>
    - fold | CV | LB 
      :-----: | :-----: | :-----: 
      0 | 0.88459 | 0.899
      2(短縮) | 0.88627 | 0.892 
      2(通常) | - | 0.896
      0&2(通常) | - | 0.895 <br>
    - アンサンブルするとスコアが悪化してしまった。<br>
- nb023(B3ns)<br>
  - ver2(B3ns)<br>
    - fold0とfold3を学習させた。<br>
    - fold | CV | LB 
      :-----: | :-----: | :-----: 
      0 | 0.89427 | 0.899
      3 | 0.89045 | 0.896 
      0 & 3 | - | 0.898
    - やはり、単体のLBは超えなかった。
- nb025(B4ns)<br>
  - ver2<br>
    - fold0とfold4を学習させたが、タイムオーバーでエラーになってしまった。<br>
  - ver3<br>
    - fold4のみを学習させる<br>
    - CV | LB 
      :-----: | :-----: 
      0.89235 | 0.898 <br>
- nb028(create_model_seresnext50_32x4d)<br>
  - ver1<br>
    - モデルをseresnextに変更した。画像サイズは380、batch_sizeは32で、foldは0のみ。<br>
    - CV | LB 
      :-----: | :-----: 
      0.89351 | 0.895 <br>
    - 画像サイズをあげればスコアはまだ伸びそう。<br>
    
### 20210131<br>
- 1月30日の分は1月29日のところにまとめて書いた。
- [このディスカッション]によると、healthyのラベルがつけられたデータのうちのほとんどは、病気になっている葉が混ざっているらしい。専門家は多少の葉が病気になっていても概ね健康であればhealthyのラベルをつけるのに対して、モデルは敏感に感知してしまい、病気のラベルに誤分類している可能性がありそう。<br>
- [arutema47さんのディスカッション]によると、クラス0と4が状況が悪く、さらに0の多くが4(healthy)に誤分類されているらしい。healthyのデータの多くに病気の葉が混ざっているせいで、病気の葉が少ないデータに対する精度が落ちてる感じがする。
- nb027<br>
  - ver4~6<br>
    - B2nsとB3nsとB4nsの3つのモデルを色々混ぜてみたが、なぜかTime overになってしまい、全てエラーになった。<br>
    - そもそもEfficientNetを大量に混ぜること自体にあまり意味が見いだせない気がする。混ぜるなら違うモデル同士を混ぜた方がいい。<br>

### 20210201<br>
- nb005<br>
  - ver49~51<br>
    - cutmix, fmix, mixupを実装した。cutmixの原論文を読んでcutmixとmixupについては大体理解できたが、fmixについてはよくわかってないので、原論文を読む必要がある。ディスカッションでは、この3つのうちどれを使うのがいいかについては意見が割れており、どれも大差ないようだった。<br>
    - これらの手法で各クラスの中間のようなもので訓練できれば、healthyクラスの一部の病気の葉に引きずられて誤分類してしまう問題が回避できる可能性がある。<br>
    - ver49_1はcutmixでbatchsizeが64, どれ以降は全てbatch_sizeは32で、ver49_2はcutmix、 ver50はfmix、ver51はmixup。ただし、mixupについては自分で実装したため、誤っている可能性がなくはない。多分あってるとは思うけど...<br>
    - 結果は次の通り。<br>
    - 手法 | CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: | :-----: | 
      mixup | 0.87130 | 0.887 | 0.9580 | 0.4436 
      cutmix | 0.87870 | 0.890 | 0.7935 | 0.4759 
      fmix | 0.88041 | 0.890 | 0.7570 | 0.4596 <br> 
    - valid_lossは以前よりかなり下がっているが、train_lossが下がりきってない。そもそもモデルが小さいことがその原因っぽいので、モデルを大きくする必要がありそう。<br>
### 20210202<br>
- nb023(B3ns)<br> 
  - ver4~6<br>
    - mixupとsnapmixを試した。<br>
    - augmentation | CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: | :-----: | 
      mixup | 0.88591 | 0.893 | 0.9320 | 0.4346 
      snapmix(p=0.5) | 0.88497 | 0.890 | 1.1941 | 0.4103 
      snapmix(p=1.0) | 0.88136 | 0.884 | 1.6599 | 0.4418 <br>
    - discussionではsnapmixは効くと言われていたのに効かなかった。snapmixの割合を落とせばいいのかもしれない。<br>

### 20210203<br>
- 今日は一日テスト勉強に費やした。<br>
### 20210204<br>
- nb028<br>
  - ver3<br>
    - 画像サイズを410にした。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----:  | :-----: 
      0.89237 | 0.897 | 0.4325 | 0.5610 <br>
    - スコアが上がった。これを見る限り、augmentation(snapmixとか)を強くしすぎてもスコアが下がるのは妥当だと思われる。<br>
- [transformerについてのQiita記事](https://qiita.com/omiita/items/07e69aef6c156d23c538)や[英語の記事](http://jalammar.github.io/illustrated-transformer/)を見てtransformerについての理解を深めた。また、ViTについての[Qiita記事](https://qiita.com/omiita/items/0049ade809c4817670d7)も読んだ。明日からVision Transformerを実装したい。<br>

### 20210205<br>
- nb030(create_model_ViT)<br>
  - [このNotebook](https://www.kaggle.com/piantic/cnn-or-transformer-pytorch-xla-tpu-for-cassava)を参考にTPUでVision Transformerを実装した。<br>
  - なぜかGPUより遅いし、2epoch目以降は全く学習をしてくれない。<br>
  - 以下、動かしててわかったことを書く。<br>
    - DistritutedSamplerを使ってDatasetからsamplerを作ってDataLoaderに渡さないと並列化ができないらしい(少なくともnprocsを8にするときは必須)。<br>
    - 最後にxmp.spawnで複数のプロセスを起動する必要がある。これはもともとNotebookの最後には書いてあったが、見逃していた。<br>
    - optimizerを更新するときは、xm.optimizer_step(optimizer)を使って、複数のプロセスの勾配を平均する必要がある。それにより、SAMが使えなくなる(使えるかもしれないけど現時点の自分の実力では無理)。
    - optimizerやmodelは複数のプロセスで共通のものを使うため、train_loopの外に書く必要がある。よって、optimizerの学習率を途中で下げるために、loopの途中でoptimizerを定義し直すと学習が進まなくなる。<br>
    - os.environ["XLA_USE_BF16"] = "1" を打たないと、TPUの混合精度にならなくて遅い。[参考記事](https://speakerdeck.com/shimacos/pytorchdeshi-meruhazimetefalsetpu?slide=17)<br> 
    - TPUが2epoch以降に早くなるのは仕様。[参考記事](https://qiita.com/koshian2/items/fb989cebe0266d1b32fc)<br>
  - ver1~3はいずれも失敗に終わった。<br>
- GitHubの更新をdatasetに反映させるためにupdateをして、notebook内でも更新をしてsession restart, ページのリロードも行ったところ、ファイルの中のコードが変わったにも関わらず、実際に動いているのは更新前のコードというバグがあった。一度Notebookを閉じて再び開けると直ったため、今後は気をつける必要がある。(ちなみに悩まされていたのはSAMのoptimizer.step(closure)をできるようにしたかったため)<br>

### 20210206<br>
- nb030<br>
  - ver4<br>
    - モデルの重みを一旦凍結して、後から解凍した時、うまく学習してるかが判別できなかったため、とりあえずそのオプションは外した。<br>
    - また、schedulerをGradualWarmupSchedulerV2に変更している。(途中でoptimizerの重みを変更できなかったため。)<br>
    - 全然学習が進まなかった。ただ、並列化などはうまく行っている。schedulerが原因である可能性が大きい<br>
    - その後も頑張り続けたが、全く動かなかった。<br>
  - ver5<br>
  - schedulerをCosineAnnealingWarmRestartsに戻した。
    
- nb023(B3ns)<br>
  - ver7・8<br>
    - Snapmixの割合をに下げた<br>
    - 結果は次の通り<br>
    - snapmix | CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: | :-----: 
      0.2 | 0.88686 | 0.896 | 0.8799 | 0.4346 
      0.1 | 0.88819 | 0.894 | 0.7494 | 0.4805 <br>
    - 効いてないように見える。snapmix系のaugmentationでラベルノイズを攻略するのは難しいと判断する。<br>
    - snapmixの確率をあげるとvalid_lossは小さくなるが、CV, LBは悪くなっている。これは根本的にラベルノイズの解決にはなってないことを意味する。(と思ってる。)<br>
- nb028(SeResNeXt)<br>
  - ver5(ver4は失敗)<br>
    - 画像サイズを440に上げた。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: | 
      0.89446 | 0.895 | 0.4779 | 0.5032 <br>
    - CVはよくなってるが、LBが悪くなってる。TTAの影響でたまたまLBがよくない可能性は否めない。<br>
  
### 20210207<br>
- [このディスカッション](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/217328)によると、shakeはあまり大きくなく、9割は+/-0.005程度に落ち着くらしい。コメントを見ると、そもそもprivateスコアがリークしてることがわかった時、ほとんどpublicスコアとの差がないことが判明されてることが指摘されている。今回はLBを重視していいと考えられる。MoAの反省も踏まえると、shakeをビビりすぎてはいけない。<br>
- 今更、optimizerをAdamにしてgradient_accumulationを上げたほうがいいような気がしてきた。ほとんど誰もSAMを使ってない。ひと段落したら試してみてもいいが、一度決めたことを変えすぎるのはよくないとMoAで反省をしたので、あまり優先度は高くしないでおく。<br>
- clean_labを使ってみた。全てを理解することはできなかったので、[論文](https://arxiv.org/pdf/1911.00068.pdf)を参考にしつつ、[公開されているNotebook](https://www.kaggle.com/telljoy/noisy-label-eda-with-cleanlab?scriptVersionId=53077552&select=oof.csv)からdenoiseされたラベルデータを取得して自分のNotebookに組み込むことにした。<br>
- nb023<br>
  - ver10(ver9は失敗)<br>
    - clean_labで更新されたlabelで訓練した。それ以外はver1と全く同じ。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----:
      0.93926 | 0.893 | 0.2333 | 0.2148 <br>
    - モデルの予測を元に更新したラベルを使ってるので、当たり前な結果が出てきた。ラベルを入れ替えるよりは、間違いっぽいラベルを外す方が良さそう。<br>
  - ver11<br>
    - 今度は、noisyラベルを外して訓練してみる。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----:
      0.95170 | 0.897 | 0.2027 | 0.1751 <br>
    - CVは明らかに高いが、一応denoiseの形としてはこっちが正解なのかもしれない。ここにsnapmixを加えてみても良さそうではある。<br>
- nb031<br>
  - ver1<br>1
    - 公開Kernelを使わせていただくことにした。DEiTを使う。(schedulerのみGradualWarmupSchedulerV2から'CosineAnnealingWarmRestarts'に変えて、epochも20から10に落としている)<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----:
      0.86729 | - | 0.5808 | 0.5694 <br>
    - 明らかに学習しきれていないのでsubはしてない。やはりepochやschedulerはいじらない方がよかったかもしれない。<br>
  - ver2<br>
    - モデルをViTに変えた。軽めに実験した感じでは、lrは1e-4が一番よかった。<br>
    - 公開Kernelを使わせていただくことにした。DEiTを使う。(schedulerのみGradualWarmupSchedulerV2から'CosineAnnealingWarmRestarts'に変えて、epochも20から10に落としている)<br>学習
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----:
      0.86729 | - | 0.6204 | 0.5708 <br>
    - ViTにしてもほぼ同じ結果が出た。本格的にわからない...<br>
    - [Discussion]を見ると、DeiTの画像サイズが384のweightがあるらしいので変更する。<br>
  
### 20210208<br>
- nb023<br>
  - ver12<br>
    - denoiseした上でsnapmixを加えてみる。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----:
      0.94716 | 0.894 | 0.7098 | 0.1719 <br>
    - スコアが悪化した。snapmixは完全に諦めることにする。<br>
- nb027(Averaging)<br>
  - ver8<br>
    - denoiseをしたB3ns(ver11)と普通のB2nsを混ぜてみた。<br>
    - B2nsが0.898、B3nsが0.897、アンサンブルが0.898なので、あまり効果がなかった。denoiseをしたとはいえEfficientNetを混ぜるだけでは意味がないのかもしれないし、そもそも確率をaveragingするよりは多数決にしたほうがいいのかもしれない。<br>
- nb028<br>
  - ver6<br>
    - 画像サイズを410に戻し、Rand_Augmentを実装した。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----:
      0.88743 | 0.895 | 0.5569 | 0.5467 <br>
    - スコアが悪化した。今回はAugmentationは厳しくしても、noisyラベルに対してはあまり効果がなさそう。もしかすると、DeiTでもRand_Augmentを抜いた方がいいのかもしれない。<br>
- nb031<br>
  - ver5<br>
    - モデルをdeit_base_patch16_384に変更した(distilledはなぜか動かなかった)。<br>
    - inferenceの時もinternetが必要ということがわかったので、githubからモデルをコピペしたけど、これはグレーゾーンだと思う。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----:
      0.88715 | 0.896 | 0.5281 | 0.5344 <br>
    - アンサンブルに十分使えそうなモデルができた。loss_functionを変えたりして微調整をしたい。<br>
  - ver6<br>
    - Bi_Tempered_Lossにした。これまで通り,t1=0.8, t2=1.5, smooth=1e-2にしている。(smooth=5e-2がデフォルトだったが、少し様子を見る限りでは1e-2の方が良さそう。)<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----:
      0.88505 | 0.894 | 0.2730 | 0.2658 <br>
    - TaylarCrossEntropyLossの方がCV, LB共によかった。逆に、EfficientNetやSeResNeXtでもそれが当てはまる可能性がある。<br>
    
- [RandAugmentに関するわかりやすい記事](https://qiita.com/takoroy/items/e2f1ee627311be5d879d)を見つけた。<br>

### 20210209<br>
- nb023<br>
  - ver14<br>
    - denoiseした上でRand_Augmentを使ってみた。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----:
      0.93829 | 0.889 | 0.3928 | 0.2525 <br>
    - 全然ダメだった。<br>
- nb028<br>
  - ver8(ver7は失敗)<br>
    - ver3の状態から、loss_functionをBiTemperedLossからTaylorCrossEntropyLoss(以下TCE)に変更した。原論文は[これ](https://www.ijcai.org/Proceedings/2020/0305.pdf)で、実装(非公式)は[ここ](https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py)。<br>
    - 結果が全く良くなかった。原論文を読んで気がついたが、どうやらテイラー展開の次数がパラメータになっていて、低ければ低いほどMAEに近く(よりロバストに)なり、大きければ大きいほどCCEに近く(よりセンシティブに)なるらしい。ver8はt=2で回しており、tが低過ぎたらしい。その後に軽くt=4で回したらかなりうまく学習が進んだ。原論文ではt=2, 4, 6が実験されていた。<br>
- nb031<br>
  - ver7<br>
    - Rand_Augmentを外してみた。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----:
      0.86449 | - | 0.4789 | 0.5421 <br>
    - 結果があまり良くなかった。Rand_Augmentもモデル次第で効いたり効かなかったりする。ますますわからん。<br>
  - ver8・ver9<br>
    - Rand_Augmentを入れ直して、TaylorCrossEntropyLossのnを4にした。<br>
    - n | CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: | :-----: 
      2 | 0.88715 | 0.896 | 0.5281 | 0.5344
      4 | 0.88668 | 0.892 | 0.5276 | 0.5346   
      6 | 0.88318 | 0.893 | 0.5522 | 0.5445 <br>
    - n=2はver5の再掲。n=2のスコアは越えられなかった。DeiTはデフォルトのスコアが限界かもしれない。<br>
- GCPのインスタンスを立てることに成功した。以下、その手順をまとめておく。
  1. [この記事](https://qiita.com/lain21/items/a33a39d465cd08b662f1)VMインスタンスを立てる。その時、「GCEインスタンスの作成」の(4)Frameworkでpytorch+cuda11.0を選んでおく。<br>
  2. [この記事](https://qiita.com/hiromu166/items/507fc0fb466c7149dccf)を参考にして、JUPYTER LABを開くを選択すれば、すぐにlabが使えるようになる。kaggle.jsonをGUIであげてkaggle apiを使えるようにすれば、インスタンスに直接データを入れられる。<br>

### 20210210<br>
- nb028<br>
  - ver9・ver10<br>
    - TCEのnを4と6にして実験した。<br>
    - n | CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: | :-----:
      4 | 0.89408 | - | 0.2960 | 0.3318   
      6 | 0.89503 | - | 0.3085 | 0.3389 <br>
    - インスタンスを消してしまったので、サブをすることができなかった。<br>
      
- GCEの中のデータをどうやって転送していいかがわからず、結局インスタンスを消した。GCEに自動でjupyter labを立てさせたはいいものの、肝心のファイルがどこにあるか全く見えなかった。sshで入っても何も見えず、docker containerの中にも何もない。notebook内で見えているpathはなんの環境の中のpathなのかが全くわからなかった。<br>

### 20210211<br>
- nb028<br>
  - ver11・ver12<br>
    - TCEのnを4と6にして実験した。<br>
    - n | CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: | :-----: 
      4 | 0.89332 | 0.898 | 0.3377 | 0.3392   
      6 | 0.89256 | 0.895 | 0.3396 | 0.3426 <br>
    - 僅差だが、TCEのn=4の時が一番LBスコアが高かった。GCPのインスタンス内で作ったモデルの方が精度が若干いいことが気になる。batch_sizeを小さくしてることが原因として考えられる。<br>
    - n=4の時のモデルで、[このNotebook](https://www.kaggle.com/japandata509/ensemble-resnext50-32x4d-efficientnet-0-903/notebook)を参考にTTAを変えてサブした(ver11_3)ところ、LBスコアは0.892だった。fold数をあげたりアンサンブルをしたりするときなどにスコアが伸びたりするかもしれないので、モデルが出来次第試す。<br>

### 20210212<br>
- nb023<br>
  - ver15・ver16<br> 
    - TCEのnを4と6にして実験した。<br>
    - n | CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: | :-----: 
      4 | 0.89161 | 0.897 | 0.3926 | 0.3492   
      6 | 0.89180 | 0.897 | 0.4022 | 0.3625 <br>

### 20210213<br>
- nb023<br>
  - ver17<br>
    - n=4で、optimizerをAdamにしてaccumulation_stepを4にして高速化を行った。訓練時間は約半分。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: 
      0.89180 | 0.895 | 0.2993 | 0.3412 <br> 
    - SAMを採用する。<br>
  - ver18<br>
    - ver15から、batch_sizeを16にあげた(batch_sizeの違いによるスコアへの影響を調べるため)。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: 
      0.89427 | 0.897 | 0.3429 | 0.3288 <br> 
    - batch_sizeは特に関係なかった。<br>
- nb028<br>
  - ver13<br>
    - n=4で、optimizerをAdamにしてaccumulation_stepを4にして高速化を行った。訓練時間は約半分。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: 
      0.88933 | 0.895 | 0.3245 | 0.3520 <br> 
    - SAMを採用する。<br>
  - ver14<br>
    - ver11から、batch_sizeを32にあげた(batch_sizeの違いによるスコアへの影響を調べるため)。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: 
      0.89560 | 0.897 | 0.3045 | 0.3340 <br> 
    - batch_sizeは特に関係なかった。<br>
- nb031<br>
  - 数日前から急に、timmをimportしようとしたらtorch._sixからcontainer_abc(的なやつ)をimportできないというエラーが吐かれ、TPUが全く動かせなくなっていた。原因は、torch._sixが数日前にupdateされて、それに合わせてtimmもupdateされていたが、timmのdatasetに変更が反映されず、torch._sixとtimmの対応がおかしくなっていたことだった。kaggleのdatasetとgithubのリポジトリのversionが合っていないためにエラーが起こるのは仕方のないことなようなので、今後は気をつける。今回は、timmのdatasetを作り直すことで解決した。<br>
- timmでswsl_resnext101_32x8dというモデルを見つけた。パラメータの数も許容範囲内で、かなり精度もいい。SeResNeXtを超える可能性がある<br>
- [Discussion](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/218907)で、NFNetというモデルが紹介されていた。パラメータ数は多いが、早いらしい。SOTAらしいので試してみたい。<br>
- nb033(several_models)<br>
  - ver1<br>
    - B3ns(ver1)とSeResNeXt(ver3)をアンサンブルした。<br>
    - LBは0.899だった。スコアの伸びが期待できそう。<br>
    - ふと気づいたが、inferenceでseedを固定していないせいで、LBスコアが再現できていない可能性がある。次回から固定して検証する。<br>
    - 確認したら、ちゃんとseedを固定していた。<br>
### 20210214<br>
- nb023<br> 
  - 再現性が心配だったので、ver3で出した、batch_sizeが128でLBスコアが0.899のものを再現できるかどうかを確かめるために、ver1_6をサブミットした。ちゃんと再現できていた。<br>
  - ver19<br>
    - ver1から、foldを1に変更した。保存するepochを5以上にしておいた。<br>
  - ver20<br>
    - foldを4に変更した。<br>
- nb028<br>
  - ver15<br>
    - ver11を全foldに拡張した。<br>
- nb031<br>
  - ver10<br>
    - ver5を全foldに拡張した。<br>
    - CV | LB 
      :-----: | :-----: 
      0.86023 | 0.887 <br>
    - fold0以外がかなりCVが低かった。fold0はちゃんと再現できていたので、fold0だけがたまたま簡単だった可能性が高い。アンサンブルには、特定のfoldだけを選ぶべきなのかもしれない。<br>
- nb034(B3ns_2)<br>
  - ver1<br>
    - nb023_ver1のfoldを2にした。<br>
- nb035(swsl_resnext50_32x4d)<br>
  - ver1
    - swsl_resnext101_32x8dは重たすぎたので、swsl_resnext50_32x4dにした。これでもseresnextよりランキングが高い上、seresnextより軽くて早い。<br>
    - 画像サイズは410で、損失関数はbitemperedlossにした。<br>
    - CV | LB | train_loss | valid_loss 
      :-----: | :-----: | :-----: | :-----: 
      0.89028 | - | 0.4992 | 0.5296 <br>

### 20210215<br>
- アンサンブルで使う3つのモデルのCVをまとめる。<br>
- fold | B3ns | SeResNeXt | DeiT 
  :-----: | :-----: | :-----: | :-----: 
  0 | 0.89427 | 0.89332 | 0.88785 
  1 | 0.89541 | 0.89522 | 0.86659
  2 | 0.89026 | 0.88437 | 0.82099
  3 | 0.89045 | 0.89520 | 0.86445
  4 | 0.89520 | 0.89615 | 0.87076 <br>
- これらを見ると、fold2の学習が総じてうまくいってないことがわかる。考えられる一番の原因はfold2のoofのラベルにノイズが多く含まれていることであるため、fold2だけはdenoiseしてみたい。<br>
- サブに用いるepochは[6, 7, 8, 9]から[7, 8, 9, 10]に変更する。(単純にスコアがいいから。)<br>
- nb023<br>
  - ver21<br>
    - cleanlab_dataを用いてdenoiseし、fold2の学習を行った。<br>
  - sub1<br>
    - ver1を全てのfoldに拡張した。(13:35~)
    - LBは0.900で、subにかかった時間は大体3時間だった。<br>
- nb028<br>
  - ver16<br>
    - ver11から、denoiseして対象をfold2に変更した。<br>
- nb033<br>
  - ver2<br>
    - nb023_sub1, nb028_ver15, nb031_ver5をアンサンブルした。<br>
    - 0.899だった。辛すぎる。<br>
- nb037(ViT)<br>
  - ver1<br>
    - nb031_ver5から、modelをViTに変更して全foldに渡って学習させた。<br>
    - 結果は記録するほどではなかった。使わない。<br>

### 20210216<br>
- nb028<br>
  - ver18<br>
    - ver15から、とりあえずfoldを0のみにして画像サイズを512にあげた。<br>
- nb033<br>
  - ver3<br>
    - [このNotebook](https://www.kaggle.com/japandata509/ensemble-resnext50-32x4d-efficientnet-0-903)を参考に、inferenceを変えてみた。今までやっていたアンサンブルの方法だと、2つのモデルを混ぜてもあまり精度が上がらないのかもしれない。(TTAをやりすぎた??)このEfficientNetB3nsとResNeXtをこの方法で混ぜて0.903に届くのなら、アンサンブル次第では自分のモデルももっと上に行けるはず。<br>
    - Time outになってしまった。<br>
  - ver4<br>
    - B3nsとSeResNeXtのCVを色々探ってみた。どうやら、全foldを混ぜてもCVはあまり伸びないらしい。大きく伸びたのはfold4のペアで、次がfold1だった。<br>
    - fold | B3ns | SeResNeXt | アンサンブル
      :-----: | :-----: | :-----: | :-----: 
      1 | 0.89541 | 0.89522 | 0.89977 
      4 | 0.89520 | 0.89615 | 0.89880 
      1&4 | 0.89530 | 0.89568 | 0.89920 
      全fold | 0.89312 | 0.89285 | 0.89577 <br>
    - これを見ると、とりあえず全部を混ぜても大した精度が得られない可能性がある。もちろん、特定のfoldだけを使うと、privateのラベルの分布からズレてshake downする可能性があるが、とりあえず上にいけないと話にならないので、やれることはやってみる。金圏にいる人はCVが0.9を超えているため、CVが上がらない限り近づける可能性は低そう。<br>
    - 2020データと2019データで、CVスコアの分布が変わっているのでそれも記録する。(表の価はアンサンブル後のもの)<br>
    - fold | 2019 | 2020 | 2019&2020 
      :-----: | :-----: | :-----: | :-----: 
      fold0 | 0.90622 | 0.89153 | 0.89427
      fold1 | 0.91221 | 0.89709 | 0.89977
      fold2 | 0.89875 | 0.88936 | 0.89121
      fold3 | 0.90329 | 0.89282 | 0.89482
      fold4 | 0.89848 | 0.89888 | 0.89880
      全fold | 0.90364 | 0.89396 | 0.89577 <br>
    - 2020のデータだと、fold4が一番CVがいいが、2019のデータになるとそうでもないらしい。fold1とfold4を混ぜてみることにする。<br>
    - 正直、ここでやっていることは意味があるか微妙だが、何もできずに終わるよりはマシなので最後までできることはする。<br>
  - ver5<br>
    - ver3から、DeiTを抜いて、さらにfoldを1と4だけにした。<br>
    - LBは0.899のままだった。案の定意味がなかった。<br>
  - ver7(ver6は失敗)<br>
    - tfn_foldは全foldに戻した。過去の実験結果をみていると、画像サイズが大きい場合にTTAがほとんど効いていないようだったので、とりあえず外してみた。軽く入れるなら意味がありそうなので、それでもいいかもしれない。<br>
    - LBは0.893だった。ある程度はTTAが効いていたらしい。<br>
  - ver8<br>
    - ver5から、foldを全てに戻して、TTAをRandomResizedcropだけにした。<br>
    - LBが0.900に上がった。こっちの方が良さそう。<br>
  - ver9<br>
    - ver3から、B3nsのepochをbestのみにした。<br>
    - LBが0.897で所要時間が3時間ほど。これは望みがあるぞ。B3nsのepochも増やせるし、seresnextのTTAも追加できる。<br>
### 20210217<br>
- nb028<br>
  - ver19<br>
    - learning_rateを3.0e-5に上げて、foldを1~4にした。<br>
- nb033<br>
  - ver11(ver9, ver10は失敗)<br>
    - B3nsのepochを[9, 10]に変更した。また、SeResNeXtのTTAを復活させ、trainsformerをRandomResizedCropだけにした。<br>
    - LBは0.898だった。なんで？？？もうまじでわからん。<br>
  - ver13(ver12は失敗)<br> 
    - ver11から、SeResNeXtのTTAの回数を減らした。<br>
  - ver14<br>
    - verから、SeResNeXtもB3nsと同じ方法で TTAを行った。<br>
  - ver15<br>
    - ver14で、stackingができるようにコードをかいた。(サブはしない。)<br>
  - ver16<br>
    - ver9からstackingを行った。<br>
    - LBは0.896。stackingは望み薄っぽい。<br>
  - ver17<br>
    - ver8から、SeResNeXtのTTAを無くした。<br>
    - LBは0.899。<br>
- nb038(stacking_mlp)<br>
  - 俵さんの[記事](https://tawara.hatenablog.com/entry/2020/12/16/132415)を参考にstackingをやってみることにした。mlpによるstackingのためのNotebook<br>
  - ver1<br>
    - nb023とnb028の全foldのoofを使ってモデルを作った。<br>
  - ver2<br>
    - nb023のoofを、inference_with_ttaのものに置き換えた。<br>
- nb039(stacking_1dcnn)<br>
  - ver1<br>
    - 1DCNNでstacking用のモデルを作った。用いたデータはnb023とnb028のoof。<br>
  - ver2<br>
    - nb023のoofを、inference_with_ttaのものに置き換えた。<br>
- nb040(stacking_2dcnn)<br>
  - ver1<br>
    - 2DCNNでstacking用のモデルを作った。用いたデータはnb023とnb028のoof。<br>
  - ver2<br>
    - nb023のoofを、inference_with_ttaのものに置き換えた。<br>
- nb041(create_oof_B3ns)<br>
  - ver1<br>
    - inferenceで行うTTAと同じAugmentationを使ってoofを作らないといけないと考えたので、oofを作るためのNotebookを作った。inference_with_ttaで、epochは[9, 10]を使うことを想定した。<br>
  - ver2<br>
    - epochをbestのものだけにした。<br>
- nb042(create_oof_SeResNeXt)<br>
  - ver1<br>
    - nb041_ver2のモデルをSeResNeXtにした。<br>

### 20210218<br>
- 今日は最終日。<br>
- nb023<br>
  - sub1から、TTAをRandomResizedCropだけにしてsub1_2とした。
  - sub1_2から、inference_with_ttaを加えてアンサンブルしてsub1_3とした。<br>
- nb033<br>
  - ver18<br>
    - ver8のSeResNeXtを、画像サイズが512のものに変更した。<br>
    - Timeoutだった。このタイミングでやらかしてしまった。<br>
  - ver20(ver19は失敗)<br>
    - ver18から、fold2を削った。今更気づいたが、訓練データを15000枚とって使えば、大体サブにかかる時間がわかる。もっと早く気付きたかった。<br>
  - ver21<br>
    - ver9から、SeResNeXtの画像サイズを512に上げて、TTAを復活させてfold2を削った。<br>
- nb041<br>
  - ver3<br>
    - 今までずっと使っていたTTAでoofを作るコードをかいた。(commitはしてない。)<br>
