![0A63BFAC-7C55-49F6-A6BA-111F1DF9F6AD_4_5005_c](https://user-images.githubusercontent.com/70050531/102084360-65bf1d80-3e58-11eb-82de-647a2e845ed7.jpeg)
# Kaggle_Cassava_Leaf_Disease_Classification<br>
Cassava Leaf Disease Classification コンペのリポジトリ  
タスク管理ボード: [リンク](https://github.com/Yuki-Tanaka-33937424/Kaggle_Cassava_Leaf_Disease_Classification/projects/2) <- 大事!!!  

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
    - CV=0.87321, LB=0.879  ほぼ変わらん。なぜ？？？  
- nb 005   
  - ver1  
    - EfficientNet-B0を実装。実行時間が30分ぐらい減った!!  
    - AdaBliefはそのまま。  
    - CV=0.85288, LB=0.861  
    - モデルは軽い方がいいため、多少精度は落ちるかもしれないがEfficientNet_B0で実験をしていく。  
CVよりLBスコアの方が高いのはなぜ？若干違和感がある。

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
