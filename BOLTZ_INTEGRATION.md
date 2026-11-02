# Boltz-2 を DiffPharma の親和性 oracle として組み込む — 設計と進捗

対象: Notion の Phase 0（Boltz-2 スループット実測と oracle 予算の確定）と
Phase 5（Boltz-2 親和性最適化への拡張と他手法比較）。

- Phase 0: https://app.notion.com/p/40c69893cf8f81cf88ced19ebe77989b
- Phase 5: https://app.notion.com/p/40c69893cf8f8163ac29e49d266ca258

論文: Passaro, Corso, Wohlwend et al., "Boltz-2: Towards Accurate and Efficient
Binding Affinity Prediction", bioRxiv 2025.06.14.659707 (CC-BY 4.0).

このファイルは**設計判断と検証済みの事実**を残すためのもので、未確認の項目は
明示的にそう書いてある。実装を引き継ぐ場合は第 6 節から読めばよい。

---

## 1. 前提: Boltz は Pharma-env に入らない（検証済み）

PyPI メタデータで確認（2026-11-02、boltz 2.2.1）。全軸で衝突し、うち 2 つは完全固定:

| | Pharma-env (`environment.yml`) | boltz 2.2.1 |
|---|---|---|
| torch | 2.1.1 (cu118) | `>=2.2` |
| numpy | 1.22.3 | `>=1.26,<2.0` |
| pytorch-lightning | 1.7.4 | **`==2.5.0`**（完全固定） |
| rdkit | 2022.03.2 | `>=2024.3.2` |
| scipy | 1.7.3 | **`==1.13.1`**（完全固定） |

Pharma-env 側を上げる逃げ道もない。`SETUP_LOCAL.md` の通り env は Linux ビルドで
完全固定されており、配布 `DiffPharma_best.ckpt` は pytorch-lightning 1.7.4 の
チェックポイントである。

**帰結: 2 つのスタックはプロセス境界で通信する。import 境界では不可能。**
`import boltz` を前提にした設計は最初から成立しない。

## 2. 配置

```
sekijima_lab_local/
  boltz/                         # upstream clone（参照・ピン留め用。改変しない）
  DiffPharma/
    scripts/boltz/setup_env.sh   # 境界の向こう側（conda env）を作る  ... 実装済み
    analysis/boltz_affinity.py   # 境界のこちら側（oracle クライアント）... 未実装
    config/boltz.yml             # ランチャと推論パラメータ            ... 未実装
```

- `boltz/` のソースは DiffPharma のビルド経路に入れない。clone は参照と
  コミット固定のため。pip install は clone から非 editable で行うので、
  clone は汚れない。
- 重いもの（weights, CCD, MSA キャッシュ, スコアキャッシュ）は
  `$HOME/workspace/DiffPharma_data/boltz_cache/` に置く。`/mnt/c` は遅く、
  20 秒/リガンドの oracle では I/O が支配的になりうる（`SETUP_LOCAL.md` 第 2 節
  と同じ理由）。
- `.gitignore` が `env/` と `lib/` を無視するので、その名前のディレクトリは使えない。

## 3. 現在の進捗

| 項目 | 状態 |
|---|---|
| boltz clone | **完了**。`../boltz`、タグ `v2.2.1`、commit `cb04aeccdd480fd4db707f0bbafde538397fa2ac` |
| `scripts/boltz/setup_env.sh` | **完了・動作確認済み**（exit 0） |
| conda env `boltz-env` | **完了・動作確認済み**（下記 provenance） |
| `analysis/boltz_affinity.py` | **実装済み・ローカルテスト済み**（未コミット） |
| `config/boltz.yml` | **実装済み**（Eagle の `boltz-env` launcher、未コミット） |
| スモークテスト（推論の実走） | **未実施**。CLI は確認済み。初回実行は ColabFold MSA server への配列送信と重み取得を伴うため、明示承認待ち。 |

env の構築コマンド:

```bash
BOLTZ_SRC=/mnt/c/Users/rixyt/workspace_eagle/sekijima_lab_local/boltz \
  bash scripts/boltz/setup_env.sh
```

スクリプトは `$BOLTZ_CACHE/provenance.txt` に boltz commit / torch バージョン /
`torch.cuda.is_available()` を書き出す。構築時の実測値:

```
boltz_commit=cb04aeccdd480fd4db707f0bbafde538397fa2ac
boltz_describe=v2.2.1
boltz_dirty=0
conda_env=boltz-env
conda_prefix=/home/rixyt/miniconda3/envs/boltz-env
python=Python 3.11.18
torch=2.15.0+cu132
cuda_available=True
built_at=2026-11-02T18:18:06+09:00
```

動作確認済み（実行して確認した内容）:

```
$ /home/rixyt/miniconda3/envs/boltz-env/bin/boltz --help
Commands:
  predict  Run predictions with Boltz.

$ .../boltz-env/bin/python -c "import boltz, torch; ..."
boltz 2.2.1
torch 2.15.0+cu132 NVIDIA GeForce RTX 4090
```

pip は torch を CUDA 13 系ホイールで入れたが、Eagle のドライバ 610.88 で
GPU は正しく見えている。**まだ推論を 1 件も走らせていない**ので、
重みのダウンロードと実際の予測は未検証（第 6.5 節の手順 2）。

## 4. Boltz-2 のインターフェース（ソースで検証済み）

`../boltz` の v2.2.1 を読んで確認した。再調査しなくてよい。

### 入力 YAML（`examples/affinity.yaml`）

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLL...     # ← 配列。PDB ではない
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
properties:
  - affinity:
      binder: B                          # どの鎖の親和性を出すか
```

ポケット条件付けも可能（`examples/pocket.yaml`）:

```yaml
constraints:
  - pocket:
      binder: B1
      contacts: [ [ A1, 829 ], [ A1, 138 ] ]   # [chain, 残基番号]
```

### 出力

`<out_dir>/boltz_results_<入力の stem>/predictions/<record_id>/affinity_<record_id>.json`

（`src/boltz/main.py:1134` と `src/boltz/data/write/writer.py:331` で確認）

入力にディレクトリを渡すとバッチになる。`stem` はそのディレクトリ名、
`record_id` は各 YAML のファイル名 stem。これが shard 実行の単位になる。

JSON のフィールド（`writer.py:306-334`）:

| フィールド | 意味 |
|---|---|
| `affinity_pred_value` | アンサンブル平均の連続値 |
| `affinity_probability_binary` | アンサンブル平均の binding likelihood |
| `affinity_pred_value1` / `2` | アンサンブル各メンバの連続値 |
| `affinity_probability_binary1` / `2` | 同、likelihood |

**単位と符号: 連続値は µM を基準にした log10 スケール（論文 §B「all converted to
the logarithmic scale using µM as standardized unit」）。したがって低い方が強い。**

論文自身の最適化で使っている報酬（§5.4、**第 1 アンサンブルメンバ**を使う）:

```
score = max((-affinity + 2) / 4, 0) * likelihood
```

### Phase 0 で振るべき CLI ノブ（`src/boltz/main.py`）

| オプション | 既定 |
|---|---|
| `--sampling_steps_affinity` | 200 |
| `--diffusion_samples_affinity` | 5 |
| `--recycling_steps` | 3 |
| `--sampling_steps` | 200（構造側） |
| `--diffusion_samples` | 1（構造側） |
| `--affinity_mw_correction` | off（分子量補正） |
| `--subsample_msa` / `--num_subsampled_msa` | / 1024 |
| `--no_kernels` | off |
| `--use_msa_server` / `--msa_server_url` | off / `https://api.colabfold.com` |
| `--seed` | None（**必ず固定すること**） |

コストの出発点: 論文は H100 で **約 20 GPU 秒/リガンド**（Table: "Boltz-2 ML
20 GPU sec"）。4090 ではその 2〜3 倍を見込む。

## 5. 未解決の設計問題（最重要）

### 5.1 Boltz は配列を要求するが、DiffPharma が持つのはポケットのみ

検証済み: `example/14gs.pdb` は **32 残基・非連続**（chain A の残基 7〜103 のうち
32 個だけ）。`dataset/test/` の 100 件も同じ `*-pocket10.pdb` 形式。

```
$ grep '^ATOM' example/14gs.pdb | cut -c22-27 | sort -u | wc -l
32
```

**この 32 残基を「配列」として Boltz に渡すのは誤り。** Boltz は 32 残基の
ペプチドとして折り畳むので、結合site の幾何にならず、親和性予測は無意味になる。

選択肢:

1. **全長配列を別途用意する**（推奨）。ファイル名が PDB ID 由来
   （`14gs-A-rec-20gs-cbd-lig-...`）なので、RCSB / UniProt から 100 ターゲット分の
   配列を引ける。`constraints.pocket` でポケット残基を指定して結合site を固定する。
   コスト: 全長を折り畳むので 1 件あたりの時間が増える。Phase 0 の実測対象。
2. ポケットのみをペプチドとして渡す → **却下**（上記の理由）。

**つまり oracle はポケット PDB だけでは動かず、ターゲット → 全長配列の対応表が
必要。** これは Phase 0 の前提として先に作るべきデータ整備タスク。

### 5.2 ポーズを Boltz に再予測させるか

Boltz-2 は複合体構造を自前で予測してから親和性を出す。DiffPharma は
相互作用制約付きで 3D ポーズを生成するので、**Boltz が独自にポーズを張り直すと、
条件付けで保持したはずの水素結合・疎水性相互作用とは別の配置がスコアされる。**

Phase 5 の受け入れ条件「最適化後の分子で、指定した水素結合・疎水性相互作用が
維持されていることが示されている」と噛み合わない可能性がある。
`constraints.pocket` で緩和できるが、生成ポーズそのものを評価させる方法は
未調査。**Phase 0 で方針を決めて記録すること。**

## 6. 次にやること（実装の指示）

### 6.1 `analysis/boltz_affinity.py`

既存の `analysis/docking_GPU.py` と同じ位置づけ（外部プロセスを叩いてスコアを
返すモジュール＋ argparse main）。リポジトリはパッケージ化されていない素の
スクリプト集なので、それに合わせる。

**核は `score_batch(records, workdir, config) -> list[BoltzResult]`。**
単発の同期呼び出しはその薄いラッパにする。逆にしてはいけない。理由は 6.3。

必須要件:

- **キャッシュ。** キーは `(canonical SMILES, ターゲット ID, boltz commit, 推論パラメータ)`
  のハッシュ。論文も 60 worker 構成でキャッシュしている。Phase 0 の予算が
  数百回規模なら再評価の無駄は致命的。置き場所は ext4 側。
- **予算カウンタを oracle 側に持たせる。** Phase 5 の科学的不変条件
  「全手法を同じ oracle 評価回数で打ち切る」は、カウンタが最適化器側にあると
  手法ごとに実装がズレて守れない。カウントはキャッシュミスのみ。
- **2 つのヘッドを早期に畳まない。** `affinity_pred_value` と
  `affinity_probability_binary`、さらにアンサンブル各メンバを全部保存する。
  Phase 5 の「Boltz-2 と独立 docking スコアの相関図」には生の値が必要。
  論文の合成スコアは optional なヘルパーに留め、保存値にはしない。
- **符号変換を 1 関数に閉じ込める。** log10 µM は低い方が良い。最適化器が
  最大化するなら反転が必要。この手のサインバグは結果が出てから気づくと痛い。
- **失敗を黙って落とさない。** boltz は失敗例を `prediction["exception"]` で
  飛ばして JSON を書かない（`writer.py:303-305`）。出力 JSON の欠損は
  「スコア 0」ではなく失敗として扱う。

### 6.2 `config/boltz.yml`

ランチャを**設定で差し替えられる**ようにする。コードに埋め込まない:

- Eagle: `conda run -n boltz-env boltz predict ...`
- TSUBAME: `apptainer exec <sif> boltz predict ...`

推論パラメータ（第 4 節のノブ）とキャッシュパスもここに置く。

### 6.3 dev/prod の分離が API 形状を決める

- **Eagle**（開発）: WSL2 Ubuntu + RTX 4090。conda env で同期呼び出しでよい。
- **TSUBAME**（本番）: H100。既存の運用パターンがあるので再利用する:
  `active_space_research/outputs/ethylene_eval/t4_remeasure/transfer/`
  - Docker イメージをローカルで作り `tar.gz` → sha256 検証 →
    TSUBAME 上で `apptainer build` （`t4_build_image.sh`）
  - 環境は `.sif` に焼き、**コードは別の tar で bind mount** する
    （`apptainer exec --cleanenv --bind $CODE:/workspace --pwd /workspace`）。
    イメージの再ビルドは稀、コード更新は自由。
  - array job で shard 分割（`workload/shard_NNN.jsonl` → `results/shard_NNN.jsonl`）
  - provenance は `APPTAINERENV_*` で git commit を注入

  これは論文の並列 worker 構成（H100 × 60、Reward Queue + Replay Buffer）と
  そのまま対応する。1 array task = 1 Boltz worker = 1 GPU。

**なので TSUBAME 側は「shard ファイル in → shard ファイル out」になる。**
単発同期を核にすると移行時に最適化ループごと書き直しになる。

### 6.4 TSUBAME で先に確認すべきこと

**計算ノードから外に出られるか。** Boltz-2 は MSA が必要で `--use_msa_server` は
外部サーバ（colabfold）を叩く。HPC の計算ノードは通常 outbound が閉じているか
proxy 経由なので、閉じていると本番で詰む。

対処は決まっている: **テストセット 100 ポケット分の MSA を Eagle で precompute し、
イメージと一緒に転送する。** ポケットは固定なので 1 回で済む。スループットにも
効くので、Phase 0 の「計測条件を固定して記録する」に MSA 有無を含めるべき。

### 6.5 推奨する順序

1. boltz 同梱の `examples/affinity.yaml` でスモークテスト（env は構築済み）:

   ```bash
   conda run -n boltz-env boltz predict \
     /mnt/c/Users/rixyt/workspace_eagle/sekijima_lab_local/boltz/examples/affinity.yaml \
     --out_dir $HOME/workspace/DiffPharma_runs/boltz_smoke \
     --cache  $HOME/workspace/DiffPharma_data/boltz_cache \
     --use_msa_server --seed 0
   ```

   初回は重み（数 GB）と CCD をダウンロードするので時間がかかる。
   → 出力 JSON のフィールドを実物で確認し、第 4 節の表を裏付ける。
   → ここで 1 件あたりの所要時間の初期値も取れる。
3. ターゲット → 全長配列の対応表を作る（5.1。100 件）
4. `analysis/boltz_affinity.py` を `score_batch` 中心で実装
5. Phase 0 のスループット測定（4090 実測 → H100 の予算に換算）
6. 5.2 の方針決定を記録

## 7. 環境上の注意

- WSL への `cd` は `wsl -d Ubuntu --cd <dir> -- <cmd>` を使う。シェル内の
  `cd` とクォートの組み合わせは Windows 側のツールによって壊れることがある。
- ODDT を使うコード（`analysis/interaction_metrics.py` 等）は
  `conda activate Pharma-env` が必須。フルパスで python を叩くと
  `OBElementTable` の AttributeError になる（`SETUP_LOCAL.md` 第 2 節）。
  Boltz 側は Pharma-env とは完全に別環境なので、この注意は無関係。
- `DiffPharma` の origin は upstream の `sekijima-lab/DiffPharma`。
  研究コードを書き進める前に fork した方がよい
  （`CAPLA` → `inu-wanwan/capla-custom` の慣習に合わせる）。
