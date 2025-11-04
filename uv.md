# 研究室向けメモ：pip / uv / conda の比較

## 要旨
Python 環境管理の代表的手段である pip, uv, conda を、研究室での実運用の観点（速度・再現性・配布・GPU依存・教育配布・CI）で比較する。結論として、**高速反復と再現性重視の研究用途には uv**、**OS依存バイナリやGPUライブラリ配布には conda（mamba）**、**互換性維持と軽量運用には pip** が最適である。

---

## 総合比較表

| 観点 | pip | uv | conda（mamba含む） |
|---|---|---|---|
| 目的 | 標準的なPythonパッケージ管理 | 高速かつ再現性の高いパッケージ管理と環境構築 | OS横断的な完全バイナリ配布と依存管理 |
| 依存解決速度 | 通常 | 非常に高速（Rust実装・並列処理） | mambaで高速、conda単体はやや遅い |
| 再現性 | 弱（外部ツール併用） | uv.lock により厳密固定可能 | env.yml による再現性（中程度） |
| GPU / CUDA | wheel次第 | wheelがあれば最速 | 強い（nvidia/conda-forgeチャネル） |
| ネイティブ依存 | ビルド必要 | キャッシュ再利用可 | ビルド済みバイナリ配布 |
| OS対応 | Python標準圏 | Python標準圏 | 強い（Win/Mac/Linux統一） |
| 教育配布 | 経験依存 | 簡潔（uv venv + uv sync） | 簡潔（conda env create -f） |
| CI/CD適性 | 一般的 | キャッシュ高速＋再現性高 | 実績多（mamba推奨） |
| 既存資産互換 | 最大 | pip互換100% | pip混在可能だが衝突注意 |
| 導入コスト | 最低 | 低（単一バイナリ） | 中（conda/mamba導入必要） |

---

## 各ツールの特徴と強み

### pip
- Python標準の最も軽量なパッケージ管理ツールである。
- 外部依存が少なく、サーバ環境などでも確実に動作する。
- 反面、環境再現や依存解決速度は弱く、再現性確保には pip-tools など外部補助が必要である。

### uv
- Ruff開発元Astral社製の次世代パッケージ管理ツールである。
- pip, venv, poetry, pip-tools の機能を統合している。
- uv venv, uv pip, uv run, uv lock で完結する統合UXを提供。
- 再現性（lockファイル）と速度（Rust実装＋キャッシュ）を両立している。
- pip互換なので既存のrequirements.txtがそのまま利用できる。
- 実験反復・輪講・再現性重視の研究環境に最適である。

### conda（mamba）
- Anaconda系のパッケージ管理であり、C/C++やFortran依存を含むバイナリを事前ビルド配布する。
- conda-forge や nvidia チャネルにより、GPUや科学技術計算系パッケージを簡単に導入できる。
- Windows / macOS / Linux の混在環境で強みを持つ。
- mamba を使えば高速化可能である。
- 反面、チャネル設定や依存競合が煩雑で、環境サイズが大きくなる傾向がある。

---

## 教育・輪講運用の適用例

### uvを中心とした運用（推奨）
1. 教員が requirements.txt を用意。
2. 学生は以下を実行するだけで同一環境を再現できる。
   - uv venv
   - uv pip install -r requirements.txt
   - 安定後に uv lock → 配布側は uv sync で完全再現。
3. 配布の手順が短く、CI/CDでもキャッシュ再利用で高速化できる。

### conda（mamba）運用
- CUDAや科学計算パッケージを扱う授業で便利である。
- Windows受講者が多い場合や、バイナリ依存が複雑な教材に適する。
- 教員は env.yml を配布し、学生は conda env create -f env.yml を実行する。

---

## 実行例（比較）

pip
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt

uv
- uv venv .venv
- uv pip install -r requirements.txt
- uv lock
- uv sync

conda（mamba）
- mamba create -n env python=3.11
- mamba activate env
- mamba install -c conda-forge -c nvidia pycuda numpy

---

## 推奨方針

1. 研究用途・反復実験・再現性重視 → uv  
   軽量・高速・lockによる完全再現。
2. 複雑なバイナリ・GPUドライバ含む教材配布 → conda（mamba）  
   OS横断・環境配布が容易。
3. 小規模スクリプトやサーバデプロイ → pip  
   標準で十分。

---

## 結論
uv は pip の上位互換的存在であり、従来の資産を壊さず再現性と速度を向上できる。
conda はGPUや科学技術分野での「動く保証」を重視する際に有効である。
研究室全体で統一方針を取るなら、
CPU実験・教育配布：uv、
GPU/NVIDIA環境・Windows含む配布：conda、
という二層運用が現実的である。