# LangChain と LangGraph による RAG・AI エージェント［実践］入門

「LangChain と LangGraph による RAG・AI エージェント［実践］入門」の GitHub リポジトリです。

https://www.amazon.co.jp/dp/4297145308

<img src="assets/cover.jpg" width="50%" />

## 各章のソースコード

| 章                                                                  | ソースコード                                                                                                                                                                          |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 第 1 章 LLM アプリケーション開発の基礎                              | -                                                                                                                                                                                     |
| 第 2 章 OpenAI の チャット API の基礎                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GenerativeAgents/agent-book/blob/main/chapter02/notebook.ipynb) |
| 第 3 章 プロンプトエンジニアリング                                  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GenerativeAgents/agent-book/blob/main/chapter03/notebook.ipynb) |
| 第 4 章 LangChain の基礎                                            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GenerativeAgents/agent-book/blob/main/chapter04/notebook.ipynb) |
| 第 5 章 LangChain Expression Language（LCEL）徹底解説               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GenerativeAgents/agent-book/blob/main/chapter05/notebook.ipynb) |
| 第 6 章 Advanced RAG                                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GenerativeAgents/agent-book/blob/main/chapter06/notebook.ipynb) |
| 第 7 章 LangSmith を使った RAG アプリケーションの評価               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GenerativeAgents/agent-book/blob/main/chapter07/notebook.ipynb) |
| 第 8 章 AI エージェントとは                                         | -                                                                                                                                                                                     |
| 第 9 章 LangGraph で作る AI エージェント実践入門                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GenerativeAgents/agent-book/blob/main/chapter09/notebook.ipynb) |
| 第 10 章 要件定義書生成 AI エージェントの開発                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GenerativeAgents/agent-book/blob/main/chapter10/notebook.ipynb) |
| 第 11 章 エージェントデザインパターン                               | -                                                                                                                                                                                     |
| 第 12 章 LangChain/LangGraph で実装するエージェントデザインパターン | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GenerativeAgents/agent-book/blob/main/chapter12/notebook.ipynb) |

## 動作確認環境

本書のソースコードは以下の環境・バージョンで動作確認しました。

- Google Colab
- Python 3.10.12

Python パッケージの動作確認済みバージョンは、各章のディレクトリの requirements.txt を参照してください。

## 既知のエラー

### 「7.4 Ragas による合成テストデータの生成」における RateLimitError

「7.4 Ragas による合成テストデータの生成」において、gpt-4o を使用すると OpenAI API の Usage tier 次第で RateLimitError が発生することが報告されています。

OpenAI API の Usage tier については公式ドキュメントの以下のページを参照してください。

https://platform.openai.com/docs/guides/rate-limits/usage-tiers

このエラーが発生した場合は、以下のどちらかの対応を実施してください。

1. 同じ Tier でも gpt-4o よりレートリミットの高い gpt-4o-mini を使用する
   - この場合、生成される合成テストデータの品質は低くなることが想定されます
2. 課金などにより Tier を上げる
   - Tier 2 で RateLimitError が発生しないことを確認済みです (2024 年 10 月 31 日時点)

## 書籍の誤り・エラーについて

書籍の誤り（誤字など）や、発生したエラーについては、GitHub の Issue からご連絡ください。

https://github.com/GenerativeAgents/agent-book/issues

## 正誤表

[正誤表](./errata.md)

## リンク

- [技術評論社](https://gihyo.jp/book/2024/978-4-297-14530-9)
- [Amazon](https://www.amazon.co.jp/dp/4297145308)
