import io
import operator
import os
from typing import Annotated, Any, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from PIL import Image
from pydantic import BaseModel, Field

# .envファイルから環境変数を読み込む
load_dotenv()  # OPENAI_API_KEY は必須。LangSmith系環境変数はお好みで


# ペルソナを表すデータモデル
class Persona(BaseModel):
    name: str = Field(..., description="ペルソナの名前")
    background: str = Field(..., description="ペルソナの持つ背景")


# ペルソナのリストを表すデータモデル
class Personas(BaseModel):
    personas: list[Persona] = Field(default_factory=list, description="ペルソナのリスト")


# インタビュー内容を表すデータモデル
class Interview(BaseModel):
    persona: Persona = Field(..., description="インタビュー対象のペルソナ")
    question: str = Field(..., description="インタビューでの質問")
    answer: str = Field(..., description="インタビューでの回答")


# インタビュー結果のリストを表すデータモデル
class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list,
        description="インタビュー結果のリスト",
    )


# 評価の結果を表すデータモデル
class EvaluationResult(BaseModel):
    reason: str = Field(..., description="判断の理由")
    is_sufficient: bool = Field(..., description="情報が十分かどうか")


class InterviewState(BaseModel):
    """要件定義生成AIエージェントのステート"""

    user_request: str = Field(..., description="ユーザーからのリクエスト")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list,
        description="生成されたペルソナのリスト",
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list,
        description="実施されたインタビューのリスト",
    )
    requirements_doc: str = Field(default="", description="生成された要件定義")
    iteration: int = Field(default=0, description="ペルソナ生成とインタビューの反復回数")
    is_information_sufficient: bool = Field(
        default=False,
        description="情報が十分かどうか",
    )


class PersonaGenerator:
    """ペルソナを生成するクラス"""

    def __init__(self, llm: ChatOpenAI, k: int = 5) -> None:
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        # プロンプトテンプレートを定義
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                (
                    "system",
                    "あなたはユーザーインタビュー用の多様なペルソナを作成する専門家です。",
                ),
                (
                    "human",
                    f"""\
以下のユーザーリクエストに関するインタビュー用に、{self.k}人の多様なペルソナを生成してください。

ユーザーリクエスト: {user_request}

各ペルソナには名前と簡単な背景を含めてください。年齢、性別、職業、技術的専門知識において多様性を確保してください。""",
                ),
            ]
        )
        # ペルソナ生成のためのチェーンを作成
        chain = prompt | self.llm
        # ペルソナを生成
        return chain.invoke({"user_request": user_request})  # type: ignore


class InterviewConductor:
    """ペルソナに対して、インタビューを実施するクラス"""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        # 質問を生成
        questions = self._generate_questions(
            user_request=user_request,
            personas=personas,
        )
        # 回答を生成
        answers = self._generate_answers(
            personas=personas,
            questions=questions,
        )
        # 質問と回答の組み合わせからインタビューリストを作成
        interviews = self._create_interviews(
            personas=personas,
            questions=questions,
            answers=answers,
        )
        # インタビュー結果を返す
        return InterviewResult(interviews=interviews)

    def _generate_questions(
        self,
        user_request: str,
        personas: list[Persona],
    ) -> list[str]:
        """各ペルソナへの質問を生成します

        Args:
            user_request (str): taskで入力した内容。e.g., 'スマートフォン向けの健康管理アプリを開発したい'
            personas (list[Persona]): generate_personasで生成したペルソナのリスト
                persona
                    - persona.name: e.g., '山田美咲'
                    - persona_background: e.g., '28歳女性、看護師。忙しい仕事の合間に健康管理を行いたいと考えているが、時間がない。技術にはあまり詳しくないが、使いやすいアプリを求めている。'

        Returns:
            list[str]: 各ペルソナへの質問リスト
                e.g.,
                - '鈴木太郎さんが健康管理アプリを使用する際に、特にどのような機能やサポートがあれば、使いやすさや効果を感じられると思いますか？',
                - '山田美咲さんが忙しい仕事の合間に健康管理を行うために、特にどのような機能（例：食事記録、運動トラッキング、ストレス管理など）が最も役立つと感じますか？',
                - '鈴木太郎さんが健康管理アプリを使用する際に、特にどのような機能やサポートがあれば、使いやすさや効果を感じられると思いますか？',
        """
        # 質問生成のためのプロンプトを定義
        question_prompt = ChatPromptTemplate.from_messages(
            messages=[
                (
                    "system",
                    "あなたはユーザー要件に基づいて適切な質問を生成する専門家です。",
                ),
                (
                    "human",
                    """\
以下のペルソナに関連するユーザーリクエストについて、1つの質問を生成してください。

ユーザーリクエスト: {user_request}

ペルソナ: {persona_name} - {persona_background}

質問は具体的で、このペルソナの視点から重要な情報を引き出すように設計してください。""",
                ),
            ]
        )
        # 質問生成のためのチェーンを作成
        question_chain = question_prompt | self.llm | StrOutputParser()

        # 各ペルソナに対する質問クエリを作成
        question_queries = [
            {
                # task で入力した内容
                "user_request": user_request,
                # e.g., '山田美咲'
                "persona_name": persona.name,
                # e.g., '28歳女性、看護師。忙しい仕事の合間に健康管理を行いたいと考えているが、時間がない。技術にはあまり詳しくないが、使いやすいアプリを求めている。'
                "persona_background": persona.background,
            }
            for persona in personas
        ]
        # 質問をバッチ処理で生成
        return question_chain.batch(question_queries)

    def _generate_answers(self, personas: list[Persona], questions: list[str]) -> list[str]:
        """各ペルソナが質問に回答します

        Args:
            personas (list[Persona]): ペルソナ。リストの長さは、最初に引数で入力するkの数と同じです。
            questions (list[str]): 各ペルソナへの質問リスト

        Returns:
            list[str]: 回答のリスト
        """
        # 回答生成のためのプロンプトを定義
        answer_prompt = ChatPromptTemplate.from_messages(
            messages=[
                (
                    "system",
                    "あなたは以下のペルソナとして回答しています: {persona_name} - {persona_background}",
                ),
                ("human", "質問: {question}"),
            ]
        )
        # 回答生成のためのチェーンを作成
        answer_chain = answer_prompt | self.llm | StrOutputParser()

        # 各ペルソナに対する回答クエリを作成
        answer_queries = [
            {
                "persona_name": persona.name,
                "persona_background": persona.background,
                "question": question,
            }
            for persona, question in zip(personas, questions)
        ]
        # 回答をバッチ処理で生成
        return answer_chain.batch(answer_queries)

    def _create_interviews(
        self,
        personas: list[Persona],
        questions: list[str],
        answers: list[str],
    ) -> list[Interview]:
        """ペルソナ毎に質問と回答の組み合わせからインタビューオブジェクトを作成

        Args:
            personas (list[Persona]): ペルソナ
            questions (list[str]): ペルソナへの質問
            answers (list[str]): ペルソナからの回答

        Returns:
            list[Interview]: インタビューオブジェクトのリスト
        """
        return [
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(
                personas,
                questions,
                answers,
            )
        ]


class InformationEvaluator:
    """情報が十分集まったかどうかを評価するクラス"""

    def __init__(self, llm: ChatOpenAI) -> None:
        # 評価結果は、EvaluationResult の形で格納
        self.llm = llm.with_structured_output(schema=EvaluationResult)

    def run(self, user_request: str, interviews: list[Interview]) -> EvaluationResult:
        """情報が十分集まったかどうかを評価します。

        Args:
            user_request (str): ユーザからの要望. 実質"--task"で入力した文字列 e.g., "作成したいアプリケーションについて記載してください"
            interviews (list[Interview]): インタビューの結果

        Returns:
            EvaluationResult: 情報が十分集まったかどうか. is_sufficient: bool があるのでそこに格納
        """

        # プロンプトを定義
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                (
                    "system",
                    "あなたは包括的な要件文書を作成するための情報の十分性を評価する専門家です。",
                ),
                (
                    "human",
                    """\
以下のユーザーリクエストとインタビュー結果に基づいて、包括的な要件文書を作成するのに十分な情報が集まったかどうかを判断してください。

ユーザーリクエスト: {user_request}

インタビュー結果:\n{interview_results}",
                    """,
                ),
            ]
        )
        # 情報の十分性を評価するチェーンを作成
        chain = prompt | self.llm
        # 評価結果を返す
        return chain.invoke(
            input={
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"""\
ペルソナ: {i.persona.name} - {i.persona.background}
質問: {i.question}
回答: {i.answer}
"""
                    for i in interviews
                ),
            }
        )  # type: ignore


# 要件定義書を生成するクラス
class RequirementsDocumentGenerator:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        # プロンプトを定義
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                (
                    "system",
                    "あなたは収集した情報に基づいて要件文書を作成する専門家です。",
                ),
                (
                    "human",
                    """
以下のユーザーリクエストと複数のペルソナからのインタビュー結果に基づいて、要件文書を作成してください。

ユーザーリクエスト: {user_request}
インタビュー結果:
{interview_results}

要件文書には以下のセクションを含めてください:
1. プロジェクト概要
2. 主要機能
3. 非機能要件
4. 制約条件
5. ターゲットユーザー
6. 優先順位
7. リスクと軽減策

出力は必ず日本語でお願いします。

要件文書:""",
                ),
            ]
        )
        # 要件定義書を生成するチェーンを作成
        chain = prompt | self.llm | StrOutputParser()
        # 要件定義書を生成
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"""\
ペルソナ: {i.persona.name} - {i.persona.background}
質問: {i.question}
回答: {i.answer}
"""
                    for i in interviews
                ),
            }
        )


# 要件定義書生成AIエージェントのクラス
class DocumentationAgent:
    def __init__(self, llm: ChatOpenAI, k: Optional[int] = None) -> None:
        # 各種ジェネレータの初期化
        if k is None:
            self.persona_generator = PersonaGenerator(llm=llm)
        else:
            self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.information_evaluator = InformationEvaluator(llm=llm)
        self.requirements_generator = RequirementsDocumentGenerator(llm=llm)

        # グラフの作成
        self.graph = self._create_graph()
        self.draw_graph()

    def draw_graph(self) -> None:
        """グラフ構造を.pngファイルとして出力します。"""

        byte_stream = io.BytesIO(self.graph.get_graph().draw_mermaid_png())
        image = Image.open(byte_stream)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image.save(f"{current_dir}/{self.__class__.__name__}.png")

        print(f"graph structure image: {current_dir}/{self.__class__.__name__}.png")

    def _create_graph(self) -> CompiledStateGraph:
        # グラフの初期化
        workflow = StateGraph(InterviewState)

        # 各ノードの追加
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("evaluate_information", self._evaluate_information)
        workflow.add_node("generate_requirements", self._generate_requirements)

        # エントリーポイントの設定
        workflow.set_entry_point(key="generate_personas")

        # ノード間のエッジの追加
        workflow.add_edge(start_key="generate_personas", end_key="conduct_interviews")
        workflow.add_edge(start_key="conduct_interviews", end_key="evaluate_information")

        # 条件付きエッジの追加
        workflow.add_conditional_edges(
            source="evaluate_information",
            path=lambda state: not state.is_information_sufficient and state.iteration < 5,
            path_map={True: "generate_personas", False: "generate_requirements"},
        )
        # ノード間のエッジの追加
        workflow.add_edge(start_key="generate_requirements", end_key=END)

        # グラフのコンパイル
        return workflow.compile()

    def _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        # ペルソナの生成
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1,
        }

    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        # インタビューの実施
        new_interviews: InterviewResult = self.interview_conductor.run(
            state.user_request,
            state.personas[-5:],
        )
        return {"interviews": new_interviews.interviews}

    def _evaluate_information(self, state: InterviewState) -> dict[str, Any]:
        # 情報の評価
        evaluation_result: EvaluationResult = self.information_evaluator.run(
            state.user_request,
            state.interviews,
        )
        return {
            "is_information_sufficient": evaluation_result.is_sufficient,
            "evaluation_reason": evaluation_result.reason,
        }

    def _generate_requirements(self, state: InterviewState) -> dict[str, Any]:
        # 要件定義書の生成
        requirements_doc: str = self.requirements_generator.run(
            state.user_request,
            state.interviews,
        )
        return {"requirements_doc": requirements_doc}

    def run(self, user_request: str) -> str:
        # 初期状態の設定
        initial_state = InterviewState(user_request=user_request)
        # グラフの実行
        final_state = self.graph.invoke(initial_state)
        # 最終的な要件定義書の取得
        return final_state["requirements_doc"]


# 実行方法:
# poetry run python -m documentation_agent.main --task "ユーザーリクエストをここに入力してください"
# 実行例）
# poetry run python -m documentation_agent.main --task "スマートフォン向けの健康管理アプリを開発したい"
#
# devcontaier+uvの場合
# uv run python /workspaces/agent-book/chapter10/documentation_agent/main.py --task "スマートフォン向けの健康管理アプリを開発したい"
def main() -> None:
    import argparse

    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description="ユーザー要求に基づいて要件定義を生成します")
    # "task"引数を追加
    parser.add_argument(
        "--task",
        type=str,
        help="作成したいアプリケーションについて記載してください",
    )
    # "k"引数を追加
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="生成するペルソナの人数を設定してください（デフォルト:5）",
    )
    # コマンドライン引数を解析
    args = parser.parse_args()

    # ChatOpenAIモデルを初期化
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)  # 良い結果が欲しい場合は、gpt-4o, o1-mini, o1-preview の利用を検討して下さい。
    # 要件定義書生成AIエージェントを初期化
    agent = DocumentationAgent(llm=llm, k=args.k)
    # エージェントを実行して最終的な出力を取得
    final_output = agent.run(user_request=args.task)

    # 最終的な出力を表示
    print(final_output)


if __name__ == "__main__":
    main()
