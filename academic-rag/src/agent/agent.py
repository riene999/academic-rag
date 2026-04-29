import json
from threading import RLock
from typing import List, Dict, Any
from loguru import logger
from openai import OpenAI

from src.cache.redis_cache import RedisCache, redis_enabled
from src.rag.pipeline import RAGPipeline
from src.utils.config import LLMConfig


# 工具定义（OpenAI Function Calling格式）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "在已索引的学术论文中语义检索与查询相关的内容片段",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询词，应该是学术性的具体问题或概念"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的相关片段数量，默认3",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_overview",
            "description": "获取论文的整体概览，包括研究问题、方法、结论",
            "parameters": {
                "type": "object",
                "properties": {
                    "aspect": {
                        "type": "string",
                        "enum": ["research_question", "methodology", "results", "conclusion"],
                        "description": "要了解的论文方面"
                    }
                },
                "required": ["aspect"]
            }
        }
    }
]

AGENT_SYSTEM_PROMPT = """你是一个学术论文分析Agent。你可以使用以下工具来帮助用户：
1. search_papers: 检索论文中的相关内容
2. get_paper_overview: 获取论文的概览信息

工作原则：
- 对于复杂问题，先分解，再逐步检索
- 每次工具调用后，基于结果决定是否需要继续检索
- 综合多次检索结果给出完整回答
- 如果检索结果不足，明确告知用户
- 如果用户追问“它/这个方法/上面的问题”等上下文相关内容，优先结合会话记忆理解指代"""


class ConversationMemory:
    """按 session_id 保存最近多轮用户问题和最终回答。"""

    def __init__(self, max_turns: int = 6, redis_config=None):
        self.max_turns = max_turns
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._lock = RLock()
        self._cache = (
            RedisCache[str, List[Dict[str, str]]](
                max_size=10000,
                ttl_seconds=24 * 60 * 60,
                redis_config=redis_config,
                value_codec="json",
            )
            if redis_enabled()
            else None
        )

    def _session_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            if self._cache is not None:
                return list(self._cache.get(self._session_key(session_id)) or [])
            return list(self._sessions.get(session_id, []))

    def add_turn(self, session_id: str, user_query: str, assistant_answer: str) -> None:
        with self._lock:
            if self._cache is not None:
                history = self.get_messages(session_id)
            else:
                history = self._sessions.setdefault(session_id, [])
            history.extend([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": assistant_answer},
            ])
            max_messages = self.max_turns * 2
            if len(history) > max_messages:
                history = history[-max_messages:]
            if self._cache is not None:
                self._cache.set(self._session_key(session_id), history)
            else:
                self._sessions[session_id] = history

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            if self._cache is not None:
                if session_id is None:
                    self._cache.clear("session:*")
                    return
                self._cache.delete(self._session_key(session_id))
                return
            if session_id is None:
                self._sessions.clear()
                return
            self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            if self._cache is not None:
                return self._cache.size("session:*")
            return len(self._sessions)


class PaperAgent:
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        llm_config: LLMConfig,
        memory_max_turns: int = 6,
    ):
        self.rag = rag_pipeline
        self.client = OpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )
        self.llm_config = llm_config
        self.memory = ConversationMemory(
            max_turns=memory_max_turns,
            redis_config=getattr(rag_pipeline.config, "redis", None),
        )

    def run(
        self,
        user_query: str,
        max_iterations: int = 5,
        session_id: str = "default",
        use_memory: bool = True,
    ) -> str:
        """
        ReAct Agent主循环
        LLM决策 -> 工具调用 -> 观察结果 -> 继续决策 -> 最终回答
        """
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        ]
        if use_memory:
            messages.extend(self.memory.get_messages(session_id))
        messages.append({"role": "user", "content": user_query})

        for iteration in range(max_iterations):
            logger.info(f"Agent迭代 {iteration + 1}/{max_iterations}")

            response = self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,  # Agent决策用低温，保证稳定
            )

            message = response.choices[0].message
            messages.append(message)

            # 没有工具调用，直接返回最终回答
            if not message.tool_calls:
                logger.info("Agent完成，返回最终回答")
                answer = message.content or ""
                if use_memory:
                    self.memory.add_turn(session_id, user_query, answer)
                return answer

            # 执行工具调用
            for tool_call in message.tool_calls:
                tool_result = self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

        # 超过最大迭代次数，强制生成回答
        logger.warning("达到最大迭代次数，强制生成回答")
        final_response = self.client.chat.completions.create(
            model=self.llm_config.model,
            messages=messages + [{"role": "user", "content": "请基于以上信息给出最终回答"}],
            temperature=self.llm_config.temperature,
        )
        answer = final_response.choices[0].message.content or ""
        if use_memory:
            self.memory.add_turn(session_id, user_query, answer)
        return answer

    def clear_memory(self, session_id: str | None = None) -> None:
        """清空指定会话或全部会话记忆。"""
        self.memory.clear(session_id)

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """执行工具调用，返回结果字符串"""
        logger.info(f"调用工具: {tool_name}, 参数: {args}")

        if tool_name == "search_papers":
            query = args["query"]
            top_k = args.get("top_k", 3)
            chunks = self.rag.retrieve_chunks(query, top_k=top_k)

            if not chunks:
                return "未找到相关内容"

            results = []
            for chunk in chunks:
                results.append(
                    f"[相似度: {chunk.score:.3f}] {chunk.document.content[:300]}..."
                )
            return "\n\n".join(results)

        elif tool_name == "get_paper_overview":
            aspect = args["aspect"]
            aspect_queries = {
                "research_question": "研究问题 研究动机 为什么",
                "methodology": "方法 模型 算法 框架",
                "results": "实验结果 性能 准确率 效果",
                "conclusion": "结论 总结 贡献 未来工作",
            }
            query = aspect_queries.get(aspect, aspect)
            chunks = self.rag.retrieve_chunks(query, top_k=3)

            if not chunks:
                return f"未找到关于{aspect}的相关内容"

            return "\n\n".join([c.document.content[:400] for c in chunks])

        return f"未知工具: {tool_name}"
