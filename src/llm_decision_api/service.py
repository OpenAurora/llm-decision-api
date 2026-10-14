"""Concurrent, independent question execution with one request deadline."""

import asyncio
from time import monotonic

from .backend import ModelBackend
from .config import Settings
from .errors import ServiceError
from .prompt import prepare_question
from .schemas import ChoiceQuestion, DecisionResponse, Usage
from .scoring import to_answer


class DecisionService:
    def __init__(self, settings: Settings, backend: ModelBackend):
        self.settings = settings
        self.backend = backend
        self._closed = False
        self.verified = False

    async def verify_backend(self):
        options = {f"candidate_{index}": f"候选编号 {index}" for index in range(self.settings.max_options)}
        prepared = prepare_question(
            {"target": 0},
            ChoiceQuestion(type="choice", instructions="选择与 target 编号一致的候选。", criteria=options),
        )
        try:
            await asyncio.wait_for(self.backend.score(prepared), timeout=self.settings.timeout_seconds)
        except asyncio.TimeoutError:
            raise ServiceError(504, "startup_probe_timeout", "模型能力检查超时。") from None
        self.verified = True

    async def evaluate(self, body) -> DecisionResponse:
        deadline = monotonic() + self.settings.timeout_seconds

        def remaining_time():
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ServiceError(504, "decision_timeout", "本次决策超过总时限。")
            return remaining

        if self._closed:
            raise ServiceError(503, "service_closed", "服务已关闭。")
        # Compilation and all input validation precede any model call.
        prepared = []
        for qid, question in body.questions.items():
            remaining_time()
            prepared.append((qid, prepare_question(body.state, question)))
            remaining_time()
        if any(len(item.labels) > self.settings.max_options for _, item in prepared):
            raise ServiceError(422, "too_many_options", "候选数量超过服务配置上限。")

        async def one(qid, item):
            try:
                remaining_time()
                scored = await self.backend.score(item)
                remaining_time()
                return qid, to_answer(item, scored), scored
            except ServiceError as exc:
                exc.question_id = qid
                raise

        tasks = [asyncio.create_task(one(qid, item)) for qid, item in prepared]
        try:
            remaining = remaining_time()
            results = await asyncio.wait_for(asyncio.gather(*tasks), remaining)
        except asyncio.TimeoutError:
            raise ServiceError(504, "decision_timeout", "本次决策超过总时限。") from None
        finally:
            # A failed or disconnected request must not leave model calls running.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        response = DecisionResponse(
            model=self.settings.model,
            answers={qid: answer for qid, answer, _ in results},
            usage=Usage(
                input_tokens=sum(scored.input_tokens for _, _, scored in results),
                output_tokens=sum(scored.output_tokens for _, _, scored in results),
            ),
        )
        remaining_time()
        return response

    async def close(self):
        self._closed = True
        await self.backend.close()
