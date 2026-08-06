"""Qwen 로컬 LLM 전용 프로세스의 생성, 호출 및 종료를 관리한다."""

from __future__ import annotations

import atexit
import multiprocessing
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any


_SHUTDOWN_TIMEOUT_SECONDS = 30.0


def _llm_worker(connection: Connection) -> None:
    """자식 프로세스 안에서만 모델과 CUDA 컨텍스트를 생성한다."""
    llm: Any | None = None
    try:
        # 부모 프로세스가 torch/CUDA를 초기화하지 않도록 자식에서 지연 import한다.
        from backend.ai_simulation_core.llm.qwen_model import LLM

        llm = LLM()
        connection.send(("ready", None))

        while True:
            command, payload = connection.recv()
            if command == "shutdown":
                break
            if command != "generate":
                connection.send(("error", f"알 수 없는 LLM 명령: {command}"))
                continue

            try:
                response = llm.generate(payload)
            except BaseException:
                connection.send(("error", traceback.format_exc()))
            else:
                connection.send(("response", response))
    except EOFError:
        # 부모가 먼저 종료되면 파이프가 닫히므로 그대로 모델을 정리한다.
        pass
    except BaseException:
        try:
            connection.send(("startup_error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            if llm is not None:
                llm.close()
        finally:
            connection.close()


class LLMProcessClient:
    """부모 프로세스에서 전용 LLM 프로세스와 통신하는 클라이언트."""

    def __init__(
        self, *, worker_target: Callable[[Connection], None] = _llm_worker
    ) -> None:
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        self._connection = parent_connection
        self._process = context.Process(
            target=worker_target,
            args=(child_connection,),
            name="qwen-local-llm",
            daemon=True,
        )
        self._closed = False

        try:
            self._process.start()
            child_connection.close()
            status, payload = self._connection.recv()
        except BaseException:
            child_connection.close()
            self.close()
            raise

        if status != "ready":
            self.close()
            raise RuntimeError(f"LLM 프로세스 시작 실패:\n{payload}")

    @property
    def pid(self) -> int | None:
        return self._process.pid

    def generate(self, prompt: str) -> str:
        if self._closed or not self._process.is_alive():
            raise RuntimeError("LLM 프로세스가 종료되어 있습니다.")

        self._connection.send(("generate", prompt))
        status, payload = self._connection.recv()
        if status == "response":
            return payload
        raise RuntimeError(f"LLM 생성 실패:\n{payload}")

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        if self._process.is_alive():
            try:
                self._connection.send(("shutdown", None))
            except (BrokenPipeError, EOFError, OSError):
                pass

            self._process.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)

        # 모델 종료가 멈춰도 프로세스를 남겨 VRAM을 점유하지 않도록 보장한다.
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)

        if self._process.is_alive():
            self._process.kill()
            self._process.join()

        self._connection.close()


_llm: LLMProcessClient | None = None


def get_llm() -> LLMProcessClient:
    global _llm

    if _llm is None:
        print("[LLM] 전용 프로세스에서 모델 가져오는 중")
        _llm = LLMProcessClient()
        print(f"[LLM] 모델 가져오기 완료 (PID: {_llm.pid})")

    return _llm


def run_llm(prompt: str) -> str:
    return get_llm().generate(prompt)


def unload_llm() -> None:
    """LLM 프로세스를 종료해 CUDA 컨텍스트와 VRAM을 함께 반환한다."""
    global _llm

    llm = _llm
    _llm = None
    if llm is None:
        return

    print(f"[LLM] 모델 프로세스 종료 중 (PID: {llm.pid})")
    llm.close()
    print("[LLM] 모델 프로세스 종료 완료")


atexit.register(unload_llm)
