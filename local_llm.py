"""
День 26. Обращение к локальной LLM через HTTP API Ollama.

Без внешних библиотек — только стандартный Python (urllib).

Что делает:
  1. проверяет, что сервер Ollama запущен локально (GET /api/version);
  2. показывает установленные модели (GET /api/tags);
  3. отправляет запросы разной сложности (POST /api/chat) из tests.json;
  4. показывает, где работает модель — на GPU или процессоре (GET /api/ps),
     и скорость генерации (токенов в секунду);
  5. автоматически проверяет ответы: факт, число, КОД (запускается на тестах), JSON.

Запуск:
    python local_llm.py                          # 4 запроса из tests.json
    python local_llm.py --model qwen2.5:3b       # другая модель
    python local_llm.py --ask "Что такое RAG?"   # один свой вопрос
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URL = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
DEFAULT_MODEL = "qwen2.5:7b"


class OllamaError(RuntimeError):
    pass


def api(url: str, path: str, payload: dict = None, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url + path, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise OllamaError(f"HTTP {e.code} на {path}: {body[:300]}") from e
    except urllib.error.URLError as e:
        raise OllamaError(f"Сервер Ollama недоступен по адресу {url}: {e.reason}. "
                          f"Запустите Ollama (значок в трее или команда «ollama serve»).") from e


def strip_think(text: str) -> str:
    """Некоторые модели (Qwen3, DeepSeek-R1) пишут рассуждения в <think>…</think> — убираем."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def chat(url: str, model: str, prompt: str) -> dict:
    started = time.perf_counter()
    r = api(url, "/api/chat", {"model": model, "stream": False, "options": {"temperature": 0.2},
                               "messages": [{"role": "user", "content": prompt}]})
    wall = time.perf_counter() - started
    eval_s = (r.get("eval_duration") or 0) / 1e9
    return {
        "answer": strip_think(r.get("message", {}).get("content", "")),
        "wall_s": round(wall, 1),
        "load_s": round((r.get("load_duration") or 0) / 1e9, 1),
        "prompt_tokens": r.get("prompt_eval_count") or 0,
        "output_tokens": r.get("eval_count") or 0,
        "tok_per_s": round((r.get("eval_count") or 0) / eval_s, 1) if eval_s else None,
    }


def placement(url: str, model: str) -> str:
    """Где сейчас загружена модель: доля в видеопамяти по данным /api/ps."""
    for m in api(url, "/api/ps").get("models", []):
        if m.get("name") == model or m.get("model") == model:
            size, vram = m.get("size") or 0, m.get("size_vram") or 0
            if not size:
                return "неизвестно"
            share = vram / size
            if share >= 0.99:
                return f"GPU (100% в видеопамяти, {vram / 2**30:.1f} ГБ)"
            if share <= 0.01:
                return f"процессор (0% в видеопамяти, {size / 2**30:.1f} ГБ в RAM)"
            return f"GPU + процессор ({share:.0%} в видеопамяти)"
    return "модель не загружена"


# ---------------------------------------------------------------------------
# Проверки ответов
# ---------------------------------------------------------------------------

UNSAFE = re.compile(r"\b(import\s+(os|sys|subprocess|shutil|socket)|__import__|open\(|exec\(|eval\()")


def extract_code(answer: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", answer, re.DOTALL)
    return textwrap.dedent(m.group(1) if m else answer).strip()


def check_python(answer: str, spec: dict) -> tuple:
    code = extract_code(answer)
    if spec["function"] not in code:
        return False, f"в ответе нет функции {spec['function']}"
    if UNSAFE.search(code):
        return False, "код содержит небезопасные операции — запуск пропущен"
    runner = code + "\n\nimport json as _j\n_cases = _j.loads(" + repr(json.dumps(spec["cases"], ensure_ascii=False)) + ")\n" + \
        f"_res = [[c[0], bool({spec['function']}(c[0])) == c[1]] for c in _cases]\nprint(_j.dumps(_res, ensure_ascii=False))\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(runner)
        try:
            out = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True,
                                 timeout=10, cwd=tmp, encoding="utf-8")
        except subprocess.TimeoutExpired:
            return False, "код завис (больше 10 секунд)"
    if out.returncode != 0:
        return False, "код упал: " + (out.stderr.strip().splitlines() or ["?"])[-1][:150]
    results = json.loads(out.stdout.strip().splitlines()[-1])
    failed = [c for c, ok in results if not ok]
    passed = len(results) - len(failed)
    return not failed, f"тесты {passed}/{len(results)}" + (f", не прошли: {failed}" if failed else "")


def check_json(answer: str, spec: dict) -> tuple:
    m = re.search(r"\{.*\}", answer, re.DOTALL)
    if not m:
        return False, "в ответе нет JSON"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return False, f"JSON не разбирается: {e.msg}"
    bad = [k for k, words in spec["fields"].items()
           if not all(w.lower() in str(obj.get(k, "")).lower() for w in words)]
    return not bad, "все поля верны" if not bad else f"неверные или пустые поля: {', '.join(bad)}"


def check(answer: str, spec: dict) -> tuple:
    if spec["type"] == "contains":
        ok = any(v.lower() in answer.lower() for v in spec["any"])
        return ok, "ожидаемое найдено" if ok else f"нет ни одного из: {', '.join(spec['any'])}"
    if spec["type"] == "python":
        return check_python(answer, spec)
    if spec["type"] == "json":
        return check_json(answer, spec)
    return False, "неизвестный тип проверки"


# ---------------------------------------------------------------------------

def main(args) -> int:
    print(f"Сервер Ollama: {args.url}")
    version = api(args.url, "/api/version", timeout=10).get("version", "?")
    print(f"✅ Сервер отвечает, версия Ollama {version}")
    models = [m["name"] for m in api(args.url, "/api/tags", timeout=10).get("models", [])]
    print(f"Установленные модели: {', '.join(models) or 'нет'}")
    if args.model not in models and f"{args.model}:latest" not in models:
        raise OllamaError(f"Модель {args.model} не установлена. Скачайте: ollama pull {args.model}")

    if args.ask:
        r = chat(args.url, args.model, args.ask)
        print(f"\nВопрос: {args.ask}\n\n{r['answer']}\n")
        print(f"⏱ {r['wall_s']} с · {r['output_tokens']} токенов · {r['tok_per_s']} ток/с · "
              f"работает на: {placement(args.url, args.model)}")
        return 0

    with open(os.path.join(BASE, "tests.json"), encoding="utf-8") as f:
        tests = json.load(f)["tests"]
    print(f"\nМодель: {args.model} · запросов: {len(tests)}\n")
    rows = []
    for t in tests:
        print(f"[{t['id']}/{len(tests)}] {t['level']}: {t['title']} … ", end="", flush=True)
        r = chat(args.url, args.model, t["prompt"])
        ok, note = check(r["answer"], t["check"])
        rows.append({**t, **r, "ok": ok, "note": note})
        print(f"{'✅' if ok else '❌'} {note} · {r['wall_s']} с · {r['tok_per_s']} ток/с")
        if r["load_s"] > 1:
            print(f"      (из них загрузка модели в память: {r['load_s']} с — только при первом запросе)")

    where = placement(args.url, args.model)
    passed = sum(r["ok"] for r in rows)
    speeds = [r["tok_per_s"] for r in rows if r["tok_per_s"]]
    print(f"\n{'=' * 70}\nИТОГ: {passed}/{len(rows)} ответов прошли проверку · модель работает на: {where}")
    if speeds:
        print(f"Скорость генерации: {min(speeds)}–{max(speeds)} токенов/с")
    print("=" * 70)
    for r in rows:
        print(f"\n── [{r['id']}] {r['title']} ({r['level']}) ──\n{r['answer'][:700]}")

    os.makedirs(os.path.join(BASE, "output"), exist_ok=True)
    path = os.path.join(BASE, "output", "local_llm_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Локальная LLM: проверка\n\nOllama {version} · модель `{args.model}` · работает на: {where} · "
                f"{datetime.now():%Y-%m-%d %H:%M}\n\n## Итог\n\n| # | Сложность | Запрос | Проверка | Время, с | Токенов | Ток/с |\n"
                f"|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['id']} | {r['level']} | {r['title']} | {'✅' if r['ok'] else '❌'} {r['note']} | "
                    f"{r['wall_s']} | {r['output_tokens']} | {r['tok_per_s']} |\n")
        f.write(f"\nПрошли проверку: **{passed}/{len(rows)}**\n\n## Ответы\n")
        for r in rows:
            f.write(f"\n### {r['id']}. {r['title']} ({r['level']})\n\n**Запрос:** {r['prompt']}\n\n**Ответ:**\n\n{r['answer']}\n")
    print(f"\nОтчёт: {os.path.relpath(path, BASE)}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Запросы к локальной LLM через HTTP API Ollama")
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--ask", help="задать один свой вопрос")
    try:
        sys.exit(main(ap.parse_args()))
    except OllamaError as e:
        print(f"\n❌ {e}")
        sys.exit(2)
