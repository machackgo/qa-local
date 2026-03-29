import os
import time
import requests
import gradio as gr

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:9044").rstrip("/")

def ask_local(context, question):
    context = (context or "").strip()
    question = (question or "").strip()

    if not context:
        return "Please paste some context/passage first.", ""
    if not question:
        return "Please type a question.", ""

    t0 = time.time()
    try:
        r = requests.post(
            f"{BACKEND_URL}/qa",
            json={"context": context, "question": question},
            timeout=120,
        )
        r.raise_for_status()

        data = r.json()
        if isinstance(data, dict):
            answer = data.get("answer", "") or "(No answer.)"
            meta = data.get("meta", "") or ""
        else:
            answer = str(data)
            meta = "Unexpected response format from backend"

        dt = time.time() - t0
        meta = f"{meta} | UI roundtrip: {dt:.2f}s" if meta else f"UI roundtrip: {dt:.2f}s"
        return answer, meta

    except requests.exceptions.HTTPError:
        return f"Backend HTTP error: {r.status_code} {r.text}", "Request failed"
    except Exception as e:
        return f"Frontend error: {e}", "Request failed"

with gr.Blocks() as demo:
    gr.Markdown("# ❓ Question Answering Assistant (Local mode)")
    gr.Markdown(f"Frontend calls backend: `{BACKEND_URL}`")

    context = gr.Textbox(label="Context / Passage", lines=10)
    question = gr.Textbox(label="Question", lines=2)

    btn = gr.Button("Get Answer")
    answer_box = gr.Textbox(label="Answer", lines=4, interactive=False)
    meta_box = gr.Textbox(label="Run info", interactive=False)

    btn.click(ask_local, [context, question], [answer_box, meta_box])

demo.launch(server_name="0.0.0.0", server_port=7044)
