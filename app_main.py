# app_main.py
from dotenv import load_dotenv, find_dotenv
from logging_setup import configure_logging, get_logger
import os, pathlib, datetime as dt
from app_graph import build_app_graph

load_dotenv(find_dotenv(), override=True)
configure_logging()
log = get_logger("app")

# The output directory is configurable, the default is ./output.
OUTPUT_DIR = pathlib.Path(os.getenv("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log.info("Program started | run_id=%s | output_dir=%s", run_id, OUTPUT_DIR.resolve())
    graph = build_app_graph()

    user_input = input("Please enter your request: ")
    state = {"user_input": user_input, "run_id": run_id, "output_dir": str(OUTPUT_DIR)}
    result_state = graph.invoke(state)

    print("\n=== FINAL OUTPUT ===")
    print(result_state.get("final_answer"))

    # —— Clearly state the location of the product (if it exists).——
    artifacts = []
    for k in ["qt_last_csv", "excel_path", "pr_curve", "f1_curve", "roc_curve"]:
        if result_state.get(k):
            artifacts.append(f"{k}: {result_state[k]}")
    if artifacts:
        print("\n[Artifacts]")
        print("\n".join(artifacts))
    else:
        print("\n[Artifacts] (none) — this run did not persist files.")

    log.info("Program finished | run_id=%s", run_id)

if __name__ == "__main__":
    main()

