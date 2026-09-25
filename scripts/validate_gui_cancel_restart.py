from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import ImageGrab

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from covermorph.quick_gui import DIRECT_MODE, QuickCoverApp  # noqa: E402


def main() -> None:
    output = ROOT / "validation_results" / "gui_cancel_restart"
    output.mkdir(parents=True, exist_ok=True)
    app = QuickCoverApp(ROOT)
    app.mode.set(DIRECT_MODE)
    app.count.set("1")
    app.output.set(str(output))
    app.input_box.delete("1.0", "end")
    app.input_box.insert(
        "1.0", "A quiet blue-hour riverside bus stop after rain, cinematic square album cover, no text"
    )
    app.title_var.set("다시 시작한 밤")
    app.subtitle_var.set("再開 · Restarted")
    app.label_var.set("CoverMorph Records")
    events: list[dict] = []
    phase = {"value": "starting"}
    deadline = time.monotonic() + 360

    def record() -> None:
        item = {
            "state": app.task.state,
            "completed": app.task.completed,
            "current_candidate": app.task.current_candidate,
            "step": app.task.step,
            "steps": app.task.steps,
            "description": app.task.description,
        }
        if not events or events[-1] != item:
            events.append(item)

    def drive() -> None:
        record()
        if time.monotonic() > deadline:
            raise TimeoutError("GUI cancel/restart validation timed out")
        worker_running = app.worker is not None and app.worker.is_alive()
        if phase["value"] == "starting" and app.task.state == "이미지 생성" and app.task.step >= 5:
            app.cancel()
            phase["value"] = "cancelling"
        elif phase["value"] == "cancelling" and app.task.state == "중단됨" and not worker_running:
            phase["value"] = "restarting"
            app.retry()
        elif phase["value"] == "restarting" and app.task.state == "완료" and not worker_running:
            app.save_cover()
            app.update_idletasks()
            x, y = app.winfo_rootx(), app.winfo_rooty()
            ImageGrab.grab((x, y, x + app.winfo_width(), y + app.winfo_height())).save(
                output / "completed_after_restart.png"
            )
            report = {
                "actual_gpu": True,
                "cancel_requested_at_or_after_step": 5,
                "final_state": app.task.state,
                "completed": app.task.completed,
                "requested": app.task.requested,
                "failed_indices": app.task.failed_indices,
                "events": events,
                "candidate_ids": list(app.project.selected_candidate_ids),
                "output_directory": str(output),
            }
            (output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            app.destroy()
            return
        app.after(100, drive)

    app.generate()
    app.after(100, drive)
    app.mainloop()


if __name__ == "__main__":
    main()
