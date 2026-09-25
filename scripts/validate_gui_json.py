from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import ImageGrab

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from covermorph.project import add_input_records, parse_input_file_detailed  # noqa: E402
from covermorph.quick_gui import LYRICS_MODE, QuickCoverApp  # noqa: E402


def main(source: Path) -> None:
    output = ROOT / "validation_results" / "gui_json"
    output.mkdir(parents=True, exist_ok=True)
    parsed = parse_input_file_detailed(source)
    records = list(parsed["records"])
    for index, record in enumerate(records):
        record.selected = index == 0

    app = QuickCoverApp(ROOT)
    add_input_records(app.project, records)
    app.records = [records[0]]
    app.mode.set(LYRICS_MODE)
    app.count.set("1")
    app.output.set(str(output))
    app.full_path_var.set(str(source))
    app.file_var.set(f"{app._short_path(source)} · 선택 1/{len(records)}곡")
    app.input_box.delete("1.0", "end")
    app.input_box.insert("1.0", records[0].lyrics or records[0].theme_mood)
    app.title_var.set("")
    app.geometry("1366x768+0+0")
    stages: list[str] = []
    events: list[dict[str, object]] = []
    deadline = time.monotonic() + 600

    def drive() -> None:
        state = app.task.state
        if not stages or stages[-1] != state:
            stages.append(state)
            events.append(
                {
                    "state": state,
                    "description": app.task.description,
                    "analyzed_songs": app.task.analyzed_songs,
                    "step": app.task.step,
                    "steps": app.task.steps,
                }
            )
        running = app.worker is not None and app.worker.is_alive()
        if time.monotonic() > deadline:
            app.cancel()
            raise TimeoutError("JSON GUI validation timed out")
        if app.task.terminal and not running:
            app.update_idletasks()
            ImageGrab.grab().save(output / "desktop_after_json_generation.png")
            report = {
                "source_copy_used_without_modification": str(source),
                "selected_song": records[0].title,
                "final_state": app.task.state,
                "completed": app.task.completed,
                "requested": app.task.requested,
                "stages": stages,
                "events": events,
                "actual_prompt": app.task.input_snapshot.get("actual_prompt", ""),
                "title_after_planning": app.title_var.get(),
                "candidate_ids": list(app.project.selected_candidate_ids),
            }
            (output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
            app.destroy()
            return
        app.after(100, drive)

    app.generate()
    app.after(100, drive)
    app.mainloop()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
