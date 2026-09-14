from scripts.validate_3b1 import inference_state


def test_validation_marks_cuda_unavailable_as_not_run() -> None:
    assert inference_state({"cuda": False, "status": "gpu_unavailable"}) == ("not_run", "gpu_unavailable")


def test_validation_marks_missing_packages_as_not_run() -> None:
    assert inference_state({"cuda": True, "status": "package_missing"}) == ("not_run", "package_missing")
