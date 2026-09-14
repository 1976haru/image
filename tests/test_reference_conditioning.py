"""Model-free tests exercise the real engine with a recording Diffusers boundary."""
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from PIL import Image
from test_generation import FakeEngine, confirmed_scene

from covermorph.generation import (
    GenerationCancelled,
    GenerationConfig,
    GenerationError,
    SDXLTextToImageEngine,
    generate_scene_candidates,
    retry_failed_candidates,
    validate_adapter_directory,
)
from covermorph.project import (
    add_person,
    add_person_reference,
    configure_scene_prompt,
    create_project,
    default_generation_presets,
    load_project,
    prepare_reference_image,
    resolve_project_path,
    save_project_atomic,
)


class RecordingPipeline:
    tokenizer = None
    def __init__(self):
        self.calls = []
        self.scale = None
        self.removed = False
    def set_ip_adapter_scale(self, scale):
        self.scale = scale
    def enable_model_cpu_offload(self):
        pass
    def unload_ip_adapter(self):
        self.removed = True
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        kwargs['callback_on_step_end'](self, 0, 1, {})
        return SimpleNamespace(images=[Image.new('RGB', (kwargs['width'], kwargs['height']))])


@pytest.fixture
def engine(monkeypatch):
    fake_torch = SimpleNamespace(Generator=lambda **kw: SimpleNamespace(manual_seed=lambda seed: seed), cuda=SimpleNamespace(is_available=lambda: False, OutOfMemoryError=MemoryError))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    instance = SDXLTextToImageEngine()
    pipelines = []
    def load(progress=None):
        instance.pipeline = RecordingPipeline()
        pipelines.append(instance.pipeline)
    def load_adapter(config, progress=None):
        instance.ip_adapter_loaded = True
    monkeypatch.setattr(instance, 'load', load)
    monkeypatch.setattr(instance, 'load_ip_adapter', load_adapter)
    return instance, pipelines


def test_real_call_modes_strength_and_image_replacement(engine):
    instance, pipelines = engine
    a, b = Image.new('RGB', (10,10), 'red'), Image.new('RGB', (10,10), 'blue')
    config = GenerationConfig(reference_mode='person', reference_strength=.5)
    instance.generate_one('a', '', config, 1, Event(), reference_image=a)
    config.reference_strength = .8
    instance.generate_one('a', '', config, 1, Event(), reference_image=b)
    assert len(pipelines) == 1
    assert pipelines[0].calls[-1]['ip_adapter_image'] is b
    assert pipelines[0].scale == .8
    config.reference_mode = 'style'
    instance.generate_one('a', '', config, 1, Event(), reference_image=b)
    assert len(pipelines) == 2 and pipelines[0].removed
    config.reference_mode = 'off'
    instance.generate_one('a', '', config, 1, Event())
    assert len(pipelines) == 3 and pipelines[1].removed
    assert 'ip_adapter_image' not in pipelines[2].calls[0]
    assert not instance.last_reference_applied


def test_cancel_without_progress_callback(engine):
    instance, pipelines = engine
    event = Event()
    instance.load()
    original = instance.pipeline.__class__.__call__
    def cancelling(self, **kwargs):
        event.set()
        return original(self, **kwargs)
    instance.pipeline.__class__.__call__ = cancelling
    try:
        with pytest.raises(GenerationCancelled):
            instance.generate_one('a', '', GenerationConfig(), 1, event)
    finally:
        RecordingPipeline.__call__ = original
    assert len(pipelines) == 1


def ref_project(tmp_path):
    project = create_project(tmp_path / 'project', 'refs')
    source = tmp_path / 'input.png'
    Image.new('RGBA', (40,30), (255,0,0,0)).save(source)
    ref = add_person_reference(project, add_person(project, 'A'), source, 'person', 'face')
    return project, ref


def test_alpha_cache_replacement_and_project_move(tmp_path):
    project, ref = ref_project(tmp_path)
    first, meta = prepare_reference_image(project, ref)
    with Image.open(first) as image:
        assert image.getpixel((0,0)) == (255,255,255)
    Image.new('RGB', (40,30), 'blue').save(resolve_project_path(project, ref.path))
    second, changed = prepare_reference_image(project, ref, (0,0,20,20))
    assert first != second and meta['source_sha256'] != changed['source_sha256']
    scene = confirmed_scene()
    scene.structured_request.update(reference_mode='person', reference_image_id=ref.image_id, reference_strength=.8, reference_crop_box=[0,0,20,20])
    project.scenes.append(scene)
    configure_scene_prompt(project, scene, default_generation_presets()[0], refresh=True)
    assert scene.structured_request['reference_image_id'] == ref.image_id
    save_project_atomic(project)
    moved = tmp_path / 'moved'
    project.project_dir.rename(moved)
    loaded = load_project(moved)
    assert loaded.scenes[0].structured_request['reference_strength'] == .8
    assert prepare_reference_image(loaded, loaded.people[0].reference_images[0])[0].is_file()


def test_live_edits_do_not_change_snapshot_or_retry(tmp_path):
    project, ref = ref_project(tmp_path)
    scene = confirmed_scene()
    config = GenerationConfig(reference_mode='person', reference_image_id=ref.image_id, seed=100, candidate_count=2)
    recorder = FakeEngine({101})
    def progress(payload):
        if payload['phase'] == 'candidate_start':
            config.reference_mode = 'off'
            config.seed = 900
            scene.prompt_user = 'changed'
            Image.new('RGB', (40,30), 'blue').save(resolve_project_path(project, ref.path))
    result = generate_scene_candidates(project, scene, recorder, config, Event(), progress)
    assert recorder.seeds == [100,101]
    assert recorder.calls == [('person',.5,True),('person',.5,True)]
    original_hash = project.generation_runs[0]['reference']['processed_sha256']
    retry = FakeEngine()
    retry_failed_candidates(project, scene, retry, config, result.failed_indices, Event())
    assert retry.seeds == [101]
    assert project.generation_runs[-1]['reference']['processed_sha256'] == original_hash
    assert project.generation_runs[-1]['prompt'] == 'a scene'
    assert project.candidates[0].generation_metadata['errors'] == []


def test_corrupt_reference_and_adapter_failure_save_nothing(tmp_path, engine, monkeypatch):
    project, ref = ref_project(tmp_path)
    config = GenerationConfig(reference_mode='person', reference_image_id=ref.image_id)
    instance, pipelines = engine
    def fail(*args):
        raise GenerationError('adapter failure')
    monkeypatch.setattr(instance, 'load_ip_adapter', fail)
    result = generate_scene_candidates(project, confirmed_scene(), instance, config, Event())
    assert result.failed == 1 and not project.candidates and not pipelines[0].calls
    resolve_project_path(project, ref.path).write_text('broken')
    with pytest.raises(Exception, match='damaged'):
        generate_scene_candidates(project, confirmed_scene(), instance, config, Event())
    assert not project.candidates


def test_incomplete_adapter_not_ready(tmp_path: Path):
    (tmp_path/'adapter_ready.json').write_text('{"revision":"fake","files":{}}')
    with pytest.raises(GenerationError, match='incomplete'):
        validate_adapter_directory(tmp_path)


def test_exact_plus_encoder_pair_and_loader_failure_cleanup(monkeypatch):
    from covermorph.generation import DEFAULT_IP_ADAPTER_REVISION, DEFAULT_IP_ADAPTER_WEIGHT
    calls = {}
    class Encoder:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls['encoder'] = (model, kwargs)
            return object()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(CLIPVisionModelWithProjection=Encoder))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(float16='fp16', cuda=SimpleNamespace(is_available=lambda: False)))
    instance = SDXLTextToImageEngine()
    class Pipeline:
        def register_modules(self, **kwargs):
            calls['register'] = kwargs
        def load_ip_adapter(self, model, **kwargs):
            calls['adapter'] = (model, kwargs)
    instance.pipeline = Pipeline()
    instance.load_ip_adapter(GenerationConfig(reference_mode='person'))
    assert calls['encoder'][1]['subfolder'] == 'models/image_encoder'
    assert calls['adapter'][1]['weight_name'] == DEFAULT_IP_ADAPTER_WEIGHT
    assert calls['adapter'][1]['image_encoder_folder'] is None
    assert calls['encoder'][1]['revision'] == calls['adapter'][1]['revision'] == DEFAULT_IP_ADAPTER_REVISION
    def fail(*args, **kwargs):
        raise ValueError('incompatible tensors')
    instance.pipeline.load_ip_adapter = fail
    with pytest.raises(GenerationError, match='fallback is disabled'):
        instance.load_ip_adapter(GenerationConfig(reference_mode='person'))
    assert instance.pipeline is None and not instance.ip_adapter_loaded


def test_exif_orientation_before_crop(tmp_path):
    project = create_project(tmp_path/'exif', 'exif')
    source = tmp_path/'rotated.jpg'
    image = Image.new('RGB', (40,20), 'red')
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    ref = add_person_reference(project, add_person(project), source, 'person')
    prepared, _meta = prepare_reference_image(project, ref)
    with Image.open(prepared) as oriented:
        assert oriented.size == (20,40)
    cropped, _meta = prepare_reference_image(project, ref, (0,0,20,30))
    with Image.open(cropped) as crop:
        assert crop.size == (20,30)
    with Image.open(source) as original:
        assert original.size == (40,20) and original.getexif()[274] == 6


def test_requested_reference_without_actual_application_is_rejected(tmp_path):
    project, ref = ref_project(tmp_path)
    class IgnoringEngine(FakeEngine):
        def generate_one(self, *args, **kwargs):
            image = super().generate_one(*args, **kwargs)
            self.last_reference_applied = False
            return image
    result = generate_scene_candidates(project, confirmed_scene(), IgnoringEngine(), GenerationConfig(reference_mode='person',reference_image_id=ref.image_id), Event())
    assert result.failed == 1 and not project.candidates
    assert 'not applied' in result.errors[0]


def test_environment_resolves_relative_model_under_app_root(tmp_path):
    from covermorph.generation import detect_generation_environment
    model = tmp_path / 'models' / 'sdxl'
    model.mkdir(parents=True)
    (model / 'model_index.json').write_text('{}')
    import sys
    from types import SimpleNamespace
    sys.modules['torch'] = SimpleNamespace(__version__='test', cuda=SimpleNamespace(is_available=lambda: False))
    result = detect_generation_environment(tmp_path, 'models/sdxl')
    assert result['resolved_model_path'] == str(model)
    assert result['model_ready'] is True
