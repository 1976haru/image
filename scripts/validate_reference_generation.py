"""Local, explicit SDXL comparison. Never downloads or falls back to CPU."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Event

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from covermorph.generation import (
    GenerationConfig,
    SDXLTextToImageEngine,
    detect_generation_environment,
    generate_scene_candidates,
)
from covermorph.project import SceneCard, add_person, add_person_reference, create_project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/sdxl_base_1.0')
    parser.add_argument('--adapter', default='models/ip_adapter')
    parser.add_argument('--person', type=Path)
    parser.add_argument('--style', type=Path)
    parser.add_argument('--prompt', default='A person sitting in a warm sunlit cafe, detailed photograph, no text')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--ratio', choices=['1:1','16:9','9:16'], default='1:1')
    parser.add_argument('--steps', type=int, default=28)
    parser.add_argument('--diagnose-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / 'validation_results' / ('3b1_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True)
    report = {'environment': detect_generation_environment(root, args.model), 'platform': platform.platform(),
              'python': sys.version, 'settings': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'inference_status': 'not_run', 'visual_quality_status': 'unverified', 'comparisons': []}
    report_path = destination / 'report.json'
    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    if args.diagnose_only or report['environment']['status'] != 'ready':
        print(f'Actual inference not run. Diagnosis: {report_path}')
        return
    import torch
    if not args.person and not args.style:
        parser.error('Provide --person and/or --style for actual comparison.')
    for mode, source in [('person', args.person), ('style', args.style)]:
        if source is None:
            continue
        project = create_project(destination / mode, mode)
        reference = add_person_reference(project, add_person(project, 'reference'), source, mode)
        scene = SceneCard('comparison', prompt_user=args.prompt, negative_prompt_user='text, watermark', prompt_confirmed=True)
        project.scenes.append(scene)
        engine = SDXLTextToImageEngine(args.model, local_files_only=True)
        try:
            for strength in [None, .5, .8]:
                config = GenerationConfig(model_id=args.model, output_ratio=args.ratio, steps=args.steps, seed=args.seed,
                    reference_mode=mode if strength is not None else 'off', reference_image_id=reference.image_id,
                    reference_strength=strength if strength is not None else .5, ip_adapter_id=args.adapter)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                result = generate_scene_candidates(project, scene, engine, config, Event())
                torch.cuda.synchronize()
                report['comparisons'].append({'mode': config.reference_mode, 'strength': strength, 'run_id': result.run_id,
                    'seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'completed': result.completed, 'errors': result.errors,
                    'reference_effect_review': 'unverified', 'face_distortion_review': 'unverified', 'composition_review': 'unverified'})
                report['inference_status'] = 'executed'
                save()
        except Exception as exc:
            report['comparisons'].append({'mode': mode, 'error': str(exc)})
            save()
        finally:
            engine.unload()
    print(f'Compare the saved PNG candidates manually. Report: {report_path}')


if __name__ == '__main__':
    main()
