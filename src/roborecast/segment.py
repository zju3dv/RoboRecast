"""SAM3 masks with one model load for sparse source-image/prompt batches."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time


def run_jobs(jobs, checkpoint):
    started = time.monotonic()
    import torch
    import numpy as np
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    load_started = time.monotonic()
    model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False, device='cuda')
    processor = Sam3Processor(model, confidence_threshold=.25)
    load_seconds = time.monotonic() - load_started
    states = {}
    reports = []
    embedding_calls = 0
    for job in jobs:
        tick = time.monotonic()
        source = Path(job['image']).resolve()
        output = Path(job['output']).resolve()
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        requested = job.get('prompt', job.get('prompts', [''])[0])
        phrases = list(dict.fromkeys(job.get('prompts', [requested])))
        if not phrases or any(not isinstance(p, str) or not p.strip() for p in phrases):
            raise ValueError('Each job requires nonempty source-grounded prompts')
        image = Image.open(source).convert('RGB')
        box=job.get('box_xyxy')
        if box is not None:
            box=np.asarray(box,float)
            if box.shape!=(4,) or not np.isfinite(box).all() or not (0<=box[0]<box[2]<=image.width and 0<=box[1]<box[3]<=image.height):raise ValueError('Invalid source pixel box')
            normalized_box=[float((box[0]+box[2])/(2*image.width)),float((box[1]+box[3])/(2*image.height)),float((box[2]-box[0])/image.width),float((box[3]-box[1])/image.height)]
        key = (str(source), source_hash)
        attempts = []
        accepted = None
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if key not in states:
                # Bound GPU memory while reusing the two external-view embeddings.
                if len(states) >= 2:
                    del states[next(iter(states))]
                states[key] = processor.set_image(image)
                embedding_calls += 1
            state = states[key]
            for phrase in phrases:
                inference_started = time.monotonic()
                processor.reset_all_prompts(state)
                result = processor.set_text_prompt(state=state, prompt=phrase)
                if box is not None:
                    result=processor.add_geometric_prompt(box=normalized_box,label=True,state=state)
                scores = result['scores'].float().cpu().numpy().reshape(-1)
                attempt = dict(prompt=phrase, returncode=1, elapsed_seconds=time.monotonic()-inference_started,
                               scores=scores.tolist(),box_xyxy=None if box is None else box.tolist())
                if len(scores):
                    best = int(scores.argmax())
                    mask = result['masks'][best].cpu().numpy().squeeze().astype(bool)
                    if mask.shape != np.asarray(image).shape[:2] or not mask.any():
                        raise ValueError('Invalid SAM3 mask; no synthetic fallback')
                    output.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask.astype('uint8')*255).save(output)
                    attempt['returncode'] = 0
                    accepted = dict(selected=best, selected_prompt=phrase, prompt=phrase,
                                    scores=scores.tolist(), mask_pixels=int(mask.sum()),
                                    mask_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
                attempts.append(attempt)
                if accepted is not None:
                    break
        report = dict(source=str(source), source_sha256=source_hash, requested_prompt=requested,
                      checkpoint=checkpoint, output=str(output), attempts=attempts,box_xyxy=None if box is None else box.tolist(),guidance_evidence=job.get('guidance_evidence'),
                      elapsed_seconds=time.monotonic()-(started if len(jobs)==1 else tick),
                      timing_boundary='whole process' if len(jobs)==1 else 'job after shared model load',
                      status='PASS' if accepted is not None else 'INSUFFICIENT_EVIDENCE')
        if accepted is not None:
            report.update(accepted)
            output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix('.attempts.json').write_text(json.dumps(report, indent=2)+'\n')
        reports.append(report)
    return dict(status='PASS' if all(r['status']=='PASS' for r in reports) else 'INSUFFICIENT_EVIDENCE',
                model_load_count=1, model_load_seconds=load_seconds, image_embedding_calls=embedding_calls,
                elapsed_seconds=time.monotonic()-started, jobs=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--prompt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--jobs', type=Path, help='JSON list of image, prompt, optional prompts, output')
    parser.add_argument('--batch-report', type=Path)
    parser.add_argument('--checkpoint', default=os.environ.get('SAM3_CHECKPOINT'), required=not os.environ.get('SAM3_CHECKPOINT'))
    args = parser.parse_args()
    if args.jobs:
        if any((args.image, args.prompt, args.output)):
            parser.error('--jobs and single-mask arguments are mutually exclusive')
        jobs = json.loads(args.jobs.read_text())
        if not isinstance(jobs, list) or not jobs:
            parser.error('--jobs must contain a nonempty list')
    else:
        if not all((args.image, args.prompt, args.output)):
            parser.error('Supply --jobs or all of --image, --prompt and --output')
        jobs = [dict(image=str(args.image), prompt=args.prompt, output=str(args.output))]
    report = run_jobs(jobs, args.checkpoint)
    if args.batch_report:
        args.batch_report.parent.mkdir(parents=True, exist_ok=True)
        args.batch_report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report if args.jobs else report['jobs'][0]))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
