"""Reproducible visual feature extraction for ImageFolder-formatted datasets.

Requires optional vision dependencies and a user-provided image directory.
No dataset or checkpoint is bundled with this repository. Labels are stored
for evaluator-side metrics and never passed to unsupervised learner objects.

The extraction interface writes memory-mapped NPY arrays to avoid holding all
ImageNet-1K embeddings in RAM. Source checkpoint provenance is recorded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def imagefolder_manifest(root: str | Path, allowed_classes: list[str] | None = None):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if allowed_classes is not None:
        requested = set(allowed_classes)
        missing = requested - set(classes)
        if missing:
            raise ValueError(f"Unknown synsets/classes in root: {sorted(missing)}")
        classes = [name for name in classes if name in requested]
    samples=[]
    suffixes={".jpeg",".jpg",".png",".webp"}
    for label,synset in enumerate(classes):
        for p in sorted((root/synset).rglob("*")):
            if p.is_file() and p.suffix.lower() in suffixes:
                samples.append((str(p),label,synset))
    if not samples:
        raise ValueError("No images discovered")
    return classes,samples


def _manifest_digest(samples):
    h=hashlib.sha256()
    for path,label,synset in samples:
        h.update(f"{synset}\0{label}\0{Path(path).name}\n".encode())
    return h.hexdigest()


def extract_imagefolder(root, output_dir, *, model, processor,
                        model_name: str, batch_size: int = 32,
                        device: str | None = None, allowed_classes=None,
                        normalize: bool = True, pretrained_source: str = "unspecified"):
    """Extract CLS/pooled tensors from a callable image encoder.

    `processor(images=[PIL,...], return_tensors='pt')` supplies pixel_values.
    `model(pixel_values)` may return last_hidden_state, pooler_output,
    a dict containing features, or a 2D tensor. The model is eval/no-grad.
    """
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install torch and Pillow extras for visual extraction") from exc
    if batch_size<1:raise ValueError("batch_size must be >=1")
    classes,samples=imagefolder_manifest(root,allowed_classes)
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device).eval()
    data_path=output_dir/"features.npy"
    labels_path=output_dir/"labels.npy"
    ids_path=output_dir/"ids.json"
    feature_map=None
    labels=np.asarray([label for _,label,_ in samples],dtype=np.int32)
    np.save(labels_path,labels)
    names=[]
    for start in range(0,len(samples),batch_size):
        batch=samples[start:start+batch_size]
        images=[]
        for path,_,_ in batch:
            with Image.open(path) as img:
                images.append(img.convert("RGB").copy())
        processed=processor(images=images,return_tensors="pt")
        if "pixel_values" not in processed:
            raise ValueError("Processor must provide pixel_values")
        pixels=processed["pixel_values"].to(device)
        with torch.inference_mode():
            result=model(pixels)
            if hasattr(result,"last_hidden_state"):
                features=result.last_hidden_state[:,0]
            elif hasattr(result,"pooler_output") and result.pooler_output is not None:
                features=result.pooler_output
            elif isinstance(result,dict) and "features" in result:
                features=result["features"]
            elif isinstance(result,torch.Tensor):
                features=result
            else:
                raise TypeError("Unsupported model output; supply a feature adapter")
        features=features.float()
        if features.ndim!=2 or features.shape[0]!=len(batch):
            raise ValueError("Expected feature tensor [batch,dim]")
        if normalize:
            features=torch.nn.functional.normalize(features,dim=-1)
        arr=features.cpu().numpy()
        if not np.isfinite(arr).all():
            raise ValueError("Encoder produced non-finite embeddings")
        if feature_map is None:
            feature_map=np.lib.format.open_memmap(data_path,mode="w+",dtype="float32",
                                                   shape=(len(samples),arr.shape[1]))
        feature_map[start:start+len(batch)]=arr
        names.extend([f"{synset}/{Path(path).name}" for path,_,synset in batch])
    feature_map.flush()
    ids_path.write_text(json.dumps(names))
    metadata={"model":model_name,"pretrained_source":pretrained_source,
              "normalized":normalize,"feature_dim":feature_map.shape[1],
              "n_images":len(samples),"classes":classes,
              "class_count":len(classes),"manifest_sha256":_manifest_digest(samples),
              "files":{"features":data_path.name,"labels":labels_path.name,"ids":ids_path.name},
              "device":device}
    (output_dir/"metadata.json").write_text(json.dumps(metadata,indent=2))
    return metadata


def load_memmap_features(root):
    root=Path(root)
    meta=json.loads((root/"metadata.json").read_text())
    x=np.load(root/meta["files"]["features"],mmap_mode="r")
    labels=np.load(root/meta["files"]["labels"],mmap_mode="r")
    ids=json.loads((root/meta["files"]["ids"]).read_text())
    if len(x)!=len(labels) or len(ids)!=len(x):
        raise ValueError("Inconsistent extracted feature files")
    return x,labels,ids,meta


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--model",default="facebook/dinov2-base")
    parser.add_argument("--revision",default=None,
                        help="Explicit checkpoint revision for reproducibility")
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--synsets-file",default=None)
    parser.add_argument("--device",default=None)
    args=parser.parse_args(argv)
    try:
        from transformers import AutoImageProcessor,AutoModel
    except ImportError as exc:
        raise SystemExit("Install lmoment-lifelong-learning[vision] before extraction") from exc
    kwargs={} if args.revision is None else {"revision":args.revision}
    processor=AutoImageProcessor.from_pretrained(args.model,**kwargs)
    model=AutoModel.from_pretrained(args.model,**kwargs)
    synsets=(Path(args.synsets_file).read_text().splitlines()
             if args.synsets_file else None)
    out=extract_imagefolder(args.image_root,args.output_dir,model=model,
                            processor=processor,model_name=args.model,
                            batch_size=args.batch_size,device=args.device,
                            allowed_classes=synsets,
                            pretrained_source=args.revision or "default_remote_revision")
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()
