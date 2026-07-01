# DualBrep — Reconstruction & Generation Inference

Turn a CAD solid (or a point cloud / single image) into a watertight, parametric **B-rep**
(STEP file). The neural stages run in PyTorch; the final B-rep assembly uses OpenCASCADE.

## Release progress

- [x] **Checkpoint release**
- [x] **Sample code release**
- [ ] Full testing results
- [ ] Training scripts & data pipeline
- [ ] VAE pipeline

# Install

A single conda environment:

```bash
conda create -n Dualbrep python=3.12 -y
conda activate Dualbrep
conda install -c conda-forge pythonocc-core=7.9.0 -y
pip install -r requirements.txt
```

Then download the checkpoints from [🤗 ADSKAILab/DualBrep](https://huggingface.co/ADSKAILab/DualBrep)
and unzip them into this folder. The **quick start** needs only `checkpoints/` (the demo point
clouds in `sample/` ship with the repo); the **full test set** (`data/`) is only needed to
reproduce the paper results.

```bash
pip install -U "huggingface_hub[cli]"
hf download ADSKAILab/DualBrep checkpoints.zip --repo-type model --local-dir .
unzip checkpoints.zip      # -> checkpoints/
```

```
DualBrep/
├── sample/        00013293.ply  00021127.ply  00026543.ply  00030157.ply   # bundled demo clouds
├── checkpoints/   ae_vae.ckpt  pc_vae.ckpt  pc_flow.ckpt  img_flow.ckpt  parametrizer.ckpt
└── data/          implicit/  pc/  imgs/  final_test.txt        # full 4150-shape test set
```

# Introduction of the pipeline

```
Input mesh / point cloud / image
   └─ AE  or  diffusion + VAE decoder      (ae_reconstruct.py / vae_generate.py)
        └─ SDF / UDF fields  →  surface mesh + Voronoi field
             └─ segmentation                (clustering.py)        →  segmented mesh (cluster.ply)
                  └─ Parametrizer            (rebuild.py)           →  per-face UV grids + edges + connectivity (tmp/<name>_<k>/post.npz)
                       └─ Rebuilder          (postprocess.py)       →  watertight B-rep (<name>.step + <name>.ply)
```

| Stage | Script | Output |
|-------|--------|--------|
| AE reconstruction (SDF/UDF) | `ae_reconstruct.py` | `recon_sdf.ply`, `recon_udf.ply`, `udf_g.npy` |
| Diffusion generation (PC / image → SDF/UDF) | `vae_generate.py` | same, conditioned |
| Segmentation (B-rep faces) | `clustering.py` (via the above) | `cluster.ply` |
| Parametrizer (UV grids + topology) | `rebuild.py` (+`rebuild_model.py`) | `tmp/<name>_<k>/post.npz` |
| Rebuilder (assemble solid) | `postprocess.py` (+`brep_post/`) | `<name>.step` + `<name>.ply` |

# Quick start on sample data

`run_pipeline.sh` runs the whole chain (point cloud → AE reconstruct → segment → parametrize →
rebuild) on the **bundled `sample/` point clouds** — 4 shapes that reconstruct into valid
solids. It needs only `checkpoints/` (no `data/` download required):

```bash
./run_pipeline.sh
# overridable: CONFIG=config.yaml (implicit AE)  OUT=out  ROTATIONS=all  TEST_RES=256
```

Results land in `output_pipeline/` — `recon/<name>/cluster.ply` and one valid B-rep per shape
`brep/<name>.step` + `brep/<name>.ply` (the 24 rotation candidates are kept under `brep/tmp/`).

# Reproduce main results

Run the stages explicitly on the full test set (4150 shapes, listed in `data/final_test.txt`).
The configs default to the 8-shape demo list (`sample_shapes*.txt`); add
`dataset.name_list=data/final_test.txt` to run the whole test set.

```bash
# 1) reconstruct (AE) OR generate (diffusion), with segmentation, at 256^3
python ae_reconstruct.py config=config.yaml dataset.name_list=data/final_test.txt runtime.test_res=256   # implicit AE
python ae_reconstruct.py config=config_pc.yaml runtime.test_res=256       # point-cloud AE
python vae_generate.py   config=config_gen_pc.yaml                         # point cloud → shape
python vae_generate.py   config=config_gen_img.yaml                        # image → shape
#   -> per shape: recon_sdf.ply / recon_udf.ply / udf_g.npy / cluster.ply

# 2) parametrize each segmented mesh (24-rotation test-time augmentation)
python rebuild.py --input <recon_output_dir> --out output_brep --rotations all

# 3) assemble the B-rep
python postprocess.py --input output_brep
```

# Detailed use case

## VAE: encoding a B-rep into a continuous vecset latent and decode it back

`ae_reconstruct.py` runs the dual-field autoencoder: it encodes sampled
surface/edge/voronoi points into a vecset latent and decodes an **SDF** (watertight surface)
+ **UDF**-to-edges field, then segments the surface. Two input modes (`dataset.input`):

| Mode | Input | Model | Config | Checkpoint |
|------|-------|-------|--------|------------|
| `implicit` | precomputed `.npz` (surface+edge+voronoi+queries) | `DualVAE` | `config.yaml` | `ae_vae.ckpt` |
| `pointcloud` | oriented `.ply` / `.obj` | `DualVAE_PC` | `config_pc.yaml` | `pc_vae.ckpt` |

```bash
python ae_reconstruct.py                              # implicit defaults (sample_shapes.txt)
python ae_reconstruct.py config=config_pc.yaml        # point-cloud defaults (sample_shapes_pc.txt)
python ae_reconstruct.py checkpoint=checkpoints/ae_vae.ckpt dataset.data_root=/dir \
    dataset.name_list=null runtime.output_dir=output runtime.test_res=256
```

Outputs per shape (`runtime.output_dir/<prefix>_<id_aug>/`): `recon_sdf.ply`, `recon_udf.ply`,
`sdf_g.npy`/`udf_g.npy`, `cluster.ply`. Key knobs: `runtime.test_res` (128/256/512),
`runtime.acc=true` (accelerated near-surface mesher — coarse 128³ then re-evaluate the
`|sdf|` band, identical iso-0 surface, ≈19× faster at 512³), `runtime.precision`.

**Using your own STEP/B-rep.** The `implicit` `.npz` (surface/edge/**voronoi** samples +
SDF/UDF queries) is precomputed from a B-rep. To build it for your own steps, compute the
voronoi/implicit fields with the C++ tool and the prep script: `C/dualfield2mesh`
(build via `C/install.sh`) and `python/data_mesh2sdfs/prepare_implicit_ray.py`.

## Shape generation: point-cloud- or image-conditioned

`vae_generate.py` runs the rectified-flow generator (conditioning → DiT flow noise→latent →
frozen `DualVAE` decode → SDF/UDF → mesh → segment):

| Mode | Input | Encoder | Config | Checkpoint |
|------|-------|---------|--------|------------|
| `pointcloud` | `.ply` (8192 pts) | `PCModel2` | `config_gen_pc.yaml` | `pc_flow.ckpt` |
| `image` | one RGB render | DINOv2 (`ImgModel`) | `config_gen_img.yaml` | `img_flow.ckpt` |

```bash
python vae_generate.py config=config_gen_pc.yaml         # point cloud → shape
python vae_generate.py config=config_gen_img.yaml        # image → shape (downloads DINOv2 ~1.1 GB)
```

Knobs: `runtime.steps` (Euler steps), `runtime.temperature` (`>0` enables classifier-free
guidance), `runtime.test_res`.

## Mesh segmentation: segment an input mesh into B-rep faces

`clustering.py` (invoked automatically when `runtime.compute_clustering=true`) segments the
reconstructed surface using the UDF edge field. The segmentation is aligned with **B-rep
faces** (one cluster per smooth face bounded by sharp edges), *not* semantic parts. It writes
`cluster.ply` (per-face `label`), and can also be re-run on existing folders:

```bash
python clustering.py output/      # each folder needs recon_sdf.ply + udf_g.npy
```

## Parametrization: segmented triangle soup → UV grids → trimmed, assembled B-rep

Two steps turn the segmented mesh into a valid solid.

**Parametrizer — `rebuild.py`** (`Parametrizer`, `parametrizer.ckpt`). Samples each labelled
face (100 pts + normals), re-fits it as a 16×16 parametric surface grid, and predicts the
face-face **intersection edges + topology**, written to `post.npz`. `--rotations all`
re-poses the mesh by each of the 24 octahedral rotations `M[k]` (index 3 = identity) into
`tmp/<name>_<k>/` — test-time augmentation: many more candidates seal than a single pose.

```bash
python rebuild.py --input <recon_dir>          --out output_brep --rotations all
python rebuild.py --input <name>/cluster.ply   --out output_brep --rotations all
```

**Rebuilder — `postprocess.py`** (OpenCASCADE). Fits a trimmed B-spline surface per face,
sews edges into wires/faces, and sews the faces into a **watertight solid**; it undoes each
candidate's rotation with `inv(M[k])`. Candidates are assembled in parallel with **Ray** (one
CPU each); for every shape the first rotation that seals is promoted to `<name>.step` +
`<name>.ply` (triangulation) in the output folder — the 24 candidates stay under `tmp/`.

```bash
python postprocess.py --input output_brep          # Ray parallel by default
# knobs: --num_cpus 16  --serial  --max_optimize_iter 200  --drop_num 0  --no_optimize
```

Across rotations, roughly half seal into closed 1-solid STEPs (`BRepCheck` valid); one is kept per shape.
