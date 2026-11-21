"""
RunPod worker for motion control (Wan2.2-Animate-14B, "animation" mode):
a character image copies the movement of a driving video.

input: {"face_image_url": <character image>, "video_url": <driving video>}
Model weights (~tens of GB) are downloaded once into the network volume at /runpod-volume.
Optional env vars: MOTION_RES (default "1280 720"), MOTION_EXTRA_ARGS, R2_* (same as the face-swap worker).
"""
import os, sys, time, uuid, shutil, logging, subprocess, urllib.request
from pathlib import Path
import runpod

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("motion")

REPO = Path("/opt/Wan2.2")
CKPT = Path(os.environ.get("MOTION_CKPT", "/runpod-volume/Wan2.2-Animate-14B"))
RES = os.environ.get("MOTION_RES", "1280 720").split()
EXTRA = os.environ.get("MOTION_EXTRA_ARGS", "--offload_model True --convert_model_dtype").split()
WORK = Path("/tmp/motion_jobs"); WORK.mkdir(parents=True, exist_ok=True)


def ensure_weights():
    if not (CKPT / "process_checkpoint").exists():
        from huggingface_hub import snapshot_download
        log.info("Downloading Wan2.2-Animate-14B weights (first run only, this takes a while)...")
        snapshot_download("Wan-AI/Wan2.2-Animate-14B", local_dir=str(CKPT))


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; MotionWorker/1.0)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as out:
        shutil.copyfileobj(r, out)
    if dest.stat().st_size == 0:
        raise ValueError("Downloaded file is empty.")


def run(cmd):
    log.info("RUN: " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    if p.returncode != 0:
        raise RuntimeError("Step failed: " + (p.stderr or p.stdout)[-700:])


def watermark(src, dst):
    vf = ("drawtext=text='AI Generated':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
          "fontcolor=white@0.85:fontsize=h/28:box=1:boxcolor=black@0.4:boxborderw=8:x=w-tw-24:y=h-th-24")
    p = subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-c:a", "copy", str(dst)], capture_output=True)
    return dst if p.returncode == 0 and dst.exists() else src


def upload(path):
    import boto3
    acct, bucket = os.environ["R2_ACCOUNT_ID"], os.environ["R2_BUCKET_NAME"]
    s3 = boto3.client("s3", endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
                      aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")
    key = f"outputs/{uuid.uuid4().hex}.mp4"
    s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600)


def handler(job):
    inp = job.get("input", {})
    img_url, vid_url = inp.get("face_image_url"), inp.get("video_url")
    if not img_url or not vid_url:
        return {"status": "failed", "error": "Missing character image or driving video."}
    jd = WORK / job.get("id", uuid.uuid4().hex)
    proc = jd / "proc"; proc.mkdir(parents=True, exist_ok=True)
    img, vid = jd / "character.jpg", jd / "drive.mp4"
    try:
        ensure_weights()
        download(img_url, img); download(vid_url, vid)
        started = time.time()
        run([sys.executable, "wan/modules/animate/preprocess/preprocess_data.py",
             "--ckpt_path", str(CKPT / "process_checkpoint"), "--video_path", str(vid), "--refer_path", str(img),
             "--save_path", str(proc), "--resolution_area", *RES, "--retarget_flag"])
        run([sys.executable, "generate.py", "--task", "animate-14B", "--ckpt_dir", str(CKPT),
             "--src_root_path", str(proc), "--refert_num", "1", *EXTRA])
        outs = [p for p in REPO.glob("*.mp4") if p.stat().st_mtime >= started]
        if not outs:
            raise RuntimeError("The model finished but no video file was found.")
        raw = max(outs, key=lambda p: p.stat().st_mtime)
        final = watermark(raw, jd / "final.mp4")
        url = upload(final)
        raw.unlink(missing_ok=True)
        return {"status": "done", "kind": "video", "output_url": url, "seconds": round(time.time() - started)}
    except Exception as e:
        log.exception("failed")
        return {"status": "failed", "error": str(e)}
    finally:
        shutil.rmtree(jd, ignore_errors=True)


runpod.serverless.start({"handler": handler})
