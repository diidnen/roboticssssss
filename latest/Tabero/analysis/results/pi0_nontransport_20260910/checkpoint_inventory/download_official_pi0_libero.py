"""Download the public official pi0_libero object tree with a hard free-space reserve."""
import base64
import calendar
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path(__file__).resolve().parent
INVENTORY = json.loads((OUT / "public_objects.json").read_text())["items"]
DEST_ROOT = pathlib.Path("/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero")
RESERVE = 5 * 1024**3
DEADLINE = calendar.timegm((2026, 9, 10, 15, 0, 0))
TOTAL = sum(int(x["size"]) for x in INVENTORY)

def available():
    return shutil.disk_usage(DEST_ROOT.parent).free

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    free = available()
    receipt = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": "gs://openpi-assets/checkpoints/pi0_libero", "destination": str(DEST_ROOT), "required_bytes": TOTAL, "free_before_bytes": free, "reserve_bytes": RESERVE, "objects": []}
    remaining = sum(int(x["size"]) for x in INVENTORY if not ((DEST_ROOT / x["name"].removeprefix("checkpoints/pi0_libero/")).exists() and (DEST_ROOT / x["name"].removeprefix("checkpoints/pi0_libero/")).stat().st_size == int(x["size"])))
    receipt["remaining_bytes_at_start"] = remaining
    if free - remaining < RESERVE:
        receipt.update({"state": "preflight_refused", "reason": "Remaining official objects would violate 5 GiB reserve."})
        (OUT / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return 2
    for item in INVENTORY:
        if time.time() >= DEADLINE:
            receipt.update({"state": "stopped_deadline", "stopped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            (OUT / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            return 4
        relative = item["name"].removeprefix("checkpoints/pi0_libero/")
        target = DEST_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        size = int(item["size"])
        if target.exists() and target.stat().st_size == size:
            digest = sha256(target)
            receipt["objects"].append({"relative_path": relative, "bytes": size, "sha256": digest, "remote_md5_base64": item.get("md5Hash"), "state": "already_complete"})
            continue
        if available() - size < RESERVE:
            receipt.update({"state": "stopped_low_space", "free_bytes": available(), "next_object": relative})
            (OUT / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            return 3
        partial = target.with_name(target.name + ".partial")
        if partial.exists():
            raise RuntimeError(f"Existing partial preserved; explicit operator resolution required: {partial}")
        url = "https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/" + urllib.parse.quote(item["name"], safe="") + "?alt=media"
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as output:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
        if partial.stat().st_size != size:
            raise RuntimeError(f"Size mismatch for {relative}: {partial.stat().st_size} != {size}")
        remote_md5 = base64.b64decode(item["md5Hash"])
        local_md5 = hashlib.md5(partial.read_bytes()).digest()
        if local_md5 != remote_md5:
            raise RuntimeError(f"MD5 mismatch for {relative}")
        digest = sha256(partial)
        os.replace(partial, target)
        receipt["objects"].append({"relative_path": relative, "bytes": size, "sha256": digest, "remote_md5_base64": item.get("md5Hash"), "state": "downloaded"})
        receipt["bytes_completed"] = sum(x["bytes"] for x in receipt["objects"])
        receipt["free_bytes"] = available()
        (OUT / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    receipt.update({"state": "complete", "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bytes_completed": TOTAL, "free_after_bytes": available()})
    (OUT / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
