
import os, json, pickle, datetime

DRIVE_ROOT = "/content/drive/MyDrive/FPX_Project"

def get_path(*parts):
    path = os.path.join(DRIVE_ROOT, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def log(message):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line)
    log_path = get_path("logs", "project_log.txt")
    with open(log_path, "a") as f:
        f.write(line + "\n")

def save_json(data, subdir, name):
    path = get_path(subdir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log(f"Saved JSON checkpoint: {subdir}/{name}.json")
    return path

def load_json(subdir, name, default=None):
    path = get_path(subdir, f"{name}.json")
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)

def save_pickle(obj, subdir, name):
    path = get_path(subdir, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log(f"Saved pickle checkpoint: {subdir}/{name}.pkl")
    return path

def load_pickle(subdir, name, default=None):
    path = get_path(subdir, f"{name}.pkl")
    if not os.path.exists(path):
        return default
    with open(path, "rb") as f:
        return pickle.load(f)

def list_checkpoints(subdir):
    d = os.path.join(DRIVE_ROOT, subdir)
    if not os.path.exists(d):
        return []
    return sorted(os.listdir(d))

def project_status():
    print("=== FPX Project Drive Status ===")
    for subdir in ["config", "data", "checkpoints/fl_sim", "checkpoints/signals",
                   "checkpoints/interface", "checkpoints/eval", "outputs", "logs"]:
        files = list_checkpoints(subdir)
        print(f"{subdir}: {len(files)} file(s)")
        for f in files[:10]:
            print(f"   - {f}")
