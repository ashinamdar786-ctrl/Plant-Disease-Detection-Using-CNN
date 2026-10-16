import os
import urllib.request
import sys

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path} ...", flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(target_path, 'wb') as out_file:
        total = resp.headers.get('Content-Length')
        total = int(total) if total else None
        downloaded = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = (downloaded / total) * 100
                sys.stdout.write(f"\r  {downloaded}/{total} bytes ({percent:.1f}%)")
                sys.stdout.flush()
    print("\n  Complete!", flush=True)

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    base_url = "https://huggingface.co/neghani/plantvillage-mobilenetv2"
    
    files = [
        ("resolve/main/model.safetensors", "models/model.safetensors"),
        ("raw/main/config.json", "models/config.json"),
        ("raw/main/preprocessor_config.json", "models/preprocessor_config.json"),
        ("raw/main/test_results.json", "models/test_results.json"),
        ("raw/main/reports/metrics.json", "models/metrics.json"),
        ("raw/main/reports/classification_report.txt", "models/classification_report.txt"),
        ("raw/main/threshold.json", "models/threshold.json")
    ]
    
    for subpath, dest in files:
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            download_file(f"{base_url}/{subpath}", dest)
        else:
            print(f"Already exists: {dest} ({os.path.getsize(dest)} bytes)")
    print("All model assets ready!")
