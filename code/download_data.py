"""Downloads the two public corpora from their official sources into ../data.

EmpatheticDialogues: Rashkin et al., ACL 2019 (Facebook Research, CC BY-NC 4.0)
ESConv:              Liu et al., ACL-IJCNLP 2021 (thu-coai)
The SHA-1 checksums are those of the files used in the paper.
"""
import hashlib, os, tarfile, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FILES = {
    "ed.tar.gz": ("https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz",
                  "e9d793c59eb92e2b4fa9c1063c1911002b806510"),
    "ESConv.json": ("https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json",
                    "4953374db4ef72c2a61645aed3c6085c78885960"),
}

os.makedirs(DATA, exist_ok=True)
for name, (url, sha1) in FILES.items():
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print("downloading", url)
        urllib.request.urlretrieve(url, path)
    h = hashlib.sha1(open(path, "rb").read()).hexdigest()
    print(name, "checksum OK" if h == sha1 else f"WARNING: checksum differs from the paper's copy ({h})")

ed_dir = os.path.join(DATA, "ed")
if not os.path.exists(os.path.join(ed_dir, "empatheticdialogues", "train.csv")):
    with tarfile.open(os.path.join(DATA, "ed.tar.gz")) as t:
        t.extractall(ed_dir, filter="data")
print("data ready in", DATA)
