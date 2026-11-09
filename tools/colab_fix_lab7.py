"""One-off patch: make Lab7 (GMM/VAE/AR) notebook runnable on Colab."""
import json, sys

p = sys.argv[1]
nb = json.load(open(p, encoding='utf-8'))
c = nb['cells'][2]
s = ''.join(c['source'])
s = s.replace('os.environ["HF_HUB_OFFLINE"] = "1"\n', '')
s = s.replace('FIG = os.path.join(HERE, "..", "slides", "assets", "figures")', 'FIG = os.path.join(HERE, "figures")')
start = s.index('if not os.path.exists(CIFAR):')
end = s.find('\n', s.index('raise SystemExit', start))
end = len(s) if end == -1 else end
new = '''if not os.path.exists(CIFAR):  # Colab: download CIFAR-10 with torchvision and cache it as a uint8 npz
    import torchvision
    os.makedirs(DATA, exist_ok=True)
    _tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True)
    _te = torchvision.datasets.CIFAR10(DATA, train=False, download=True)
    np.savez(CIFAR, train_x=_tr.data, train_y=np.array(_tr.targets), test_x=_te.data,
             test_y=np.array(_te.targets), classes=np.array(_tr.classes))
    print("built", CIFAR)'''
s = s[:start] + new + s[end:]
c['source'] = s.splitlines(keepends=True)
install = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
           "source": ["# Colab: install packages not preinstalled (torchmetrics FID needs torch-fidelity)\n",
                      "!pip install -q torchmetrics torch-fidelity imageio"]}
nb['cells'].insert(2, install)
json.dump(nb, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('ok')
