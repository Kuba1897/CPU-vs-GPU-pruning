# CPU-vs-GPU-pruning
Piotr Nowak, Jakub Ledwoń

## 1. Wersja Pythona
Projekt wymaga Pythona **3.10 lub 3.11**.

## 2. CUDA
Projekt wykorzystuje PyTorch i może działać zarówno na CPU, jak i GPU (CUDA).
Do działania na GPU wymagane są:
- karta NVIDIA z obsługą CUDA
- zainstalowany PyTorch z obsługą CUDA (np. `cu121`)
- aktualne sterowniki NVIDIA

## 3. Instalacja dependencies
pip install -r requirements.txt

UWAGA! Jeśli chcesz używać GPU, zainstaluj PyTorch osobno:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 , lub inne w zależności od wersji karty