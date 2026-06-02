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

## 4. Opis modelu testowego
Funckja aktywacji w każdej warstwie to ReLU. Liczba klas jest równa 10

1. Warstwa konwolucyjna *32x32*
2. Warstwa konwolucyjna i pooling *16x16*
3. Warstwa konwolucyjna i pooling *8x8*
4. Warstwa konwolucyjna i pooling *4x4*
5. Warstwa feed-forward *256x4x4 -> 512*
6. Warstwa feed-forward *512 -> 10* 

## 5. Statystyki używane do porównywania działania sieci
Rozpiska funkcji z pliku statistics.py:

 - `accuracy`: Zwraca wynik Accuracy score dla danego modelu oraz czas przejścia 1 batcha (Jest szansa, że ta funkcja stanie się niepotrzebna)
 - `count_parameters`: Zwraca ilość parametrów, ilość niezerowych parametrów oraz sparcity sieci
 - `roc_and_statistics`: Zwraca wykres ROC oraz raport klasyfikacji (czyli recall, precision, f1-score dla wszystkich klas oraz accuracy) 

## 6. Pruning
Pruning mamy w dwóch wariantach: unstructured (usuwanie pojedyńczych wag) oraz structured (usuwanie grup wag). Dodatkowo każdy z nich może zostać przeprowadzony z wykorzystaniem regularyzacji L1 lub L2. W przypadku tego projektu używamy pruningu wbudowanego w bibliotekę PyTorch (lub fRAmEWOrk jeżeli chce któs być biznesowym ważniakiem) - co oznacza, że możemy porównać różnice w wykorzystaniu regularyzacji tylko dla structured pruningu, gdyż dla unstructured efektywnie nie ma implementacji L2. Oficjalny powód jest taki, że nie ma to wpływu na wynik końcowy.

 - `apply_unstructured_pruning`: unstructured pruning ~~no co ty nie powiesz~~, przyjmuje model oraz ammount od 0.0 do 1.0, jeżeli nie zostanie sprecyzowany to ustawia się na 0.5
 - `apply_structured_pruning`: structured pruning, przyjmuje to co powyżej, oraz regularization mogącą być równa 1 lub 2 (1 odpowiada L1, a 2 - L2)
