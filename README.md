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

 - `timer`: Wypisuje czas przejścia dla 1 batcha
 - `count_parameters`: Wypisuje ilość parametrów, MAC-ów oraz schemat modelu
 - `roc_and_statistics`: Zwraca wykres ROC oraz raport klasyfikacji (czyli recall, precision, f1-score dla wszystkich klas oraz accuracy) 

## 6. Pruning
Pruning mamy w dwóch wariantach: unstructured (usuwanie pojedyńczych wag) oraz structured (usuwanie grup wag). Dodatkowo każdy z nich może zostać przeprowadzony z wykorzystaniem regularyzacji L1 lub L2. W przypadku tego projektu używamy pruningu wbudowanego w bibliotekę PyTorch (lub fRAmEWOrk jeżeli chce któs być biznesowym ważniakiem) - co oznacza, że możemy porównać różnice w wykorzystaniu regularyzacji tylko dla structured pruningu, gdyż dla unstructured efektywnie nie ma implementacji L2. Oficjalny powód jest taki, że nie ma to wpływu na wynik końcowy.

 - `apply_structured_pruning`: robi structured pruning L1 lub L2 - zależy co wybierzesz
 - `apply_unstructured_pruning`: robi unstructured __WAŻNE__: parametr 'part_step'(domyślnie False) słuzy do decydowania czy chcemy zrobić kilka takich pruningów pod rząd - wtedy ten parametr musi być True dla wszytskich poza ostatnim. Na sam koniec pruningu wypisuje informacje o ilości zerowych parametrów, wszystkich parametró oraz sparcity

## 7. Jak popełniać pruning?
 - __Unstructured__ (L1/L2):
    Trening modelu -> Analiza wyników na danych testowych -> `apply_unstructured_pruning` -> Dotrenowanie modelu -> Analiza wyników na danych testowych
    __Ważne! Tutaj jeżeli unstructured pruning ma part_step = True, to nie uda się użyć `count_parameters` - wynika to z tego, że na sieci jest maska, która jest niekompatybilna z tą funkcją - można jej użyć dopiero po zroobieniu pruningu z part_step = False__
    
 - __Structured L1/L2__:
    Trening modelu -> Analiza wyników na danych testowych -> `apply_structured_pruning` -> Dotrenowanie modelu -> Analiza wyników na danych testowych
