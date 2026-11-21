# Przewidywanie udziałów mikrostruktury stali S355J2

## Instalacja

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Użyj Pythona 3.12. W PowerShell aktywuj środowisko poleceniem `.\.venv\Scripts\Activate.ps1`.

## Generowanie i symulacja danych

Na Windowsie umieść `Trans_e.exe`, plik konfiguracyjny `coeff.dat` dla S355J2 i wymagane biblioteki DLL w katalogu `program/`.

```sh
python generate.py --program program --out output/labelled.csv
```

## Pipeline treningu i analizy

Trening korzysta z `data/dataset.csv` z zapisanym podziałem na trening, test i foldy walidacji.

```sh
python train.py --out output/main --profile extended_v1 --output-modes joint separate --device recorded
python report.py --baseline output/main --out output/tables
python figures.py --results output/tables
```

`--device recorded` używa urządzeń zapisanych dla wariantów kampanii. Na komputerze bez MPS użyj `--device cpu` w poleceniach treningu.

## Pojedynczy model i badania uzupełniające

```sh
# Ridge dla t(T) i K=32
python train.py --out output/ridge --profile extended_v1 --models ridge --representations tT --k 32

# Rozszerzone strojenie lasu losowego i LightGBM
python train.py --out output/wide --profile wide_v1 --models rf lightgbm --representations Tt --k 128

# Cztery wyjścia softmax w MLP i CNN1D
python train.py --out output/softmax --profile extended_v1 --models mlp cnn1d --representations tT --k 128 --output-modes softmax --device mps

# Trening po wyłączeniu rodzin krzywych
python holdout.py --results output/main --out output/holdout

# Raport z badań uzupełniających
python report.py --baseline output/main --wide output/wide --softmax output/softmax --holdout output/holdout --out output/tables
```

Wznowienie treningu głównego otwiera zapisany katalog przez `python train.py --out output/main --resume`.
