# OcenaOceny
Ocena oceny jest narzędziem służącym do analizowania sentymentu recenzji tekstowych. 
Program automatycznie ocenia, czy dana opinia jest pozytywna czy negatywna, co ułatwia przetwarzanie dużych zbiorów danych opinii.

## Programistki
- Magdalena Pakuła
- Gabriela Szkiłondź

## Funkcjonalności
- Analiza sentymentu: Program używa modelu SVM do klasyfikacji tekstu jako pozytywnego lub negatywnego na podstawie treści recenzji.
- Interfejs graficzny: GUI umożliwia użytkownikowi wprowadzenie recenzji tekstowej i natychmiastowe uzyskanie oceny sentymentu.

## Konfiguracja
### Wymagania
- Python 3.x
- Biblioteki Pythona: tkinter, joblib, scikit-learn

### Instalacja zależności
Aby zainstalować niezbędne biblioteki, użyj poniższego polecenia:

`
pip install -r requirements.txt
`

### Szkolenie modelu
Aby uruchomić program, musisz najpierw przeszkolić model SVM na swoich danych. 
Uruchom main.py, który automatycznie przetworzy dane, przeszkoli model i zapisze go do plików model.joblib, vectorizer.joblib.

`
python main.py
`

### Uruchomienie GUI
Po przeszkoleniu modelu możesz uruchomić GUI, które umożliwia analizę sentymentu wprowadzonych recenzji.

`
python GUI/main_interface.py
`

### Struktura plików
- main.py: Plik główny do przetwarzania danych, szkolenia modelu SVM i zapisu wyników.
- GUI/main_interface.py: Interfejs graficzny do interaktywnej analizy sentymentu recenzji.
- data/: Katalog zawierający dane treningowe, np. Translated_IMDB_Dataset_MERGED.csv.
- model.joblib: Zapisany model SVM.
- explainer.joblib: Zapisany obiekt do wyjaśniania predykcji.

### Uwagi
- Upewnij się, że ścieżki do zapisanych modeli (model.joblib, vectorizer.joblib, explainerNEW.joblib) są poprawnie ustawione w main.py oraz GUI/main_interface.py, aby GUI mogło poprawnie załadować wcześniej przeszkolony model.