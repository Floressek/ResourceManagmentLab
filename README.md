# System Ekspertowy - Dobór Modeli Machine Learning

Projekt laboratoryjny z Reprezentacji Wiedzy w systemach informatycznych (ZWI).

## Cel projektu

Implementacja systemu ekspertowego w języku Prolog, który na podstawie charakterystyki datasetu automatycznie rekomenduje odpowiednie modele uczenia maszynowego wraz z pipeline'em preprocessingu i wyjaśnieniem wyboru.

## Struktura projektu

```
src/project/
├── lab1.pl                    # Główna baza wiedzy w Prologu
└── pytania_do_projektu        # Przykładowe zapytania testujące system
```

## Opis działania

System ekspertowy analizuje właściwości datasetu (rozmiar, balans klas, brakujące wartości, typ danych, outliers) i na tej podstawie:
1. Ocenia przydatność różnych modeli ML (scoring 0-100 punktów)
2. Rekomenduje odpowiednie kroki preprocessingu
3. Generuje kompletny pipeline wraz z wyjaśnieniem
4. Pozwala porównywać modele między sobą

### Komponenty systemu punktacji

- **Ocena bazowa** - zgodność modelu z typem problemu
- **Bonus interpretacji** - czy model jest interpretowalny (ważne w biznesie)
- **Bonus szybkości** - szybkość treningu i predykcji
- **Bonus odporności** - radzenie sobie z outlierami, niezbalansowaniem
- **Bonus priorytetu** - spełnienie priorytetów datasetu (dokładność/szybkość/interpretacja)

## Funkcjonalności

- **Rekomendacja modeli** z oceną punktową i wyjaśnieniem
- **Ranking Top-N** najlepszych modeli dla datasetu
- **Porównanie modeli** - który jest lepszy i dlaczego
- **Generowanie pipeline'u preprocessingu** z walidacją
- **Analiza trudności datasetu** - ocena złożoności problemu
- **Zapytania o właściwości** - interpretowalność, wymagania GPU, czas treningu

## Uruchomienie

### Wymagania
- SWI-Prolog 9.2.9 lub nowszy

### Komendy

Załadowanie bazy wiedzy:
```bash
swipl -s src/project/lab1.pl
```

Lub w interpreterze:
```prolog
?- ['src/project/lab1.pl'].
```

## Przykłady użycia

### Podstawowe zapytania

**Top 3 modele dla datasetu:**
```prolog
?- top_modele(customer_churn, 3, T).
T = [[95, logistic_regression, [normalizacja, imputacja]], 
     [90, random_forest, [normalizacja, imputacja]], 
     [60, xgboost, [normalizacja, imputacja]]].
```

**Rekomendacja z wyjaśnieniem:**
```prolog
?- rekomenduj(customer_churn, Model, Ocena, Preprocessing, Wyjasnienie).
Model = logistic_regression,
Ocena = 95,
Preprocessing = [normalizacja, imputacja],
Wyjasnienie = "Model interpretowalny, szybki, dobry dla średnich danych".
```

**Porównanie modeli:**
```prolog
?- porownaj_modele(customer_churn, random_forest, xgboost, Lepszy).
Lepszy = random_forest.
```

**Pipeline preprocessingu:**
```prolog
?- zbuduj_pipeline(house_prices, xgboost, Pipeline).
Pipeline = [preprocessing([normalizacja, imputacja, ...]), 
            model(xgboost), 
            postprocessing([walidacja, metryki])].
```

**Więcej przykładów w pliku:** `src/project/pytania_do_projektu`

## Obsługiwane modele

- **Logistic Regression** - szybki, interpretowalny
- **Random Forest** - odporny, dobry dla średnich danych
- **XGBoost** - wysoka dokładność, obsługa niezbalansowania
- **Neural Network** - duże dane, problemy nieliniowe
- **SVM** - małe/średnie datasety, wymaga normalizacji
- **K-Means** - klasteryzacja

## Obsługiwane datasety

| Dataset | Typ problemu | Charakterystyka |
|---------|--------------|-----------------|
| `customer_churn` | Klasyfikacja binarna | Priorytet: interpretacja |
| `credit_fraud` | Klasyfikacja binarna | Bardzo niezbalansowany (0.002) |
| `house_prices` | Regresja | Outliery, wartości brakujące |
| `mnist` | Klasyfikacja wieloklasowa | Duży (60k), dane obrazkowe |
| `customer_segments` | Klasteryzacja | Dane numeryczne |

## Testowanie

Plik `pytania_do_projektu` zawiera 21 przykładowych zapytań sprawdzających działanie systemu:
- Podstawowe rekomendacje (1-5)
- Porównania i analiza (6-9)
- Preprocessing i pipeline (10-12)
- Właściwości modeli (13-17)
- Zaawansowane zapytania (18-21)

## Cel

Projekt zaliczeniowy - Zarządzanie Wiedzą

