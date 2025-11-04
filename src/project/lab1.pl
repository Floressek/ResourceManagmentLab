% POLECENIE:
%
%zadanie na ocene:
%
%Należy napisać program w Prologu wyciągający wnioski dotyczące wybranej dziedziny.
%Program powinien zawierać:
%	•	pokaźny zbiór reguł,
%	•	zbiór faktów,
%	•	przykładowe zapytania demonstrujące działanie wnioskowania.
%
%Proszę zwrócić szczególną uwagę na sens i poprawność reguł.
%Powinny one wnioskowywać raczej drobne fakty, które powinny stanowić punkt wyjścia dla kolejnych reguł.
%Aby uzyskać wyższe oceny należy pokazać umiejętność posługiwania się takimi elementami języka, jak:
%	•	alternatywa, negacja (;, \+),
%	•	operatory logiczne i matematyczne,
%	•	predykaty biblioteczne,
%	•	listy,
%	•	rekurencja,
%	•	cięcia (!),
%	•	operatory specjalne (np. ->, =.., .),
%	•	wczytywanie faktów z pliku.
%
%Program nie musi wypisywać odpowiedzi na ekranie.
%Należy raczej pokazać wnioskowanie przy pomocy przykładowych zapytań.
%
%Przykładowo:
%	•	Program wnioskujący na podstawie danych pacjentów, takich jak temperatura, puls, wiek, lista objawów itp.
%Program wskazuje dla zadanego pacjenta możliwe choroby oraz sposoby ich leczenia.
%	•	Program klasyfikujący zwierzęta, pojazdy itp.

% =====================================================
% SYSTEM DOBORU MODELU MACHINE LEARNING
% =====================================================

% ================== FAKTY BAZOWE ==================

% Typy problemow ML
typ_problemu(klasyfikacja_binarna).
typ_problemu(klasyfikacja_wieloklasowa).
typ_problemu(regresja).
typ_problemu(klasteryzacja).
typ_problemu(wykrywanie_anomalii).

% Modele M
model(logistic_regression, supervised, [klasyfikacja_binarna]).
model(random_forest, supervised, [klasyfikacja_binarna, klasyfikacja_wieloklasowa, regresja]).
model(xgboost, supervised, [klasyfikacja_binarna, klasyfikacja_wieloklasowa, regresja]).
model(svm, supervised, [klasyfikacja_binarna, klasyfikacja_wieloklasowa, regresja]).
model(neural_network, supervised, [klasyfikacja_binarna, klasyfikacja_wieloklasowa, regresja]).
model(kmeans, unsupervised, [klasteryzacja]).
model(isolation_forest, unsupervised, [wykrywanie_anomalii]).

% Charakterystyki modeli
interpretable(logistic_regression).
interpretable(random_forest).

szybki_trening(logistic_regression).
szybki_predykcja(logistic_regression).

radzi_sobie_z_duzymi_danymi(xgboost).
radzi_sobie_z_duzymi_danymi(random_forest).
radzi_sobie_z_duzymi_danymi(neural_network).

dobry_dla_malych_danych(svm).
dobry_dla_malych_danych(logistic_regression).

odporny_na_overfitting(random_forest).
odporny_na_overfitting(xgboost).

radzi_sobie_z_nierownowaga(xgboost).
radzi_sobie_z_nierownowaga(random_forest).

wymaga_normalizacji(svm).
wymaga_normalizacji(neural_network).
wymaga_normalizacji(logistic_regression).

radzi_sobie_z_outlierami(random_forest).
radzi_sobie_z_outlierami(xgboost).
radzi_sobie_z_outlierami(isolation_forest).

dobry_dla_wysokowymiarowych(svm).
dobry_dla_wysokowymiarowych(neural_network).
dobry_dla_wysokowymiarowych(xgboost).

dataset(customer_churn, [
    rozmiar(50000),
    liczba_cech(20),
    typ_cech([numeryczne, kategoryczne]),
    problem(klasyfikacja_binarna),
    balans_klas(0.7),
    brakujace_wartosci(tak),
    outliery(malo),
    nieliniowy(tak),
    priorytet(interpretacja)
]).

dataset(house_prices, [
    rozmiar(5000),
    liczba_cech(80),
    typ_cech([numeryczne, kategoryczne]),
    problem(regresja),
    balans_klas(1.0),
    brakujace_wartosci(tak),
    outliery(duzo),
    nieliniowy(tak),
    priorytet(dokladnosc)
]).

dataset(mnist, [
    rozmiar(60000),
    liczba_cech(784),
    typ_cech([numeryczne]),
    problem(klasyfikacja_wieloklasowa),
    balans_klas(0.9),
    brakujace_wartosci(nie),
    outliery(malo),
    nieliniowy(tak),
    priorytet(dokladnosc)
]).

dataset(credit_fraud, [
    rozmiar(284807),
    liczba_cech(30),
    typ_cech([numeryczne]),
    problem(klasyfikacja_binarna),
    balans_klas(0.001),
    brakujace_wartosci(nie),
    outliery(duzo),
    nieliniowy(tak),
    priorytet(dokladnosc)
]).

dataset(customer_segments, [
    rozmiar(8500),
    liczba_cech(15),
    typ_cech([numeryczne, kategoryczne]),
    problem(klasteryzacja),
    balans_klas(1.0),
    brakujace_wartosci(tak),
    outliery(malo),
    nieliniowy(tak),
    priorytet(szybkosc)
]).

% ================== PREDYKATY POMOCNICZE ==================

pobierz_wartosc(Klucz, [Term|_], Wartosc) :-
    Term =.. [Klucz, Wartosc], !.
pobierz_wartosc(Klucz, [_|Ogon], Wartosc) :-
    pobierz_wartosc(Klucz, Ogon, Wartosc).

dataset_ma_ceche(Dataset, Cecha, Wartosc) :-
    dataset(Dataset, Wlasciwosci),
    pobierz_wartosc(Cecha, Wlasciwosci, Wartosc).

% badanie wymiarowosci -> >0.5 potrzebna selekcja danych
stosunek_cech_do_probek(Dataset, Stosunek) :-
    dataset_ma_ceche(Dataset, rozmiar, Rozmiar),
    dataset_ma_ceche(Dataset, liczba_cech, Cechy),
    Stosunek is Cechy / Rozmiar.

rozmiar_datasetu(Dataset, maly) :-
    dataset_ma_ceche(Dataset, rozmiar, R), R < 10000.
rozmiar_datasetu(Dataset, sredni) :-
    dataset_ma_ceche(Dataset, rozmiar, R), R >= 10000, R =< 100000.
rozmiar_datasetu(Dataset, duzy) :-
    dataset_ma_ceche(Dataset, rozmiar, R), R > 100000.

dane_zbalasowane(Dataset) :-
    dataset_ma_ceche(Dataset, balans_klas, B), B >= 0.7.

dane_bardzo_niezbalansowane(Dataset) :-
    dataset_ma_ceche(Dataset, balans_klas, B), B < 0.2.