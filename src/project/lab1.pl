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

% ================== FAKTY BAZOWE - DANE / DATASEY ==================

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

dane_zbalansowane(Dataset) :-
    dataset_ma_ceche(Dataset, balans_klas, B), B >= 0.7.

dane_bardzo_niezbalansowane(Dataset) :-
    dataset_ma_ceche(Dataset, balans_klas, B), B < 0.2.

wysokowymiarowy(Dataset) :-
    dataset_ma_ceche(Dataset, liczba_cech, P),
    P >= 100, !.

dobry_dla_wysokowymiarowych(svm, Dataset) :-
    wysokowymiarowy(Dataset), !.
dobry_dla_wysokowymiarowych(neural_network, Dataset) :-
    wysokowymiarowy(Dataset), !.
dobry_dla_wysokowymiarowych(xgboost, Dataset) :-
    wysokowymiarowy(Dataset), !.

% ================== REGUŁY WNIOSKOWANIA - PREPROCESSING ==================

preprocessing_potrzebny(Dataset, normalizacja) :-
    dataset_ma_ceche(Dataset, typ_cech, Typy),
    member(numeryczne, Typy).

preprocessing_potrzebny(Dataset, imputacja) :-
    dataset_ma_ceche(Dataset, brakujace_wartosci, tak).

preprocessing_potrzebny(Dataset, usuwanie_outlierow) :-
    dataset_ma_ceche(Dataset, outliery, duzo).

preprocessing_potrzebny(Dataset, class_balancing) :-
    dane_bardzo_niezbalansowane(Dataset).

preprocessing_potrzebny(Dataset, feature_selection) :-
    stosunek_cech_do_probek(Dataset, Stosunek), Stosunek > 0.5.

% ================== REGUŁY WNIOSKOWANIA - WARUNKOWANIE MODELI ==================

model_pasuje_do_problemu(Model, Problem) :-
    model(Model, _, ListaProblemow),
    member(Problem, ListaProblemow).

model_odpowiedni_do_rozmiaru(Model, Dataset) :-
   rozmiar_datasetu(Dataset, maly),
    dobry_dla_malych_danych(Model), !.
model_odpowiedni_do_rozmiaru(Model, Dataset) :-
   rozmiar_datasetu(Dataset, duzy),
    radzi_sobie_z_duzymi_danymi(Model), !.
model_odpowiedni_do_rozmiaru(_, _).

model_dla_niezbalansowanych(Model, Dataset) :-
    \+ dane_zbalasowane(Dataset),
    radzi_sobie_z_nierownowaga(Model), !.
%kiedy dane zbalansowane to kazdy model jest ok
model_dla_niezbalansowanych(_, Dataset) :-
    dane_zbalasowane(Dataset).

model_dla_outlierow(Model, Dataset) :-
    dataset_ma_ceche(Dataset, outliery, duzo),
    radzi_sobie_z_outlierami(Model), !.
model_dla_outlierow(_, Dataset) :-
   \+ dataset_ma_ceche(Dataset, outliery, duzo).

model_spelnia_priotytety(Model, Dataset) :-
    dataset_ma_ceche(Dataset, priorytet, interpretacja),
    interpretable(Model), !.
model_spelnia_priotytety(Model, Dataset) :-
    dataset_ma_ceche(Dataset, priorytet, szybkosc),
    szybki_trening(Model),
    szybki_predykcja(Model), !.
model_spelnia_priotytety(_, _).

% ================== REGUŁY WNIOSKOWANIA - DOBIERANIE MODELU ==================
model_odpowiedni(Model, Dataset) :-
   dataset_ma_ceche(Dataset, problem, Problem),
   model_pasuje_do_problemu(Model, Problem),
   model_odpowiedni_do_rozmiaru(Model, Dataset),
   model_dla_niezbalansowanych(Model, Dataset),
   model_dla_outlierow(Model, Dataset).

% ================== SCORING MODELI ==================
ocen_model(Model, Dataset, Ocena) :-
   model_odpowiedni(Model, Dataset),
   oblicz_ocene(Model, Dataset, Ocena).

oblicz_ocene(Model, Dataset, Ocena) :-
   ocena_bazowa(Model, Dataset, Baza),
   bonus_intepretacji(Model, Dataset, Bonus1),
   bonus_szybkosc(Model, Bonus2),
   bonus_odpornosc(Model, Bonus3),
   bonus_priorytet(Model, Dataset, Bonus4),
    OcenaSuma is Baza + Bonus1 + Bonus2 + Bonus3 + Bonus4,
    (OcenaSuma > 100 -> Ocena = 100 ; Ocena = OcenaSuma).

ocena_bazowa(Model, Dataset, 60) :-
    rozmiar_datasetu(Dataset, duzy),
    radzi_sobie_z_duzymi_danymi(Model), !.
ocena_bazowa(Model, Dataset, 55) :-
    rozmiar_datasetu(Dataset, maly),
    dobry_dla_malych_danych(Model), !.
ocena_bazowa(_, _, 50).

bonus_intepretacji(Model, Dataset, 15) :-
    interpretable(Model),
    dataset_ma_ceche(Dataset, priorytet, interpretacja), !.
bonus_intepretacji(Model, _, 8) :-
    interpretable(Model), !.
bonus_intepretacji(_, _, 0).

bonus_szybkosc(Model, 15) :-
    szybki_trening(Model),
    szybki_predykcja(Model), !.
bonus_szybkosc(Model, 8) :-
   (szybki_trening(Model) ; szybki_predykcja(Model)), !.
bonus_szybkosc(_, 0).

bonus_odpornosc(Model, 10) :-
    odporny_na_overfitting(Model), !.
bonus_odpornosc(_, 0).

bonus_priorytet(Model, Dataset, 15) :-
    dataset_ma_ceche(Dataset, priorytet, P),
    (
      % --- PRIORYTET: DOKŁADNOŚĆ ---
      P = dokladnosc ->
        (
          (dataset_ma_ceche(Dataset, nieliniowy, tak) ->
            (Model = xgboost ; Model = neural_network ; Model = svm)
          ; true),
          (dane_bardzo_niezbalansowane(Dataset) ->
            radzi_sobie_z_nierownowaga(Model)
          ; true),
          (dataset_ma_ceche(Dataset, outliery, duzo) ->
            radzi_sobie_z_outlierami(Model)
          ; true),
          (rozmiar_datasetu(Dataset, duzy) ->
            radzi_sobie_z_duzymi_danymi(Model)
          ; true),
          (wysokowymiarowy(Dataset) ->
            dobry_dla_wysokowymiarowych(Model, Dataset)
          ; true)
        )
      ; P = szybkosc ->
        szybki_trening(Model),
        szybki_predykcja(Model)
      % --- PRIORYTET: INTERPRETACJA ---
      ; P = interpretacja ->
        interpretable(Model)
      ; fail
    ), !.
bonus_priorytet(_, _, 0).

% ================== REKOMENDACJE ==================
rekomenduj(Dataset, Model, Ocena, Preprocessing, Wyjasnienie) :-
    ocen_model(Model, Dataset, Ocena),
    findall(P, preprocessing_potrzebny(Dataset, P), Preprocessing),
    wygeneruj_wyjasnienie(Model, Dataset, Ocena, Wyjasnienie).

wygeneruj_wyjasnienie(Model, Dataset, Ocena, Wyjasnienie) :-
    dataset_ma_ceche(Dataset, problem, Problem),
    rozmiar_datasetu(Dataset, Rozmiar),
    Wyjasnienie =.. [rekomendacja, Model, Problem, Rozmiar, Ocena].

top_modele(Dataset, N, TopModele) :-
    findall(
        [Ocena, Model, Preprocessing],
        rekomenduj(Dataset, Model, Ocena, Preprocessing, _),
        WszystkieModele
    ),
    sortuj_malejaco(WszystkieModele, PosortowaneModele),
    wez_n_pierwszych(PosortowaneModele, N, TopModele).

%helper funkcje
%sortuj_malejaco(Lista, Posortowane) :-
%    predsort(cmp, Lista, Posortowane).

sortuj_malejaco(Lista, Posortowane) :-
    sort(0, @>=, Lista, Posortowane).

%rekurencyjne wyciaganie top n elementow z listy
wez_n_pierwszych(_, 0, []) :- !. %zero el
wez_n_pierwszych([], 0, []) :- !. %pusta lista
wez_n_pierwszych([H|T], N, [H|Reszta]) :-
    N > 0,
    N1 is N - 1,
    wez_n_pierwszych(T, N1, Reszta).

% ================== PIPELINE ML ==================
zbuduj_pipeline(Dataset, Model, Pipeline) :-
    findall(P, preprocessing_potrzebny(Dataset, P), Preprocessing),
    Pipeline = [preprocessing(Preprocessing), model(Model), postprocessing([walidacja, metryki])].

waliduj_pipeline([preprocessing(Steps), model(Model)|_]) :-
    preprocessing_kompletny(Steps, Model).

preprocessing_kompletny(Steps, Model) :-
    (wymaga_normalizacji(Model) -> member(normalizacja, Steps) ; true).
% ================== PORÓWNYWANIE MODELI ==================

porownaj_modele(Dataset, Model1, Model2, Lepszy) :-
    ocen_model(Model1, Dataset, Ocena1),
    ocen_model(Model2, Dataset, Ocena2),
    (Ocena1 > Ocena2 -> Lepszy = Model1; Lepszy = Model2).

najlepszy_z_listy(Dataset, Modele, Najlepszy, NajlepszaOcena) :-
    findall(Score-Model,
            ( member(Model, Modele),
              ocen_model(Model, Dataset, Score)
            ),
            Pary),
    Pary \= [],
    keysort(Pary, PosortAsc),
    reverse(PosortAsc, [NajlepszaOcena-Najlepszy | _]).

najlepszy_z_listy(Dataset, Modele, none, 0) :-
    \+ ( member(M, Modele), ocen_model(M, Dataset, _) ).


% ================== ANALIZA DATASETU ==================

analizuj_dataset(Dataset, Analiza) :-
    dataset(Dataset, _),
    rozmiar_datasetu(Dataset, Rozmiar),
    dataset_ma_ceche(Dataset, problem, Problem),
    (dane_zbalansowane(Dataset) -> Balans = zbalansowany; Balans = niezbalansowany),
    stosunek_cech_do_probek(Dataset, Stosunek),
    Analiza = analiza(Dataset, Problem, Rozmiar, Balans, Stosunek).

modele_dla_problemu(Problem, Modele) :-
    findall(Model, model_pasuje_do_problemu(Model, Problem), Modele).

ile_modeli_dla_problemu(Problem, Liczba) :-
    findall(M, model_pasuje_do_problemu(M, Problem), Modele),
    length(Modele, Liczba).

wymaga_gpu(neural_network).

czas_treningu(Model, Dataset, szybki) :-
    szybki_trening(Model),
    rozmiar_datasetu(Dataset, R),
    R \= duzy.
czas_treningu(Model, Dataset, sredni) :-
    \+ szybki_trening(Model),
    rozmiar_datasetu(Dataset, maly).
czas_treningu(Model, Dataset, dlugi) :-
    rozmiar_datasetu(Dataset, duzy),
    (Model = neural_network ; Model = svm).
czas_treningu(_M, _D, sredni).