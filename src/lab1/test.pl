:- use_module(library(lists)).

zdobylPunkty(jan , 2).
zdobylPunkty(anna, 3).
zdobylPunkty(tom, 2).
zdobylPunkty(ala, 1).
zdobylPunkty(brad, 3).
jestWDruzynie(jan, d1).
jestWDruzynie(anna, d1).
jestWDruzynie(tom, d2).
jestWDruzynie(ala, d2).
graczDruzynyMaPunkty(G, D, P) :- zdobylPunkty(G, P), jestWDruzynie(G, D).
druzynaMaListe(D, L) :- findall(P, graczDruzynyMaPunkty(_, D, P), L).
druzynaMaPunkty(D, S) :- druzynaMaListe(D, L), sum_list(L, S).
remis :- druzynaMaPunkty(d1, S1), druzynaMaPunkty(d2, S2), S1 =:= S2.
%special cases
druzynaWygrala(d1) :- druzynaMaPunkty(d1, S1), druzynaMaPunkty(d2, S2), S1 > S2.
druzynaWygrala(d2) :- \+remis, \+druzynaWygrala(d1).

dlugosc([], 0).
dlugosc([G|Ogon], Dlug) :- dlugosc(Ogon, DlOgona), Dlug is DlOgona + 1.

%nl stands for newline
wypisz([]).
wypisz([Glowa|Ogon]) :- nl, write(Glowa), wypisz(Ogon).

%wykrzynik daje nam to że jak znajdzie dopasowanie to nie szuka dalej
nalezy(X, [X|_]) :- !.
nalezy(X, [_|Ogon]) :- nalezy(X, Ogon).

%ala if
max(X, Y, X) :- X >= Y, !.
max(_X, Y, Y).

%for loop
count_to_ten(10) :- !.
count_to_ten(X) :- write(X), nl, X1 is X + 1, count_to_ten(X1).

%zasady oceniania przedmiotu := trzy oceny: kolos obliczeniowy, zadanie z prologu, zadanie z protosa? ontologia?
%dwa laby nastepne na napisanie kodu do tej oceny z prologu

