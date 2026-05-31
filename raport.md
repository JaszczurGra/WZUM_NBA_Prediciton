![Politechnika Poznańska Logo](https://ankiety.put.poznan.pl/images/logo.png)


<h1>Wybrane zagadnienia uczenia maszynowego <br>Raport - Predykcja nagród NBA (All-NBA i All-Rookie)</h1>
<h2> Prowadzący: dr inż. Michał Fularz

Autorstwa: Julian Mikołajczak</h2>

<!-- Projekt rozwiązuje zadanie przewidzenia, którzy zawodnicy NBA zostaną wybrani do -->
<!-- zespołów **All-NBA** (First / Second / Third Team - łącznie 15 zawodników) oraz -->
<!-- **All-Rookie** (First / Second Team - łącznie 10 zawodników) w sezonie **2025-26**. -->
<!--  -->
<!-- Problem potraktowano jako **zadanie rankingowe**: trenowany jest klasyfikator binarny -->
<!-- (wybrany vs. niewybrany), a następnie wszyscy zawodnicy są sortowani według -->
<!-- przewidzianego prawdopodobieństwa wyboru. Zawodnicy z najwyższym prawdopodobieństwem -->
<!-- zapełniają kolejno drużyny: miejsca 1–5 → First Team, 6–10 → Second Team, 11–15 → Third -->
<!-- Team (dla rookies 1–5 → First, 6–10 → Second). -->

## 0. Uruchomienie projektu 

**Krok 1 - instalacja zależności.** Wszystkie wymagane biblioteki znajdują się w pliku
`requirements.txt`:

```
pip install -r requirements.txt
```

**Krok 2 - pobranie danych.** 
Dane zostały dołączone do projektu ale można je wygenerować na nowo. W tym przypadku najpierw pobierane będą etykiety nagród (scraping z Basketball
Reference), a następnie statystyki zawodników (z NBA API). Oba skrypty zapisują pliki CSV do
`data/raw/`:

```
python data/collect_awards.py
python data/collect_stats.py
```

**Krok 3 - trening modeli.** Trening wczytuje dane historyczne, buduje cechy i zapisuje dwa
modele jeden wyspecjalizowany w wyberianiu z wszystkich zawodnikow a drugi tylko z rookies (`models/allnba_model.pkl`, `models/rookie_model.pkl`):

```
python src/train.py
```

**Krok 4 - predykcja.** Predykcja dla najnowszego sezonu zapisywana jest do pliku
wyjściowego ścieżka jet pierwszym argumentem:

```
python src/predict.py output.json
```

Dodanie flagi `--score` powoduje dodatkowe porównanie predykcji z rzeczywistymi
wynikami sezonu ogłoszonymi przez NBA i wypisanie liczby punktów według punktacji
zadania - ułatwiło to szybkie porównywanie kolejnych wersji modelu

---

## 1. Zbieranie danych

### Statystyki zawodników

Statystyki zawodników zostały pobrane z API NBA (`nba_api`, endpoint
`LeagueDashPlayerStats`) dla sezonów **1996-97 – 2025-26**. Tylko ten zakres był dostępny,
ale jest on w zupełności wystarczający. Dla każdego sezonu pobierane są:

- statystyki tradycyjne (per game): `PTS`, `REB`, `AST`, `STL`, `BLK`, `GP`, `MIN`,
  `FG_PCT`, `FG3_PCT`, `FT_PCT`,
- statystyki zaawansowane: `NET_RATING`, `AST_PCT`, `REB_PCT`, `USG_PCT`, `PIE`, `TS_PCT`,
- liczba zwycięstw drużyny (`TEAM_WINS`) z tabeli `LeagueStandingsV3`.

Każdy sezon zapisywany jest do osobnego pliku `data/raw/player_stats_{sezon}.csv`.

### Rok debiutu (rookies)

Dla rookies dodatkowo wyodrębniono **rok debiutu** do osobnego pliku
`data/raw/players_rookie_year.csv`. Mapowanie `PLAYER_ID → DEBUT_YEAR` budowane jest na
podstawie `DraftHistory` (rok draftu = rok startu pierwszego sezonu), a dla zawodników
niedraftowanych uzupełniane jest przez `CommonPlayerInfo` (pole `FROM_YEAR`). Pozwala to
później oznaczyć flagą, kto w danym sezonie był debiutantem.

### Nagrody (etykiety)

Etykiety nagród zostały zescrapowane z **Basketball Reference**:

- All-NBA: `https://www.basketball-reference.com/awards/all_league.html`
- All-Rookie: `https://www.basketball-reference.com/awards/all_rookie.html`

Dane były dostępne od **1962** roku dla All-Rookie i od **1946** dla All-NBA, jednak
wykorzystano tylko sezony istotne dla projektu, czyli **od 1996-97**, aby pokrywały się ze
statystykami zawodników.

Każdy zawodnik dostaje etykietę `tier`:

- All-NBA: 1 / 2 / 3 (First / Second / Third Team),
- All-Rookie: 1 / 2 (First / Second Team).



Dane są dzielone na dwie części:

- **historyczne** (wszystkie sezony oprócz najnowszego) - do treningu,
- **bieżące** (najnowszy sezon) - do oceny (`allnba_labels_current.csv`,
  `allrookie_labels_current.csv`).

---

## 2. Wstępne przetwarzanie danych (preprocessing)

Inżynieria cech znajduje się w pliku `src/features.py`. Tylko część pól wymaga
normalizacji.


Nazwy zawodników są normalizowane do UTF-8 i dopasowywane do statystyk po kluczu
**niewrażliwym na akcenty**, tak aby nazwiska z diakrytykami (np. Jokić, Dončić, Šarić)
poprawnie łączyły się ze swoimi etykietami, zamiast być po cichu pomijane wśród przykładów
pozytywnych.

### Pola już procentowe

Są w zakresie od 0-1 reprezntując wartości procentowe, zostają niezmienione

| Pole      | Znaczenie                                                                             |
| --------- | ------------------------------------------------------------------------------------- |
| `FG_PCT`  | Skuteczność rzutów z gry (Field Goal %) - trafione / oddane rzuty z gry               |
| `FG3_PCT` | Skuteczność rzutów za 3 punkty (3-Point %)                                            |
| `FT_PCT`  | Skuteczność rzutów wolnych (Free Throw %)                                             |
| `TS_PCT`  | True Shooting % - łączna skuteczność rzutowa uwzględniająca rzuty z gry, za 3 i wolne |
| `USG_PCT` | Usage Rate - odsetek akcji drużyny kończonych przez zawodnika, gdy jest na parkiecie  |
| `AST_PCT` | Assist % - odsetek koszy kolegów z drużyny zaliczonych po asyście zawodnika           |
| `REB_PCT` | Rebound % - odsetek dostępnych zbiórek schwytanych przez zawodnika                    |

### Pola rankingowane jako percentyl względem aktualnego sezonu

Te wartości mają różne, nieograniczone skale (np. punkty per game vs. liczba zwycięstw
drużyny), dlatego są przekształcane na percentyl. Dlatego że ranking liczony jest **per sezon**, cechy są porównywalne między epokami i
niezależne od ogólnego poziomu/tempa gry w danym roku (np. inflacja punktów we współczesnej
NBA nie zaburza modelu). Dla każdego powstaje
cecha `{nazwa}_pctile`.

| Pole         | Znaczenie                                                               |
| ------------ | ----------------------------------------------------------------------- |
| `PTS`        | Punkty na mecz (Points per game)                                        |
| `REB`        | Zbiórki na mecz (Rebounds per game)                                     |
| `AST`        | Asysty na mecz (Assists per game)                                       |
| `STL`        | Przechwyty na mecz (Steals per game)                                    |
| `BLK`        | Bloki na mecz (Blocks per game)                                         |
| `GP`         | Liczba rozegranych meczów (Games Played)                                |
| `MIN`        | Minuty na mecz (Minutes per game)                                       |
| `NET_RATING` | Różnica punktów drużyny na 100 posiadań, gdy zawodnik jest na parkiecie |
| `PIE`        | Player Impact Estimate - szacowany całościowy wkład zawodnika w grę     |
| `TEAM_WINS`  | Liczba zwycięstw drużyny zawodnika w sezonie                            |


### Dlaczego własny percentyl, a nie gotowe kolumny `*_RANK` z API

NBA API zwraca wprawdzie gotowe kolumny rankingowe (`PTS_RANK`, `REB_RANK`, …), ale są one
bezwzględną pozycją (1, 2, … N) o orientacji odwrotnej (1 = najlepszy). Liczba zawodników
N rośnie z sezonu na sezon (od ok. 440 w 1998-99 do 582 w 2025-26), więc surowy ranking nie
jest porównywalny między epokami i ma kierunek przeciwny do reszty cech („wyżej = lepiej”).

Co najważniejsze, `TEAM_WINS` pochodzi z innego endpointu (`LeagueStandingsV3`) i w ogóle
nie ma gotowego rankingu - tę cechę trzeba więc i tak zrankingować samodzielnie. A nawet
gdybyśmy chcieli skorzystać z gotowych kolumn, trzeba by je odwrócić i znormalizować przez
liczbę graczy - czyli ręcznie odtworzyć dokładnie to, co robi percentyl. Dlatego prościej
i spójniej jest policzyć go samodzielnie, jednolicie dla wszystkich 10 pól.

### Dodatkowo

Na podstawie roku debiutu wyznaczana jest flaga `is_rookie` (przetwarzany sezon == rok
  debiutu), która ułatwia potem implementacje ucznia modelu dla debiutantów

Brakujące wartości są uzupełniane zerem,

Etykiety dołączane są po znormalizowanej nazwie zawodnika, a problem klasyfikacji sprowadzono do binarnego: wybrany (dowolny tier) vs. niewybrany.

---

## 3. Cechy wykorzystane do uczenia


 Endpointy NBA API (`LeagueDashPlayerStats` w wariancie *traditional* oraz
*advanced*) zwracają **kilkadziesiąt kolumn** na zawodnika - oprócz samych statystyk także
identyfikatory (`PLAYER_ID`, `TEAM_ID`, `PLAYER_NAME`), kolumny pomocnicze oraz cały komplet
kolumn rankingowych `*_RANK` generowanych automatycznie przez API (co praktycznie podwaja
liczbę kolumn). Po odrzuceniu kolumn rankingowych, identfikatorów, nazw oraz  składowych redundantnych względem wskaźników skuteczności (np. `FGM`/`FGA`,
  `FTM`/`FTA` - ich treść zawiera się już w `FG_PCT`, `FT_PCT`, `TS_PCT`) zostaje **17 cech** które są wypisane w sekcji 2.


### Dlaczego wybrano akurat te statystyki

Dobór cech wynika wprost z tego, **na co realnie patrzą głosujący** przy wyborze do zespołów
All-NBA / All-Rookie. Cechy grupują się w kilka kategorii, z których każda wnosi inny rodzaj
informacji:

- **Produkcja (box score): `PTS`, `REB`, `AST`, `STL`, `BLK`.** Podstawowe liczby, które
  definują „wielkość” sezonu zawodnika. To pierwsze, co widzą głosujący, i historycznie
  najsilniejszy sygnał - gwiazdy All-NBA prawie zawsze są w czołówce co najmniej kilku z tych
  kategorii. `STL` i `BLK` dodatkowo wnoszą wymiar **defensywny**, którego nie widać w
  punktach.

- **Skuteczność: `FG_PCT`, `FG3_PCT`, `FT_PCT`, `TS_PCT`.** Same punkty nie wystarczą -
  liczy się też, *jak efektywnie* zawodnik je zdobywa. `TS_PCT` jest tu kluczowy, bo łączy
  rzuty z gry, za 3 i wolne w jeden wskaźnik jakości rzutowej. Pozwala to odróżnić
  wartościowego, efektywnego strzelca od zawodnika, który „nabija” punkty dużą liczbą
  nieefektywnych rzutów.

- **Rola i zaangażowanie: `USG_PCT`, `AST_PCT`, `REB_PCT`.** Opisują, jak centralną rolę
  zawodnik pełni w drużynie (`USG_PCT`) oraz w jakim stopniu napędza grę zespołu
  asystami/zbiórkami niezależnie od tempa i liczby posiadań. Są to wskaźniki względne
  (procentowe), więc dobrze porównywalne między zawodnikami i drużynami.

- **Wpływ całościowy: `NET_RATING`, `PIE`.** Pojedyncze wskaźniki „syntetyczne”, które
  próbują uchwycić łączny wpływ zawodnika na grę po obu stronach parkietu. Stanowią uzupełnienie
  surowego box score - pomagają docenić zawodników wartościowych w sposób, którego nie widać
  w pojedynczych kategoriach.

- **Dostępność: `GP`, `MIN`.** Głosujący karzą zawodników, którzy opuścili dużą część sezonu
  (czego formalnym wyrazem jest m.in. wprowadzona w 2023 r. reguła minimum 65 meczów). `GP` i
  `MIN` pozwalają modelowi nauczyć się, że nawet znakomite statystyki przy niskiej liczbie
  meczów rzadko prowadzą do wyboru.

- **Sukces drużyny: `TEAM_WINS`.** Głosowanie do All-NBA historycznie faworyzuje liderów
  wygrywających drużyn. Liczba zwycięstw to prosty, mocny sygnał kontekstowy, który koryguje
  sytuacje, gdy zawodnik z dobrymi liczbami gra w słabym zespole.

---

## 4. Model

### Jak działa `XGBClassifier` (gradient boosting)

Model to **XGBoost** w wariancie klasyfikacji binarnej (`XGBClassifier`). XGBoost należy do
rodziny **gradient boosted trees** - buduje **zespół (ensemble) płytkich drzew decyzyjnych
dodawanych po kolei**, gdzie każde kolejne drzewo uczy się poprawiać błędy całego dotychczasowego
zespołu:

1. Model startuje od prostej predykcji bazowej (np. stałego logarytmu szansy klasy
   pozytywnej).
2. Liczony jest **gradient funkcji straty** (tutaj `logloss`, czyli binarna entropia
   krzyżowa) - czyli kierunek, w którym należy poprawić predykcję dla każdego przykładu.
3. Dopasowywane jest **nowe drzewo**, które przewiduje ten gradient (intuicyjnie: „gdzie i o
   ile model się myli”).
4. Predykcja drzewa jest dodawana do zespołu, przeskalowana przez **`learning_rate`** (małe
   kroki → stabilniejsze, lepiej generalizujące uczenie).
5. Kroki 2–4 powtarzają się `n_estimators` razy.

Końcowa predykcja to suma „wkładów” wszystkich drzew, przepuszczona przez funkcję logistyczną
do postaci **prawdopodobieństwa** `P(zawodnik zostanie wybrany)`. To prawdopodobieństwo jest
właśnie wartością, według której rankingujemy zawodników. Drzewa potrafią wychwytywać
**nieliniowe progi i interakcje** między cechami (np. „wysoki `USG_PCT` *i* jednocześnie
wysoki `TS_PCT`” jest dużo lepszym sygnałem niż każda z tych cech z osobna) - czego model
liniowy nie uchwyciłby bez ręcznego dodawania interakcji.

### Konfiguracja modelu

Trening (`src/train.py`) odbywa się na **wszystkich sezonach oprócz ostatniego** - ostatni
sezon (`2025-26`) jest pomijany przy wczytywaniu danych i służy do predykcji. Użyto
następującej konfiguracji:

- `n_estimators=300` - liczba kolejno dodawanych drzew,
- `max_depth=5` - maksymalna głębokość pojedynczego drzewa (płytkie drzewa ograniczają
  przeuczenie),
- `learning_rate=0.05` - niewielki krok uczenia,
- `scale_pos_weight` - ustawiany **dynamicznie** na podstawie proporcji klas (patrz niżej),
- `eval_metric="logloss"`, `random_state=42` (powtarzalność wyników).

Trenowane są **dwa osobne modele** na tym samym zestawie cech:

- **All-NBA** - na pełnym zbiorze zawodników,
- **All-Rookie** - wyłącznie na wierszach z flagą `is_rookie`.

Modele zapisywane są do `models/allnba_model.pkl` oraz `models/rookie_model.pkl`.

### Ograniczenia monotoniczne (monotonic constraints)

Każda cecha jest ograniczona tak, aby przewidywane prawdopodobieństwo wyboru było wobec niej
**niemalejące** (`monotone_constraints = (1, …, 1)`). Ponieważ wszystkie cechy są zorientowane
„wyżej = lepiej” (percentyle i wskaźniki skuteczności), koduje to jedno proste, dziedzinowo
neutralne założenie: **więcej produkcji nigdy nie obniża oceny zawodnika.**

To standardowa regularyzacja - działa jednolicie przez cechy i **nie wskazuje, nie waży ani
nie wybiera ręcznie żadnego konkretnego zawodnika**; model nadal dokonuje całego wyboru
samodzielnie. Ograniczenie okazało się potrzebne, bo model bez niego nauczył się **odwrotnej,
fałszywej zależności** - wysoki `REB_PCT` i `AST_PCT` (które historycznie korelują z
„rolowymi” wysokimi zawodnikami spoza ścisłej czołówki) traktował jako sygnał *negatywny*, co
„zakopywało” naprawdę wybitnych, grających na rozegraniu wysokich, takich jak Nikola Jokić.
Dodanie ograniczenia koryguje to bez żadnej ręcznej interwencji i poprawia trafność w walidacji
krzyżowej między sezonami.

### Dlaczego wybrano właśnie ten model (szczegółowo)

Wybór gradient boosting / XGBoost nie jest przypadkowy - wynika z charakteru danych i zadania:

1. **Dane tabelaryczne o mieszanych skalach i z interakcjami.** Mamy ~17 heterogenicznych
   cech (percentyle, odsetki, wskaźniki syntetyczne). Zespoły drzew są od lat
   najskuteczniejszą klasą modeli na takich danych i radzą sobie z nieliniowymi progami oraz
   interakcjami bez ręcznej inżynierii (czego wymagałaby np. regresja logistyczna).

2. **Silne niezbalansowanie klas.** Wybranych zawodników jest bardzo mało w stosunku do
   wszystkich (dla All-NBA ok. 15 na ~450+ zawodników w sezonie, czyli rzędu 3% - proporcja
   ok. 33:1). XGBoost obsługuje to natywnie parametrem `scale_pos_weight`, który skaluje
   wkład klasy pozytywnej w gradiencie, dzięki czemu model nie ignoruje rzadkich pozytywów.

3. **Wsparcie dla ograniczeń monotonicznych.** To kluczowa, rzadka cecha - pozwoliła wprost
   zakodować założenie „wyżej = lepiej” i naprawić problem z Jokiciem (patrz wyżej). Niewiele
   rodzin modeli udostępnia tak czyste, wbudowane ograniczenia monotoniczne.

4. **Odporność i niewielki preprocessing.** Drzewa są niewrażliwe na skalę cech i odstające
   wartości - nie wymagają standaryzacji ani usuwania outlierów. Percentyl liczymy z innego
   powodu (porównywalność między sezonami), a nie dlatego, że model tego wymaga.

5. **Interpretowalność.** Można odczytać **ważność cech** (feature importance), co jest cenne
   w raporcie i przy diagnozowaniu modelu (np. wykrycie wspomnianej odwrotnej zależności).

6. **Wynik probabilistyczny pod ranking.** Model zwraca prawdopodobieństwo, które idealnie
   pasuje do sformułowania zadania jako rankingu top-N.

7. **Ograniczenie projektowe.** Zadanie zakładało **niekorzystanie z głębokich sieci
   neuronowych** - XGBoost spełnia ten warunek, oferując przy tym bardzo dobrą skuteczność na
   danych tej wielkości (przy zaledwie tysiącach obserwacji głębokie sieci i tak zwykle
   przegrywają z boostingiem).

Dla porównania rozważane alternatywy mają wady w tym konkretnym zadaniu: **regresja
logistyczna** jest liniowa (gubi interakcje), **las losowy** nie skaluje wkładu klasy
pozytywnej w gradiencie tak wygodnie i trudniej w nim o czyste ograniczenia monotoniczne,
**SVM / kNN** są wrażliwe na skalę i nie dają tak naturalnego, dobrze skalibrowanego rankingu
prawdopodobieństw, a pojedyncze **drzewo decyzyjne** łatwo się przeucza. Stąd XGBoost jako
najlepszy kompromis między skutecznością, obsługą niezbalansowania, możliwością narzucenia
wiedzy dziedzinowej i interpretowalnością.

---

## 5. Walidacja i ocena (scoring)

Dodano funkcję oceniającą, która pozwala natychmiast zobaczyć skuteczność modelu. Działa ona
dwojako.

### Walidacja krzyżowa Leave-One-Season-Out - funkcja `evaluate_logo`

Podczas rozwoju modelu stosowana jest **walidacja krzyżowa „leave-one-season-out”**
zrealizowana funkcją:

```python
def evaluate_logo(X, y, groups, n_select: int, label: str):
    logo = LeaveOneGroupOut()
    ...
```

Działanie i znaczenie argumentów:

- **`LeaveOneGroupOut`** to schemat walidacji z biblioteki scikit-learn, w którym każdy
  „fold” pomija **całą jedną grupę**. Jako grupy (`groups`) podajemy **sezony** (`SEASON`).
  W efekcie w każdym foldzie model trenuje na wszystkich sezonach oprócz jednego i jest
  testowany na tym jednym pominiętym sezonie.
- **`X`, `y`** - macierz cech i binarne etykiety (wybrany / niewybrany).
- **`n_select`** - ile osób wybrać w pominiętym sezonie zgodnie z realnymi rozmiarami
  zespołów: **15** dla All-NBA (3 piątki) i **10** dla All-Rookie (2 piątki). Dla każdego
  pominiętego sezonu bierzemy top-`n_select` zawodników o najwyższym przewidzianym
  prawdopodobieństwie i liczymy **pokrycie (overlap)** z faktycznie wybranymi zawodnikami w
  tym sezonie.
- **`label`** - etykieta opisowa (np. `"All-NBA"` / `"All-Rookie"`) używana wyłącznie do
  czytelnego wypisywania wyników.

**Dlaczego akurat taki schemat, a nie zwykły K-fold?** Bo dokładnie odwzorowuje on prawdziwe
zadanie: przewidzieć sezon, którego model **nigdy nie widział**. Zwykły losowy podział mógłby
umieścić zawodników z tego samego sezonu (a nawet sąsiednie sezony tego samego, silnie
skorelowanego zawodnika) po obu stronach podziału, co zawyżałoby wynik przez „przeciek”
informacji. Grupowanie po sezonie eliminuje ten problem. Naturalną miarą jest też **pokrycie
top-N**, a nie zwykła dokładność wiersz-po-wierszu - bo finalnie wybieramy stałą liczbę osób
do drużyn. Dla orientacji: losowy baseline trafia ~3%, a dobry model osiąga 60%+ pokrycia.

### Ocena predykcji końcowej - flaga `--score`

Drugi tryb to **ocena predykcji** względem rzeczywistych etykiet ostatniego sezonu -
uruchamiana przez dodanie flagi `--score` do `src/predict.py` według zasad w instruckji max 450. punktów, ponieważ branę pod uwagę są jedynie oficjalne wyniki NBA


## 6. Niezbalansowanie klas: All-NBA vs All-Rookie

Warto zwrócić uwagę na istotną różnicę w **proporcji klas** między oboma zadaniami, mimo że
All-Rookie ma *mniej* zespołów 2 (10 osób) zamiast 3 (15 osób):

- **All-NBA:** wybieranych jest 15 zawodników z całej ligi - czyli z puli rzędu 450+
  zawodników w sezonie. Daje to bardzo mały odsetek pozytywów (proporcja ~31.5:1).
- **All-Rookie:** wybieranych jest 10 zawodników, ale **wyłącznie spośród debiutantów** -
  a tych grających na zauważalnym poziomie jest w sezonie zaledwie kilkudziesięciu. W efekcie
  **stosunek liczby wybranych do liczby kandydatów jest dużo wyższy dla rookies** niż dla
  All-NBA (proporcja ~7.2:1).

Innymi słowy: choć rookies mają tylko dwa zespoły zamiast trzech, ich pula kandydatów jest na
tyle mała, że **niezbalansowanie klas dla modelu rookie jest znacznie łagodniejsze**. Ma to
praktyczne konsekwencje: dynamicznie dobierany `scale_pos_weight` jest dla modelu rookie
odpowiednio mniejszy, a samo „trafienie” jest statystycznie łatwiejsze niż w przypadku
All-NBA (mniejsza i bardziej jednorodna pula kandydatów).
