# Raport — Predykcja nagród NBA (All-NBA i All-Rookie)

## 1. Zbieranie danych

### Statystyki zawodników
Statystyki zawodników zostały pobrane z API NBA (`nba_api`, endpoint `LeagueDashPlayerStats`)
dla sezonów **1996-97 – 2025-26**. Tylko ten zakres był dostępny, ale jest on w zupełności
wystarczający. Dla każdego sezonu pobierane są:
- statystyki tradycyjne (per game): `PTS`, `REB`, `AST`, `STL`, `BLK`, `GP`, `MIN`, `FG_PCT`, `FG3_PCT`, `FT_PCT`,
- statystyki zaawansowane: `NET_RATING`, `AST_PCT`, `REB_PCT`, `USG_PCT`, `PIE`, `TS_PCT`,
- liczba zwycięstw drużyny (`TEAM_WINS`) z tabeli `LeagueStandingsV3`.

Każdy sezon zapisywany jest do osobnego pliku `data/raw/player_stats_{sezon}.csv`.

### Rok debiutu (rookies)
Dla rookies dodatkowo wyodrębniono **rok debiutu** do osobnego pliku
`data/raw/players_rookie_year.csv`. Mapowanie `PLAYER_ID → DEBUT_YEAR` budowane jest na
podstawie `DraftHistory` (rok draftu = rok startu pierwszego sezonu), a dla zawodników
niedraftowanych uzupełniane jest przez `CommonPlayerInfo` (pole `FROM_YEAR`).
Pozwala to później oznaczyć flagą, kto w danym sezonie był debiutantem.

### Nagrody (etykiety)
Etykiety nagród zostały zescrapowane z **Basketball Reference**:
- All-NBA: `https://www.basketball-reference.com/awards/all_league.html`
- All-Rookie: `https://www.basketball-reference.com/awards/all_rookie.html`

Dane były dostępne od **1962** roku dla All-Rookie i od **1946** dla All-NBA, jednak
wykorzystano tylko sezony istotne dla projektu, czyli **od 1996-97**, aby pokrywały się
ze statystykami zawodników.

Każdy zawodnik dostaje etykietę `tier`:
- All-NBA: 1 / 2 / 3 (First / Second / Third Team),
- All-Rookie: 1 / 2 (First / Second Team).

Dane są dzielone na dwie części:
- **historyczne** (wszystkie sezony oprócz najnowszego) — do treningu,
- **bieżące** (najnowszy sezon) — do oceny (`allnba_labels_current.csv`, `allrookie_labels_current.csv`).

## 2. Wstępne przetwarzanie danych (preprocessing)

Inżynieria cech znajduje się w pliku `src/features.py`. Tylko część pól wymaga normalizacji.

### Pola już procentowe (zakres 0–1) — pozostają niezmienione

Te statystyki są z natury ułamkami/odsetkami, więc są już porównywalne między sezonami
i zawodnikami — nie ma potrzeby ich przekształcać.

| Pole | Znaczenie |
|------|-----------|
| `FG_PCT`  | Skuteczność rzutów z gry (Field Goal %) — trafione / oddane rzuty z gry |
| `FG3_PCT` | Skuteczność rzutów za 3 punkty (3-Point %) |
| `FT_PCT`  | Skuteczność rzutów wolnych (Free Throw %) |
| `TS_PCT`  | True Shooting % — łączna skuteczność rzutowa uwzględniająca rzuty z gry, za 3 i wolne |
| `USG_PCT` | Usage Rate — odsetek akcji drużyny kończonych przez zawodnika, gdy jest na parkiecie |
| `AST_PCT` | Assist % — odsetek koszy kolegów z drużyny zaliczonych po asyście zawodnika |
| `REB_PCT` | Rebound % — odsetek dostępnych zbiórek schwytanych przez zawodnika |

### Pola spoza zakresu 0–1 — rankingowane jako percentyl względem aktualnego sezonu

Te wartości mają różne, nieograniczone skale (np. punkty per game vs. liczba zwycięstw drużyny),
dlatego są przekształcane na percentyl w obrębie sezonu. Dla każdego powstaje cecha `{nazwa}_pctile`.

| Pole | Znaczenie |
|------|-----------|
| `PTS`        | Punkty na mecz (Points per game) |
| `REB`        | Zbiórki na mecz (Rebounds per game) |
| `AST`        | Asysty na mecz (Assists per game) |
| `STL`        | Przechwyty na mecz (Steals per game) |
| `BLK`        | Bloki na mecz (Blocks per game) |
| `GP`         | Liczba rozegranych meczów (Games Played) |
| `MIN`        | Minuty na mecz (Minutes per game) |
| `NET_RATING` | Różnica punktów drużyny na 100 posiadań, gdy zawodnik jest na parkiecie |
| `PIE`        | Player Impact Estimate — szacowany całościowy wkład zawodnika w grę |
| `TEAM_WINS`  | Liczba zwycięstw drużyny zawodnika w sezonie |

### Jak działa percentyl

Percentyl liczony jest osobno w obrębie każdego sezonu:

```python
df[f"{col}_pctile"] = df.groupby("SEASON")[col].rank(pct=True)
```

Dla danej kolumny i danego sezonu wszyscy zawodnicy są ustawiani w kolejności od najmniejszej do
największej wartości, a następnie ich pozycja w tym rankingu jest przeliczana na ułamek z zakresu
**0–1** (`pct=True`). Oznacza to:
- wartość **1.0** → najlepszy zawodnik w danym sezonie pod względem tej statystyki,
- wartość **0.5** → mediana (połowa zawodników ma mniej, połowa więcej),
- wartość **bliska 0** → jedna z najniższych wartości w sezonie.

Przykład: jeśli w sezonie 2010-11 zawodnik zdobywał 28 PTS/mecz i był to drugi najwyższy wynik
spośród 400 zawodników, jego `PTS_pctile` wyniesie ok. 0.9975.

Dzięki temu, że ranking liczony jest **per sezon**, cechy są porównywalne między epokami i
niezależne od ogólnego poziomu/tempa gry w danym roku (np. inflacja punktów we współczesnej NBA
nie zaburza modelu) — liczy się **pozycja względem rówieśników**, a nie surowa liczba.

Dodatkowo:
- na podstawie roku debiutu wyznaczana jest flaga `is_rookie` (sezon zawodnika == rok debiutu),
- brakujące wartości są uzupełniane zerem,
- etykiety dołączane są po znormalizowanej nazwie zawodnika (małe litery, bez akcentów i znaków
  specjalnych), a problem klasyfikacji sprowadzono do binarnego: wybrany (dowolny tier) vs. niewybrany.

## 3. Trening

Trening (`src/train.py`) odbywa się na **wszystkich sezonach oprócz ostatniego** —
ostatni sezon (`2025-26`) jest pomijany przy wczytywaniu danych i służy do predykcji.

Wykorzystano model **XGBoost** (`XGBClassifier`) w konfiguracji:
- `n_estimators=300`, `max_depth=5`, `learning_rate=0.05`,
- `scale_pos_weight` ustawiany dynamicznie na podstawie proporcji klas (mocne niezbalansowanie —
  wybranych zawodników jest niewielu),
- `eval_metric="logloss"`, `random_state=42`.

Trenowane są dwa osobne modele:
- **All-NBA** — na pełnym zbiorze zawodników,
- **All-Rookie** — wyłącznie na wierszach z flagą `is_rookie`.

Modele zapisywane są do `models/allnba_model.pkl` oraz `models/rookie_model.pkl`.

### Funkcja oceniająca (scoring)
Dodano funkcję oceniającą, która pozwala natychmiast zobaczyć skuteczność modelu.
Działa ona dwojako:

- **Walidacja krzyżowa Leave-One-Season-Out** podczas treningu — dla każdego pominiętego
  sezonu liczone jest pokrycie (overlap) między top-N predykcji a faktycznie wybranymi
  zawodnikami (top-15 dla All-NBA, top-10 dla All-Rookie).

- **Ocena predykcji** względem rzeczywistych etykiet ostatniego sezonu — uruchamiana przez
  dodanie flagi `--score` do predykcji:

  ```
  python src/predict.py output.json --score
  ```

  Punktacja (maks. 900) premiuje trafienie zawodnika w danym tierze: 10 pkt za dokładny tier,
  8 pkt przy różnicy o 1, 6 pkt przy różnicy o 2, plus bonus za liczbę dokładnych trafień
  w drużynie (2→5, 3→10, 4→20, 5→40 pkt).



