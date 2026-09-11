# Deep literature review: la prossima estensione dopo la Sparse PCA

**Progetto proposto:** *Global-to-Local Factor and Spillover Networks in Intraday U.S. Financials*

**Sottotitolo:** separare esposizioni comuni, fattori locali e collegamenti residuali contemporanei o direzionali
**Data della ricognizione:** 10 settembre 2026

## Decisione esecutiva

La migliore estensione del progetto non è un'altra variante di PCA presa isolatamente e non è ancora un backtest di trading. È una decomposizione **global factor → local factor → residual network**, seguita da una verifica rigorosamente fuori campione.

Il progetto dovrebbe rispondere a una domanda precisa:

> I gruppi di banche emersi dopo la residualizzazione SPY/XLF sono veri fattori locali condivisi, oppure nascondono collegamenti condizionali e spillover dinamici tra singoli titoli? Quale parte della struttura è stabile e migliora una previsione fuori campione?

La proposta combina due filoni recenti e complementari:

1. l'**ℓ1-rotation criterion** di Freyaldenhoven, pubblicato nel 2026, per passare da “componenti sparse utili” a un tentativo formalmente motivato di identificare fattori locali;[^2]
2. **FNETS**, pubblicato nel 2024, per togliere la componente fattoriale e stimare sul residuo una rete Granger direzionale, una rete contemporanea di correlazioni parziali e una rete di lungo periodo.[^7]

La combinazione è particolarmente adatta a un résumé quant perché mostra, nello stesso lavoro, econometria dei fattori, ottimizzazione sparsa, serie temporali ad alta dimensione, network analysis, microstruttura, validazione walk-forward e disciplina anti-overfitting. Entrambi i pilastri hanno codice pubblico: `l1rotation` include un package R e un replication package verificato dalla rivista; `fnets` è disponibile su CRAN con selezione data-driven di fattori, lag e soglie.[^3][^8]

### Verdetto operativo

La sequenza raccomandata è:

1. **Step 13 — Local-factor identification:** confrontare PCA, Varimax, Elastic-Net Sparse PCA e ℓ1 rotation, con bootstrap a blocchi per sessione.
2. **Step 14 — Factor-adjusted networks:** stimare reti sui rendimenti residuali a 5 minuti e, come specifica principale per previsione, sulle log-realized volatility giornaliere costruite dai dati intraday.
3. **Step 15 — Stability and regimes:** misurare stabilità di supporti e archi, confrontare comunità fattoriali e comunità dinamiche, studiare finestre mobili e il regime di marzo 2023.
4. **Step 16 — Walk-forward forecasting:** confrontare factor-only, HAR-RV, sparse VAR senza aggiustamento e FNETS, con 2026 mantenuto come test finale intoccato.
5. **Step 17 opzionale — Economic layer:** solo se la previsione supera i benchmark, tradurre i segnali in una strategia residuale con costi e turnover espliciti.

Il prodotto minimo credibile richiede gli Step 13–16. Lo Step 17 è una possibile estensione, non una condizione per rendere il progetto forte.

## 1. Il punto di partenza reale del repository

Il repository dispone già di una base insolitamente buona per questo tipo di lavoro: dati SIP a un minuto dal 2023-01-01 al 2026-09-09, 924 sessioni, 12 titoli nel campione CORE, 18 nel FULL, benchmark SPY e XLF, controlli di qualità, normalizzazione intraday, residualizzazione, variance ledger, PCA rolling, Varimax e Sparse PCA.[^1]

I risultati correnti danno una motivazione quantitativa, non soltanto narrativa, alla nuova estensione:

- nella correlation PCA grezza, PC1 spiega il **63,77%** e le prime tre componenti il **75,85%**;
- dopo la residualizzazione SPY/XLF, la correlation PCA scende al **35,71%** per PC1 e al **56,11%** cumulato per le prime tre componenti;
- l'Elastic-Net Sparse PCA residuale conserva il **55,75%** di reconstruction share: una perdita di soli 0,36 punti percentuali rispetto allo spazio PCA a tre componenti;
- nella decomposizione su covarianza, SPY/XLF assorbono il **43,59%** della varianza grezza; il primo PC residuale rappresenta il **41,79%** della varianza residuale, pari al **23,58%** della varianza grezza;
- con la specifica riportata `L1=0.10`, `L2=0.10`, il primo sparse factor residuale seleziona soprattutto `USB`, `TFC`, `KEY`, `RF`, `FITB`, `CFG`, `HBAN` e un peso quasi nullo su `BAC`; il secondo seleziona `JPM`, `BAC`, `WFC`, `C`; il terzo è quasi interamente `MS`, con un peso minimo su `C`.[^1]

Questi tre insiemi sono ottime **ipotesi esplorative**: regional banks, money-center banks e una componente distinta legata a Morgan Stanley. Non sono ancora fattori identificati in senso econometrico. Il README stesso mantiene correttamente questa distinzione, trattando Varimax e Sparse PCA come strumenti interpretativi e non causali.[^1]

La domanda successiva non è quindi “posso rendere i loadings ancora più belli?”, ma:

- i supporti sopravvivono a cambi di campione, frequenza, numero di fattori e penalità?
- i titoli inclusi nello stesso fattore condividono soltanto uno shock comune o si predicono anche a vicenda?
- dopo aver rimosso sia i benchmark globali sia i fattori locali, quali collegamenti condizionali restano?
- la struttura stimata nel passato migliora una previsione realmente futura?

## 2. Perché la Sparse PCA non conclude il problema di identificazione

### 2.1 Riduzione dimensionale e identificazione sono oggetti diversi

La Sparse PCA di Zou, Hastie e Tibshirani riformula la PCA come problema di regressione penalizzata e produce componenti con pesi nulli, utili per compressione e interpretazione.[^5] Tuttavia, una soluzione sparsa non dimostra che ogni colonna stimata corrisponda a un fattore economico univoco. Inoltre, come già documentato dal progetto, le sparse components possono non essere ortogonali né ordinate per varianza spiegata.

In un modello fattoriale

\[
X_t = \Lambda F_t + \xi_t,
\]

la coppia \((\Lambda,F_t)\) è osservazionalmente equivalente a molte rotazioni. Senza restrizioni aggiuntive, assegnare un significato alla singola colonna di \(\Lambda\) non è giustificato. Freyaldenhoven separa esplicitamente questo problema dalla Sparse PCA: se esistono fattori **locali**, cioè fattori che caricano soltanto su sottoinsiemi delle variabili, la rotazione vera può essere la più sparsa; la minimizzazione della norma ℓ1 dei loadings consente allora di recuperare i singoli loading vectors sotto condizioni sufficienti.[^2]

Il vantaggio rispetto a Varimax non è solo estetico. Il lavoro del 2026 fornisce:

- condizioni di identificazione sotto sparsità esatta e approssimata;
- garanzie teoriche per il recupero dei loading vectors;
- confronti Monte Carlo favorevoli rispetto a criteri tradizionali come Varimax;
- un diagnostico per verificare se il numero di loadings piccoli è compatibile con veri fattori locali;
- due applicazioni economiche, inclusa una su 272 serie di rendimenti azionari internazionali;
- package e replication archive pubblici.[^2][^3]

Il lavoro precedente dello stesso autore chiarisce inoltre che i fattori locali possono avere forza diversa, interessare sottoinsiemi ignoti e generare lo spettro continuo di autovalori spesso osservato nelle applicazioni; usa congiuntamente autovalori e autovettori per determinare quali fattori siano abbastanza pervasivi da essere rilevanti o stimabili.[^4]

### 2.2 L'ℓ1 rotation non sostituisce automaticamente FNETS

I due metodi svolgono funzioni diverse:

| Metodo | Oggetto | Domanda a cui risponde |
|---|---|---|
| Elastic-Net Sparse PCA | componenti sparse di ricostruzione | quali titoli bastano a rappresentare bene lo spazio? |
| ℓ1 rotation | rotazione identificante di un factor-loading space | esistono loading vectors locali economicamente interpretabili? |
| FNETS/dynamic PCA | rimozione della dipendenza comune dinamica | quali dipendenze idiosincratiche restano prima di stimare la rete? |
| Sparse VAR | coefficienti laggati residuali | chi contiene informazione predittiva per chi? |
| Precision matrix | innovazioni residuali | chi resta collegato contemporaneamente, condizionando sugli altri? |

L'uso corretto è quindi parallelo e poi integrato:

- usare ℓ1 rotation per interpretare e testare i fattori locali;
- usare un factor adjustment statico e uno dinamico come specifiche concorrenti della rete;
- confrontare, **senza imporlo a priori**, se le comunità della rete coincidono con i supporti fattoriali.

Una pubblicazione del 2026 propone anche un regularized factor-augmented VAR con loadings sparsi, ottenendo fattori interpretabili e impulse responses.[^6] È una naturale estensione avanzata, ma non dovrebbe essere il primo obiettivo: chiamare “strutturale” uno shock SPY o XLF sarebbe difficile da difendere, perché XLF contiene molti titoli del campione e non è esogeno. Per la prima release è più corretto parlare di dipendenza condizionale, Granger-predictive links e generalized responses, non di causalità strutturale.

## 3. La sintesi più forte: global → local → network

### 3.1 Modello concettuale

La decomposizione proposta è:

\[
r_{i,t} = s_i(m_t)
          + \beta_{i,M,t}r_{SPY,t}
          + \beta_{i,F,t}r_{XLF,t}
          + \lambda_i' f_t
          + \xi_{i,t},
\]

dove \(s_i(m_t)\) è la stagionalità intraday stimata usando solo il passato, SPY/XLF sono la componente osservata globale/settoriale, \(f_t\) contiene i fattori locali latenti e \(\xi_t\) è il residuo idiosincratico.

Sul residuo si stima:

\[
\xi_t = \sum_{\ell=1}^{d} A_{\ell}\xi_{t-\ell}+u_t,
\qquad
\Omega = \operatorname{Cov}(u_t)^{-1}.
\]

Questa parametrizzazione produce tre oggetti distinti:

1. **rete Granger direzionale:** esiste un arco \(j\rightarrow i\) se almeno un coefficiente \((A_{\ell})_{ij}\) è stabilmente diverso da zero;
2. **rete contemporanea non direzionale:** esiste un arco \(i-j\) se \(\Omega_{ij}\neq0\), quindi le innovazioni restano linearmente dipendenti condizionando sulle altre;
3. **rete di lungo periodo:** combina transizioni e precision matrix, usando la struttura associata a \(A(1)'\Omega A(1)\), a meno di fattori di scala.

FNETS formalizza precisamente una decomposizione tra componente comune dinamica e processo idiosincratico sparse VAR; stima le tre reti mediante dynamic PCA, Yule–Walker regolarizzato ℓ1 e matrici di correlazione parziale, fornendo risultati di consistenza anche con code più pesanti della Gaussiana e fattori deboli.[^7] Il metodo estende l'idea di NETS, che combina sparse VAR e precision matrix per reti Granger e contemporanee.[^9]

L'uso congiunto di PCA e misure Granger per studiare interconnessione e rischio sistemico nel settore finanziario ha inoltre un precedente diretto in Billio et al.; la proposta qui aggiunge factor identification, regolarizzazione high-dimensional e una separazione esplicita tra rete contemporanea e predittiva.[^16]

### 3.2 La domanda empirica che rende il progetto originale

Non è necessario sostenere che i singoli metodi siano nuovi. Il contributo originale e verificabile del progetto è il confronto tra livelli di dipendenza:

- **Rete 0 — raw:** quale rete appare senza rimuovere nulla?
- **Rete 1 — global-adjusted:** cosa resta dopo SPY/XLF?
- **Rete 2 — global-and-local-adjusted:** cosa resta dopo SPY/XLF e fattori locali ℓ1-identificati?
- **Rete 3 — dynamic-factor-adjusted:** cosa resta usando il factor adjustment dinamico di FNETS?

Se gli archi entro il gruppo delle regional banks scompaiono dalla Rete 1 alla Rete 2, il risultato suggerisce che il loro legame era soprattutto esposizione a un fattore locale condiviso. Se archi direzionali stabili persistono nella Rete 2, vi è invece informazione predittiva residuale non riassunta dal fattore. Se la rete raw è molto densa e quella factor-adjusted è sparsa e stabile, il progetto mostra empiricamente perché confondere co-movimento comune e spillover produce grafi fuorvianti.

Questa motivazione è coerente con la letteratura: una sparse VAR applicata senza rimuovere forti dipendenze comuni tende a produrre rappresentazioni poco interpretabili; FNETS nasce proprio per correggere questo problema.[^7] Barigozzi e Hallin avevano già mostrato, su volatilità S&P 100, come una decomposizione dynamic-factor più sparse VAR potesse separare shock comuni e interconnessioni idiosincratiche.[^10] POET fornisce lo stesso principio generale dal lato della covarianza: struttura low-rank comune più covarianza residuale sparsa.[^11]

### 3.3 Ipotesi falsificabili

Le ipotesi vanno registrate prima di guardare il test 2026:

- **H1 — local factors:** l'ℓ1 rotation rileva più loadings piccoli di quanto atteso sotto un loading matrix denso e recupera gruppi simili a quelli dell'Elastic-Net Sparse PCA.
- **H2 — support stability:** i supporti regional, money-center e `MS` sopravvivono a bootstrap per sessione, CORE/FULL, frequenze e valori ragionevoli di \(K\).
- **H3 — factor versus link:** la rimozione dei fattori locali riduce sensibilmente densità e forza degli archi entro gruppo, ma lascia un sottoinsieme di archi residuali stabili.
- **H4 — regimes:** la struttura fattoriale e/o di rete cambia durante lo stress bancario di marzo 2023 rispetto alle finestre adiacenti.
- **H5 — economic usefulness:** un forecast factor-adjusted network migliora la previsione di realized volatility rispetto a HAR-RV, factor-only e sparse VAR non aggiustata nel test 2026.

Un risultato negativo su una o più ipotesi resta informativo. La regola scientifica è non rinominare ex post i gruppi, non cambiare la frequenza principale perché “funziona meglio” e non promuovere un arco instabile a storia economica.

## 4. Disegno dei dati e frequenze

### 4.1 Due pannelli, due funzioni diverse

Si raccomandano due pannelli derivati dalla stessa fonte SIP:

| Pannello | Costruzione | Uso principale |
|---|---|---|
| **R5** | log-return a 5 minuti, sommando cinque return a 1 minuto entro sessione | lead–lag intraday, sparse VAR, robustness microstrutturale |
| **V1D** | \(\log(RV_{i,d}+\varepsilon)\), con \(RV_{i,d}=\sum_m r_{i,d,m}^2\) da return a 5 minuti | rete di volatilità, regimi, forecast one-day-ahead |

La realized volatility integra in modo naturale i dati intraday nella misura e previsione della volatilità giornaliera.[^21] HAR-RV fornisce un benchmark parsimonioso con componenti giornaliera, settimanale e mensile.[^22] È anche l'oggetto più vicino alle applicazioni finanziarie di FNETS e della letteratura sulle reti di contagio.[^7][^10]

Il minuto non va abbandonato: serve per costruire RV e per una sensitivity analysis `1/5/10/15 minuti`. La frequenza principale per i lead–lag dovrebbe però essere 5 minuti, perché a frequenze molto alte correlazioni e direzioni possono riflettere asincronia, liquidità o microstruttura. L'Epps effect documenta la forte dipendenza delle correlazioni dalla lunghezza dell'intervallo già nei dati azionari ad alta frequenza.[^20]

### 4.2 Vincoli di costruzione indispensabili

- Nessun lag intraday deve attraversare artificialmente il confine tra chiusura di una sessione e apertura della successiva.
- La sessione anomala 2023-01-24 resta esclusa secondo la decisione già documentata.
- Il profilo di volatilità per minuto, le scale, i winsorization cutoffs, i beta SPY/XLF e ogni iperparametro devono essere stimati **solo sul training disponibile a quella data**.
- Per V1D, riportare risultati con RV a 1, 5, 10 e 15 minuti; 5 minuti è la specifica primaria, non il valore selezionato dopo aver visto il test.
- Chiamare V1D **intraday realized volatility**: non cucire implicitamente il close di una sessione all'open successivo. Se è disponibile un overnight return corretto per corporate actions, aggiungerne il quadrato come feature separata e riportare anche la specifica intraday-only.
- Conservare `volume`, `trade_count`, percentuale di zero returns e missing/contamination flags come diagnostici. Un edge che compare soltanto tra un titolo liquido e uno meno liquido a 1 minuto è sospetto.
- SPY/XLF sono controlli osservati, non shock esogeni. Riportare sempre la specifica con SPY soltanto, SPY+XLF e latent factor adjustment.

### 4.3 Split temporale

Una divisione semplice e difendibile è:

| Blocco | Periodo | Uso consentito |
|---|---|---|
| **Development/train** | 2023-01-03 – 2024-12-31 | costruzione, scelta ragionata del metodo, studio in-sample di marzo 2023 |
| **Validation** | 2025-01-01 – 2025-12-31 | scelta finale di \(K\), lag, penalità, soglie e frequenza |
| **Locked test** | 2026-01-01 – 2026-09-09 | una valutazione finale dopo il freeze della configurazione |

All'interno di train e validation si usa una procedura expanding-window o rolling-window, mai una cross-validation casuale delle righe. Marzo 2023 può essere un case study pre-specificato, ma non prova performance fuori campione perché ha già influenzato lo sviluppo del progetto.

Con circa 924 osservazioni giornaliere e 12/18 titoli, V1D è statisticamente gestibile ma non davvero “ultra-high-dimensional”. Le garanzie FNETS sono asintotiche in \(p,n\); vanno citate come motivazione, non come certificato automatico per \(p=12\). L'estensione a 50–100 istituzioni finanziarie è quindi parte importante della roadmap, non un abbellimento.

## 5. Specifica degli esperimenti

### E0 — Riproduzione dello stato corrente

Bloccare una tabella machine-readable con:

- varianza spiegata PCA raw/residual;
- reconstruction share per Varimax/SPCA;
- supporti per componente e penalty path;
- mapping di segni e permutazioni;
- hash o manifest dei dati e configurazione.

Questo diventa il riferimento contro cui misurare ogni estensione. Nessun metodo nuovo può cambiare silenziosamente l'universo o la normalizzazione.

### E1 — Identificazione dei fattori locali

1. Stimare lo spazio PCA per \(K\in\{2,3,4,5\}\), mantenendo \(K=3\) come specifica di continuità.
2. Confrontare criteri Bai–Ng, eigenvalue-ratio di Ahn–Horenstein e, se implementato, il criterio per local factors di Freyaldenhoven.[^18][^19][^4]
3. Applicare ℓ1 rotation allo stesso spazio usato da Varimax.
4. Eseguire il diagnostico di local-factor presence fornito dal metodo.
5. Allineare repliche bootstrap per segno e permutazione prima di aggregare.
6. Misurare per ciascun titolo la selection probability, per ogni coppia di loading vectors la cosine similarity e per lo spazio complessivo i principal angles.
7. Confrontare i supporti con Elastic-Net SPCA via Jaccard, adjusted Rand index (ARI) e normalized mutual information (NMI).

La porta di qualità è duplice: replicare anzitutto un esempio del replication package pubblicato e confrontare la propria implementazione Python con il package R `l1rotation`; poi applicarla ai dati del progetto.[^2][^3] Più inizializzazioni e un report degli obiettivi finali sono obbligatori, perché l'ottimizzazione della rotazione può avere minimi locali.

### E2 — Rete contemporanea factor-adjusted

Per ogni livello raw/global/global+local/dynamic-factor:

1. stimare le innovazioni o i residui;
2. standardizzarli con parametri train-only;
3. stimare una precision matrix con Graphical Lasso e una specifica CLIME/FNETS di robustezza;[^35][^36]
4. selezionare la penalità con EBIC e controllo di stabilità;
5. riportare archi, segni delle partial correlations, densità, componenti, degree/strength e stabilità bootstrap.

POET/shrinkage della covarianza va inserito come controllo per evitare che un risultato dipenda dall'inversione instabile della matrice campionaria.[^11][^33]

### E3 — Rete direzionale sparse VAR

1. Stimare VAR lag \(d\in\{1,2,3,6\}\) su R5 e un insieme parsimonioso su V1D.
2. Usare Yule–Walker ℓ1 di FNETS come specifica principale e un Elastic-Net/group-lasso across lags come controllo.
3. Imporre stabilità del VAR: tutte le radici devono essere compatibili con una rappresentazione causale/stazionaria.
4. Definire l'arco \(j\to i\) sulla presenza stabile di almeno un coefficiente laggato, registrando anche il lag e la somma firmata.
5. Valutare forecast-error variance decompositions soltanto da modelli stabili; il framework di Diebold–Yılmaz fornisce un benchmark di connectedness pesata e direzionale.[^17]

La teoria della regularizzazione per sparse high-dimensional time series chiarisce che dipendenza temporale e cross-sectional cambiano le condizioni e i tassi rispetto al Lasso i.i.d.; non basta chiamare `LassoCV` su righe casualmente mescolate.[^12]

### E4 — Stabilità, comunità e regimi

- Bootstrap a blocchi di sessioni per i fattori; blocchi temporali contigui per R5/V1D.
- Selection probability di ogni arco lungo un percorso di penalità.
- StARS-like instability per il grafo contemporaneo, dichiarando esplicitamente che l'adattamento a blocchi è pragmatico e che la teoria StARS originaria non è stata derivata per questo preciso campione dipendente.[^25]
- Stability selection come quadro generale per controllare la fragilità della selezione strutturale.[^24]
- Finestre mobili iniziali di 60 sessioni, coerenti con l'infrastruttura rolling già presente.
- Confronto tra comunità fattoriali e comunità della sparse VAR con ARI/NMI e null da permutazione delle etichette.

Lo stochastic-block VAR di Guðmundsson e Brownlees stima gruppi latenti nei quali gli spillover sono più forti within-group e propone spectral clustering sulle matrici VAR.[^13] La versione 2026 separa addirittura gruppi “giver” e “receiver”, ma richiede una cross-section più ampia ed è quindi un eccellente obiettivo per la versione FULL+ del progetto, non per CORE a 12 titoli.[^14]

Una volta stabilizzata la pipeline, la regressione locale penalizzata e la precision matrix time-varying di Chen et al. possono sostituire le finestre mobili arbitrarie.[^15] È un'estensione di ricerca, non il primo MVP: una procedura rolling trasparente è più facile da validare e spiegare in colloquio.

### E5 — Previsione walk-forward

Il target principale è \(\log RV_{i,d+1}\). I benchmark minimi sono:

1. media storica/naive persistence;
2. AR(1) per titolo;
3. HAR-RV con medie daily, weekly e monthly;[^22]
4. factor-only forecast;
5. sparse VAR senza factor adjustment;
6. SPY/XLF-adjusted sparse VAR;
7. FNETS statico e dinamico;
8. eventuale global+local-adjusted sparse VAR.

Metriche principali:

- QLIKE media cross-sectional e per titolo;
- MAE/RMSE su log RV come diagnostici;
- differenziale di loss rispetto a HAR-RV;
- coverage di prediction intervals, se prodotti;
- performance distinta per regime normale e high-volatility.

QLIKE è una scelta importante perché il target di volatilità è una proxy imperfetta; Patton identifica classi di loss robuste al rumore della proxy e mostra il ruolo di QLIKE nelle comparazioni.[^23] I differenziali fuori campione vanno valutati con test di predictive ability che mantengano l'incertezza di stima; Giacomini–White offre il quadro appropriato per modelli anche misspecified, nested o non-nested.[^26]

Poiché verranno esplorati diversi \(K\), lag, penalità e frequenze, occorre registrare tutti i tentativi e applicare una Reality Check o procedura analoga sul set finale di modelli. Riutilizzare la stessa storia per selezione e inferenza può trasformare il miglior risultato della griglia in un falso positivo.[^27]

### E6 — Layer economico opzionale

Solo dopo H5:

- costruire residual forecasts o spread neutralizzati rispetto ai fattori;
- evitare sovrapposizione di posizioni che trasformi lo stesso edge in molte scommesse duplicate;
- includere turnover, ritardo di esecuzione, scenari di spread/slippage e vincoli di liquidità;
- riportare gross e net separatamente;
- usare il numero completo dei trial nel correggere lo Sharpe per selezione e non-normalità.[^34]

Senza quote bid/ask affidabili non si deve dichiarare un backtest “tradable”; si può presentare una sensitivity analysis dei costi. Il valore scientifico centrale del progetto resta la separazione fattori/rete e il forecast, anche se nessuna strategia sopravvive ai costi.

## 6. Scorecard di validazione e criteri di stop

La forza del progetto dipenderà meno dal numero di metodi implementati e più dalla capacità di dire in anticipo che cosa conta come evidenza.

| Claim | Evidenza necessaria | Condizione che impedisce il claim |
|---|---|---|
| “Esistono fattori locali” | diagnostico ℓ1-rotation pre-specificato, stabilità a \(K\) e bootstrap | struttura compatibile con loadings densi o fattori che cambiano identità tra repliche |
| “Il gruppo regional è stabile” | selection probability per titolo, Jaccard/ARI tra finestre, CORE/FULL coerenti | gruppo prodotto da una sola penalità o da una sola finestra |
| “Esiste un collegamento residuale” | arco stabile lungo blocchi e penalità, presente dopo factor adjustment | arco solo raw, solo 1-min o dipendente da un singolo giorno |
| “A predice B” | coefficiente laggato stabile e incremental forecast gain | semplice correlazione contemporanea o artefatto di allineamento |
| “La rete cambia in stress” | distanza tra grafi oltre una distribuzione bootstrap/null | selezione visuale della finestra dopo aver visto il grafico |
| “Il modello prevede meglio” | differenziale QLIKE OOS con intervallo e correzione per model search | vantaggio soltanto in-sample o soltanto sulla specifica migliore ex post |
| “Il segnale è economicamente usabile” | performance netta su test locked, cost sensitivity e turnover | PnL lordo senza ritardo/costi o metriche scelte dopo il test |

Per supporti e archi si può pre-registrare una soglia primaria di selection probability del 70%, mostrando sensibilità 60/80%; per StARS si può usare un limite di instability del 5%. Queste sono regole operative, non verità universali. Devono essere congelate sulla validation e non ottimizzate sul 2026.

### Cosa fare se il risultato è negativo

- Se l'ℓ1 diagnostic non supporta local factors, conservare Sparse PCA come compressione descrittiva e non assegnare nomi economici forti ai componenti.
- Se i supporti sono instabili a \(p=12\), passare a FULL e poi a una cross-section point-in-time più ampia prima di concludere che non vi siano fattori locali.
- Se gli archi spariscono a 5/10 minuti, classificarli come possibili effetti microstrutturali e non come spillover.
- Se la rete è stabile ma non migliora il forecast, presentarla come struttura descrittiva, senza alpha claim.
- Se FNETS non batte HAR-RV nel test, il risultato è comunque utile: dimostra che la complessità cross-sectional non aggiunge valore predittivo in questo campione.

Questi stop criteria rendono il lavoro più credibile in colloquio, perché eliminano la necessità di trovare per forza una storia positiva.

## 7. Placebo e negative controls

Un progetto di network discovery può produrre grafi convincenti anche dal rumore. Sono quindi necessari controlli costruiti sul problema specifico:

1. **Independent within-day circular shifts:** traslare indipendentemente ciascun titolo entro la sessione per distruggere i collegamenti cross-sectional preservando distribuzione intraday e parte dell'autocorrelazione.
2. **Date shifts per V1D:** traslare circolarmente le date di ciascun titolo entro blocchi annuali/regime per rompere la sincronia senza mescolare indiscriminatamente periodi di volatilità diversa.
3. **Label permutation:** permutare le etichette di gruppo per ottenere il null di ARI/NMI tra comunità fattoriali e di rete.
4. **Lead placebo:** inserire deliberatamente una variabile futura in una pipeline di test; ogni potere predittivo indica leakage o errore di allineamento.
5. **Boundary test:** verificare con unit test che nessuna riga `close → next open` entri come lag intraday.
6. **Liquidity-delay sensitivity:** ritardare di uno step i titoli meno liquidi e verificare se la direzione degli archi si ribalta.
7. **Sampling grid:** ripetere su 1/5/10/15 minuti, mantenendo 5 minuti come specifica primaria.
8. **Factor-removal ladder:** confrontare raw, SPY, SPY+XLF, PCA statico, local-factor e dynamic-factor adjustment.
9. **Equal-density null:** confrontare metriche di centralità e comunità con grafi casuali aventi stessa densità e distribuzione dei pesi.
10. **Estimated-null simulation:** simulare un modello con la stessa persistenza e covarianza marginale ma senza cross-lag edges, poi misurare il false-edge rate dell'intera pipeline.

## 8. Confronto delle possibili estensioni

I punteggi seguenti sono direzionali, da 1 a 5, e servono a rendere trasparente il trade-off; non rappresentano una misurazione scientifica.

| Candidato | Continuità col repo | Profondità | Replicabilità | Storia da résumé | Fit con i dati attuali | Totale |
|---|---:|---:|---:|---:|---:|---:|
| **A. ℓ1 local factors + factor-adjusted networks** | 5 | 5 | 5 | 5 | 5 | **25** |
| B. Sparse mean-reverting portfolios | 4 | 4 | 5 | 4 | 3 | 20 |
| C. RMT + covariance/precision shrinkage | 4 | 4 | 5 | 3 | 4 | 20 |
| D. Sparse RFAVAR + structural impulse responses | 4 | 5 | 3 | 5 | 2 | 19 |
| E. Hawkes/order-flow network | 2 | 5 | 3 | 5 | 1 | 16 |

### A. ℓ1 local factors + factor-adjusted networks — raccomandato

È l'unico candidato che risolve contemporaneamente i due problemi già aperti dal repository: interpretazione dei fattori e identificazione dei collegamenti. Ha una replica software immediata, una versione incrementale attuabile su CORE e un'estensione realmente high-dimensional su un universo più ampio.

### B. Sparse mean-reverting portfolios — secondo progetto possibile

Box–Tiao ordina combinazioni lineari da meno a più prevedibili; d'Aspremont trasforma la ricerca di portafogli mean-reverting piccoli in sparse canonical correlation/generalized eigenvalue problems; Cuturi e d'Aspremont aggiungono una soglia di varianza per evitare spread teoricamente prevedibili ma troppo piccoli rispetto alle frizioni.[^28][^29][^30]

È un'ottima diramazione dopo il network project: usare comunità e archi stabili per limitare l'universo dei basket, anziché cercare su tutte le combinazioni. Come progetto principale è però più esposto a data snooping e costi. La classica applicazione PCA/ETF residual di Avellaneda–Lee è replicabile, ma i risultati storici non possono essere trasferiti al campione corrente e devono essere rivalutati netti di frizioni.[^31]

### C. Random Matrix Theory e shrinkage — controllo necessario, flagship debole

RMT aiuta a distinguere autovalori informativi dal bulk rumoroso; Laloux et al. mostrano il problema del “noise dressing” nelle matrici di correlazione finanziarie.[^32] Ledoit–Wolf produce matrici di covarianza ben condizionate mediante shrinkage e offre un benchmark robusto per inversione, portafogli e precision matrices.[^33]

Questa estensione dovrebbe entrare come benchmark negli Step 13–14. Da sola, tuttavia, racconta meno della domanda fattori-versus-links e rischia di apparire come un esercizio standard di covariance cleaning.

### D. Sparse RFAVAR — stretch con dati di shock esterni

Il RFAVAR del 2026 è metodologicamente molto avanzato e vicinissimo al tema dell'identificazione tramite loadings sparsi.[^6] Per usarlo bene servono però shock osservati o restrizioni strutturali credibili. Una futura versione potrebbe integrare monetary-policy surprises ad alta frequenza e studiare risposte di fattori locali e singole banche. SPY e XLF, da soli, non forniscono questa identificazione.

### E. Hawkes/order flow — non con i dati attuali

Una rete Hawkes richiederebbe tempi evento, trade signs, quote e dinamica del limit order book. Le barre a un minuto aggregano troppo. Implementarla adesso allargherebbe lo scope e renderebbe fragili le conclusioni; va considerata soltanto dopo aver acquisito dati tick/quote adeguati.

### Altre estensioni recenti da tenere in riserva

- Il modello time-varying factor-adjusted network di Chen et al. è il naturale upgrade delle rolling windows.[^15]
- Il giver/receiver block-VAR consente gruppi diversi per chi trasmette e chi riceve shock, ma ha senso dopo l'espansione della cross-section.[^14]
- Il Factor Network Autoregression del 2025 comprime molte matrici di rete osservate in pochi “network factors” con tensor PCA.[^37] È adatto quando si disponga di più layer esterni — ownership, esposizioni di bilancio, geografia, funding — non per scoprire la prima rete soltanto dai rendimenti.
- Il sparse RFAVAR può aggiungere generalized/structural impulse responses una volta risolto il problema degli shock.[^6]

## 9. Piano di implementazione nel repository

### 9.1 Moduli proposti

| Script | Responsabilità | Output essenziale |
|---|---|---|
| `13_l1_local_factor_identification.py` | ℓ1 rotation, factor-number sensitivity, diagnostic, bootstrap | loadings, support probabilities, factor alignment, comparison PCA/Varimax/SPCA |
| `14_realized_volatility_panel.py` | R5 e V1D, controllo confini di sessione, trasformazioni train-only | panel 5-min, daily log RV, sampling-frequency diagnostics |
| `15_factor_adjusted_networks.py` | raw/global/local/dynamic adjustments, sparse VAR, precision matrices | tre reti per specifica, tuning path, stability/causality checks |
| `16_network_stability_regimes.py` | bootstrap, rolling 60-session, community detection | edge probabilities, graph distances, ARI/NMI, regime tables |
| `17_walk_forward_forecasting.py` | expanding-window forecast e benchmark | QLIKE/MAE/RMSE, model comparison, locked-test report |
| `18_economic_layer.py` opzionale | mapping forecast → positions e cost model | turnover, gross/net PnL, cost surface, corrected Sharpe |

I nomi sono una proposta; la numerazione mantiene il flusso già leggibile del repository.

### 9.2 API interne utili

Conviene separare funzioni pure e testabili:

- `aggregate_intraday_returns(frame, minutes, reset_each_session=True)`;
- `fit_causal_intraday_scaler(train)` e `transform(test)`;
- `align_factors(reference, candidate)` con Hungarian matching, segno e permutazione;
- `fit_l1_rotation(loadings, n_starts, seed)`;
- `session_block_bootstrap(index, block_length, seed)`;
- `fit_sparse_var(x, lag, penalty, groups=None)`;
- `check_var_stability(coef_matrices)`;
- `estimate_precision(innovations, method, penalty)`;
- `extract_granger_graph`, `extract_contemporaneous_graph`, `extract_long_run_graph`;
- `walk_forward_splits(dates, train_end, validation_end)`;
- `qlike(realized, forecast)`.

### 9.3 Test che devono fallire se vi è leakage

- la normalizzazione di un punto non cambia quando si modificano osservazioni future;
- il beta SPY/XLF di una forecast origin dipende solo dalla storia disponibile;
- i lag non attraversano sessioni;
- la configurazione test non viene usata dal tuner;
- ogni forecast ha un timestamp strettamente precedente al target;
- fattori permutati o con segno invertito sono riallineati correttamente;
- una precision matrix stimata è simmetrica e positiva definita quando il metodo lo richiede;
- un VAR instabile non genera FEVD o forecast pubblicati;
- bootstrap e solver sono riproducibili dati seed e manifest.

### 9.4 Benchmark software e replica

La pipeline principale può restare Python, ma deve essere verificata contro due riferimenti R:

1. `l1rotation::local_factors` per almeno un dataset del replication package e poi per una matrice del progetto;[^3]
2. `fnets` per una simulazione inclusa nel package e per una versione ridotta di V1D.[^8]

La verifica non richiede che ogni numero coincida se preprocessing e solver differiscono; richiede una tolleranza definita prima, stessi supporti essenziali e reti comparabili lungo la penalty path. Una piccola tabella “Python vs reference implementation” è un artefatto da résumé molto più convincente di un notebook non testato.

### 9.5 Output finali

Il progetto completo dovrebbe produrre almeno:

1. tabella di confronto PCA/Varimax/SPCA/ℓ1 rotation;
2. heatmap dei loadings con support probability;
3. diagnostic plot per local factors;
4. matrice a quattro colonne delle reti raw/global/global+local/dynamic;
5. grafo direzionale con archi colorati per segno e spessore per stability;
6. grafo contemporaneo separato, senza mescolare i due significati;
7. decomposition plot della densità/strength rimossa a ogni layer;
8. timeline rolling di connectedness, edge turnover ed effective dimension;
9. community-alignment table con null da permutazione;
10. cumulative OOS loss difference rispetto a HAR-RV;
11. locked-test scorecard con confidence intervals e trial count;
12. model card finale con limiti, non-causalità e condizioni di riproduzione.

## 10. Roadmap realistica

### Release A — MVP serio, 3–4 settimane part-time

- Freeze di dati, split e configurazione.
- Implementazione e replica ℓ1 rotation.
- Bootstrap dei supporti sul CORE e robustezza FULL.
- V1D da return a 5 minuti.
- Rete contemporanea e sparse VAR static-factor-adjusted.
- Walk-forward contro AR(1), HAR-RV e factor-only.

**Gate:** il codice deve produrre risultati senza leggere il test 2026 durante tuning o sviluppo.

### Release B — flagship da résumé, 6–8 settimane complessive

- FNETS dinamico verificato contro R.
- Tre reti e ladder raw/global/local/dynamic.
- Rolling 60-session e caso marzo 2023.
- Stability selection, placebo e model-search correction.
- Report tecnico, figure curate e résumé bullets misurati.

### Release C — estensione di ricerca

- universo point-in-time di 50–100 financial firms, includendo broker, insurers, asset managers e regional banks;
- giver/receiver communities;
- rete time-varying locale al posto delle finestre fisse;
- RFAVAR con shock esterni credibili;
- multilayer FNAR se vengono raccolti network layers esterni;
- economic layer con quote/costi migliori.

Nell'espansione dell'universo bisogna fissare i componenti in modo point-in-time, includere delisting/corporate actions e documentare il criterio di selezione. Un universo costruito oggi e retrocesso al 2023 introdurrebbe survivorship e look-ahead bias.

## 11. Come presentarlo nel résumé e in colloquio

### Titolo consigliato

**Factor-Adjusted Spillover Networks in Intraday U.S. Financials**

### One-line description

> Built a global-to-local factor and sparse network decomposition for U.S. financial stocks using 3.5+ years of SIP intraday data, separating common exposure from conditional and Granger-predictive residual links and evaluating next-day volatility forecasts walk-forward.

### Bullet templates da completare solo con risultati reali

- Implemented and cross-validated PCA, Varimax, Elastic-Net Sparse PCA and ℓ1-rotation factor identification on 12/18 U.S. financial stocks; measured factor-support stability with session-block bootstrap.
- Estimated factor-adjusted sparse VAR and precision-matrix networks on 5-minute returns and daily realized volatility, distinguishing contemporaneous, Granger-predictive and long-run linkages across market regimes.
- Evaluated `N` frozen model variants on a locked 2026 test against HAR-RV and factor-only baselines, achieving `[Δ QLIKE]` with `[confidence interval]` after model-search correction.

Le parentesi vanno riempite soltanto dopo il test finale. Se non vi è miglioramento, il terzo bullet può diventare: “found no robust incremental forecast gain after factor adjustment, despite stable descriptive communities”, che è comunque una conclusione rigorosa.

### Narrazione da colloquio

1. **Problema:** PCA vede co-movimento, ma confonde fattori comuni e collegamenti tra titoli.
2. **Insight:** Sparse PCA ha suggerito gruppi, ma la rotational indeterminacy impediva di considerarli fattori identificati.
3. **Metodo:** ℓ1 rotation per i local factors; factor-adjusted sparse VAR e precision matrix per le reti.
4. **Rischio principale:** microstruttura e data snooping; affrontati con aggregazione multi-scala, blocchi per sessione, walk-forward e test locked.
5. **Risultato:** numeri OOS, stabilità e failure cases, non soltanto un grafo visivamente plausibile.

Questa sequenza dimostra capacità di formulare una domanda econometrica, non solo di applicare librerie.

## 12. Limiti da dichiarare esplicitamente

- **Cross-section piccola:** CORE/FULL non soddisfano automaticamente il regime asintotico dei metodi high-dimensional.
- **Benchmark endogeno:** XLF contiene imprese finanziarie del campione; è un controllo sintetico, non un intervento esterno.
- **Granger non è causalità strutturale:** un edge indica contenuto predittivo condizionale al modello e all'information set.
- **Microstruttura:** asincronia, zero returns e differenze di liquidità possono creare lead–lag apparenti.
- **Regime dependence:** un solo episodio di stress bancario non basta per generalizzare a tutte le crisi.
- **Proxy error:** realized volatility è stimata, non osservata; sampling frequency e loss function contano.
- **Model selection:** fattori, penalità, lag, frequenze e soglie moltiplicano i trial.
- **Economic value:** un miglior forecast statistico non garantisce PnL dopo costi, limiti di shorting e capacity.
- **Interpretazione dei singoli titoli:** un fattore quasi singleton come `MS` può essere segnale reale, outlier o sintomo di universo troppo stretto; deve sopravvivere all'espansione.

## 13. Conclusione

L'estensione con il miglior rapporto tra avanzamento scientifico, replicabilità e valore professionale è:

> **identificare i fattori locali con ℓ1 rotation, rimuovere separatamente esposizioni globali e locali, stimare reti residuali contemporanee/direzionali e chiedere se la struttura è stabile e utile fuori campione.**

Il punto distintivo non sarà “ho fatto una network plot”, ma una decomposizione verificabile di ciò che appare simile nei dati:

\[
\text{co-movimento globale}
\;\neq\;
\text{fattore locale}
\;\neq\;
\text{dipendenza condizionale}
\;\neq\;
\text{predittività laggata}
\;\neq\;
\text{causalità}.
\]

Questa gerarchia completa in modo naturale il lavoro già svolto e lascia almeno quattro direzioni estendibili: universo più ampio, reti time-varying, giver/receiver communities e shock strutturali esterni. È abbastanza avanzata da essere discussa con un quant researcher, ma abbastanza replicabile da poter essere implementata e falsificata con il dataset già disponibile.

## Fonti

[^1]: Evidenza del progetto: [README](README.md), [`12_factor_metrics.csv`](alpaca_us_banks_1m/reports/internal_factor_isolation/12_factor_metrics.csv), [`12_factor_loadings.csv`](alpaca_us_banks_1m/reports/internal_factor_isolation/12_factor_loadings.csv) e [`10_global_variance_decomposition.csv`](alpaca_us_banks_1m/reports/variance_decomposition/10_global_variance_decomposition.csv).

[^2]: S. Freyaldenhoven (2026), “Identification through sparsity in factor models: The ℓ1-rotation criterion,” *Quantitative Economics* 17, 461–496. [DOI e testo open access](https://doi.org/10.3982/QE2369); [replication package](https://doi.org/10.5281/zenodo.17945084).

[^3]: S. Freyaldenhoven e R. Kobler, [`l1rotation` package, versione 1.0.2](https://kobleary.github.io/l1rotation/), con documentazione, sorgente e esempio riproducibile.

[^4]: S. Freyaldenhoven (2022), “Factor models with local factors—Determining the number of relevant factors,” *Journal of Econometrics* 229, 80–102. [DOI](https://doi.org/10.1016/j.jeconom.2021.04.006); [working paper open access](https://www.philadelphiafed.org/-/media/frbp/assets/working-papers/2021/wp21-15.pdf).

[^5]: H. Zou, T. Hastie e R. Tibshirani (2006), “Sparse Principal Component Analysis,” *Journal of Computational and Graphical Statistics* 15, 265–286. [DOI](https://doi.org/10.1198/106186006X113430).

[^6]: M. Daniele e J. Schnaitmann (2026), “Sparsity-Induced Identification of Factor-Augmented VAR Models,” *Journal of Business & Economic Statistics*. [Articolo open access](https://doi.org/10.1080/07350015.2026.2671238).

[^7]: M. Barigozzi, H. Cho e D. Owens (2024), “FNETS: Factor-adjusted network estimation and forecasting for high-dimensional time series,” *Journal of Business & Economic Statistics* 42, 890–902. [DOI](https://doi.org/10.1080/07350015.2023.2257270); [versione completa open access](https://arxiv.org/html/2201.06110).

[^8]: D. Owens, H. Cho e M. Barigozzi (2023), “fnets: An R Package for Network Estimation and Forecasting via Factor-Adjusted VAR Modelling,” *The R Journal* 15, 214–239. [Articolo e documentazione](https://doi.org/10.32614/RJ-2023-070); [indice CRAN del package](https://search.r-project.org/CRAN/refmans/fnets/html/00Index.html).

[^9]: M. Barigozzi e C. Brownlees (2019), “NETS: Network Estimation for Time Series,” *Journal of Applied Econometrics* 34, 347–364. [DOI](https://doi.org/10.1002/jae.2676).

[^10]: M. Barigozzi e M. Hallin (2017), “A Network Analysis of the Volatility of High-Dimensional Financial Series,” *Journal of the Royal Statistical Society: Series C* 66, 581–605. [Articolo](https://doi.org/10.1111/rssc.12177).

[^11]: J. Fan, Y. Liao e M. Mincheva (2013), “Large Covariance Estimation by Thresholding Principal Orthogonal Complements,” *Journal of the Royal Statistical Society: Series B* 75, 603–680. [DOI](https://doi.org/10.1111/rssb.12016); [testo open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC3859166/).

[^12]: S. Basu e G. Michailidis (2015), “Regularized Estimation in Sparse High-Dimensional Time Series Models,” *Annals of Statistics* 43, 1535–1567. [DOI](https://doi.org/10.1214/15-AOS1315); [preprint](https://arxiv.org/abs/1311.4175).

[^13]: G. S. Guðmundsson e C. Brownlees (2021), “Detecting Groups in Large Vector Autoregressions,” *Journal of Econometrics* 225, 2–26. [DOI](https://doi.org/10.1016/j.jeconom.2021.03.012); [accepted manuscript](https://pure.au.dk/ws/files/366080542/Brownlees_Gudmundsson_JBES_2021_AM.pdf).

[^14]: G. S. Guðmundsson (2026), “Detecting Giver and Receiver Spillover Groups in Large Vector Autoregressions,” *Journal of Business & Economic Statistics* 44, 297–308. [DOI](https://doi.org/10.1080/07350015.2025.2526430).

[^15]: J. Chen, D. Li, Y. Li e O. Linton (2025), “Estimating Time-Varying Networks for High-Dimensional Time Series,” *Journal of Econometrics* 249, 105941. [DOI](https://doi.org/10.1016/j.jeconom.2024.105941); [accepted/open PDF](https://eprints.whiterose.ac.uk/id/eprint/212538/15/1-s2.0-S0304407624002926-main.pdf).

[^16]: M. Billio, M. Getmansky, A. Lo e L. Pelizzon (2012), “Econometric Measures of Connectedness and Systemic Risk in the Finance and Insurance Sectors,” *Journal of Financial Economics* 104, 535–559. [DOI](https://doi.org/10.1016/j.jfineco.2011.12.010); [NBER version](https://www.nber.org/papers/w16223).

[^17]: F. X. Diebold e K. Yılmaz (2014), “On the Network Topology of Variance Decompositions: Measuring the Connectedness of Financial Firms,” *Journal of Econometrics* 182, 119–134. [Articolo](https://doi.org/10.1016/j.jeconom.2014.04.012).

[^18]: S. C. Ahn e A. R. Horenstein (2013), “Eigenvalue Ratio Test for the Number of Factors,” *Econometrica* 81, 1203–1227. [DOI](https://doi.org/10.3982/ECTA8968).

[^19]: J. Bai e S. Ng (2002), “Determining the Number of Factors in Approximate Factor Models,” *Econometrica* 70, 191–221. [DOI](https://doi.org/10.1111/1468-0262.00273).

[^20]: T. W. Epps (1979), “Comovements in Stock Prices in the Very Short Run,” *Journal of the American Statistical Association* 74, 291–298. [DOI](https://doi.org/10.1080/01621459.1979.10482508).

[^21]: T. G. Andersen, T. Bollerslev, F. X. Diebold e P. Labys (2003), “Modeling and Forecasting Realized Volatility,” *Econometrica* 71, 579–625. [DOI](https://doi.org/10.1111/1468-0262.00418).

[^22]: F. Corsi (2009), “A Simple Approximate Long-Memory Model of Realized Volatility,” *Journal of Financial Econometrics* 7, 174–196. [DOI](https://doi.org/10.1093/jjfinec/nbp001).

[^23]: A. J. Patton (2011), “Volatility Forecast Comparison Using Imperfect Volatility Proxies,” *Journal of Econometrics* 160, 246–256. [DOI](https://doi.org/10.1016/j.jeconom.2010.03.034); [author PDF](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf).

[^24]: N. Meinshausen e P. Bühlmann (2010), “Stability Selection,” *Journal of the Royal Statistical Society: Series B* 72, 417–473. [DOI](https://doi.org/10.1111/j.1467-9868.2010.00740.x).

[^25]: H. Liu, K. Roeder e L. Wasserman (2010), “Stability Approach to Regularization Selection (StARS) for High Dimensional Graphical Models,” *NeurIPS*. [Paper](https://papers.nips.cc/paper_files/paper/2010/hash/301ad0e3bd5cb1627a2044908a42fdc2-Abstract.html).

[^26]: R. Giacomini e H. White (2006), “Tests of Conditional Predictive Ability,” *Econometrica* 74, 1545–1578. [DOI](https://doi.org/10.1111/j.1468-0262.2006.00718.x).

[^27]: H. White (2000), “A Reality Check for Data Snooping,” *Econometrica* 68, 1097–1126. [DOI](https://doi.org/10.1111/1468-0262.00152).

[^28]: G. E. P. Box e G. C. Tiao (1977), “A Canonical Analysis of Multiple Time Series,” *Biometrika* 64, 355–365. [DOI](https://doi.org/10.1093/biomet/64.2.355).

[^29]: A. d'Aspremont (2011), “Identifying Small Mean-Reverting Portfolios,” *Quantitative Finance* 11, 351–364. [DOI](https://doi.org/10.1080/14697688.2010.481634); [preprint](https://arxiv.org/abs/0708.3048).

[^30]: M. Cuturi e A. d'Aspremont (2013), “Mean Reversion with a Variance Threshold,” *Proceedings of Machine Learning Research* 28, 271–279. [Articolo e codice bibliografico](https://proceedings.mlr.press/v28/cuturi13.html).

[^31]: M. Avellaneda e J.-H. Lee (2010), “Statistical Arbitrage in the U.S. Equities Market,” *Quantitative Finance* 10, 761–782. [DOI](https://doi.org/10.1080/14697680903124632).

[^32]: L. Laloux, P. Cizeau, J.-P. Bouchaud e M. Potters (1999), “Noise Dressing of Financial Correlation Matrices,” *Physical Review Letters* 83, 1467–1470. [DOI](https://doi.org/10.1103/PhysRevLett.83.1467).

[^33]: O. Ledoit e M. Wolf (2004), “A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices,” *Journal of Multivariate Analysis* 88, 365–411. [DOI](https://doi.org/10.1016/S0047-259X(03)00096-4). Si veda anche O. Ledoit e M. Wolf (2012), “Nonlinear Shrinkage Estimation of Large-Dimensional Covariance Matrices,” *Annals of Statistics* 40, 1024–1060, [preprint](https://arxiv.org/abs/1207.5322).

[^34]: D. H. Bailey e M. López de Prado (2014), “The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality,” *Journal of Portfolio Management* 40, 94–107. [SSRN/DOI](https://doi.org/10.2139/ssrn.2460551).

[^35]: J. Friedman, T. Hastie e R. Tibshirani (2008), “Sparse Inverse Covariance Estimation with the Graphical Lasso,” *Biostatistics* 9, 432–441. [DOI](https://doi.org/10.1093/biostatistics/kxm045); [author PDF](https://hastie.su.domains/Papers/graph.pdf).

[^36]: T. Cai, W. Liu e X. Luo (2011), “A Constrained ℓ1 Minimization Approach to Sparse Precision Matrix Estimation,” *Journal of the American Statistical Association* 106, 594–607. [DOI](https://doi.org/10.1198/jasa.2011.tm10155); [preprint](https://arxiv.org/abs/1102.2233).

[^37]: M. Barigozzi, G. Cavaliere e G. Moramarco (2025), “Factor Network Autoregressions,” *Journal of Business & Economic Statistics* 43, 1105–1118. [DOI](https://doi.org/10.1080/07350015.2025.2476695); [open repository record](https://cris.unibo.it/handle/11585/1008627).
