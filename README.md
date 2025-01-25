# Assignment_3_orc
- Questa repo contiene il progetto finale del corso di optimization robot control (progetto scelto: **A**)

- Il seguente tratterà il come utilizzare gli script, come eseguire alcuni esempi

## Runnare il codice
in ogni tipologia di pendolo (singolo, doppio, triplo) è possibile trovare un file chiamato `main_test.py` il quale se fatto partire andrà ad eseguire i seguenti sttep:
1. Creazione del dataset atttraverso la parte: "OCP_part"
2. Traning del dataset utilizzando una FNN
3. Test del MPC cone 3 configurazioni

il test del MPC viene eseguito tenendo conto di tre criteri:
1. horizon M without terminal cost;
2. horizon M with neural network function J as terminal cost;
3. horizon N + M without terminal cost.

dove ogni criterio è applicato ad ogni possibile configurazione

per runnare il codice fare:
```
python3 main_test.py
```

Nel caso in cui si volessero cambiare le configurazioni del sistema, utilizzare il file `conf_double_pendulum.py`, il quale pemertte di cambiare:
- valori di M e N per il time horizon del OCP
- step di simulazione per MPC
- pesi del sistema (targati con `w_`)
- Lunghezze del pendolo (queste vanno a modificare solo a livello di visualizzazione dei risultati, nella simulazione le lunghezze del pendolo sono prese nel file URDF)
- percorsi di salvataggio dei file
