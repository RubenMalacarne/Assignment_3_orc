# Assignment_3_orc
- Questa repo contiene il progetto finale del corso di optimization robot control (progetto scelto: **A**)

- Il seguente README tratterà il come utilizzare gli script, come eseguire alcuni esempi

width="430" height="323"> <img src="Video_simulation"   width="430" height="323">



## Runnare il codice
in ogni tipologia di pendolo (singolo, doppio, triplo) è possibile trovare un file chiamato `main_test.py` il quale se fatto partire andrà ad eseguire i seguenti sttep in base alla condizione messa nella variabili:

1.  `OCP_step =      True`  --> Run the OCP Part
2.  `NN_step =       True`  --> Run the Neural Network Part
3.  `MPC_step =      True`  --> Run the MPC Part
4.  `RESULT_step =   True`  --> Run the RESULT saved on NPZ file

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
