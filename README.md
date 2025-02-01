# Assignment_3_orc
- This repo contains the final project of the optimization robot control course (chosen project: **A**).

- The following README will cover how to use the scripts, how to run some examples

<img src="Video_simulation.gif">



## Runnare il codice
in each type of pendulum (single, double, triple) you can find a file called `main_test.py` which, if run will execute the following steps according to the condition put in the variables:

1.  `OCP_step =      True`  --> Run the OCP Part
2.  `NN_step =       True`  --> Run the Neural Network Part
3.  `MPC_step =      True`  --> Run the MPC Part
4.  `RESULT_step =   True`  --> Run the RESULT saved on NPZ file

l'esecuzione verrà eseguita per ogni configurazione, nella parte di MPC vengono eseguiti 5 differenti modelli:
1. Mpc_M   --> only withM time horizon
2. Mpc_M_N  --> with M + N time horizon
3. Mpc_M_TC_NN --> with M and Terminal cost with Neural Network
4. MPC_M_TC_standard --> with M and standard Terminal cost 
5. Mpc_M_HY --> --> with M and Terminal cost with Neural Network and with standard way

to exectue file use this code:
```
python3 main_test.py
```

In case you want to change the system configurations, use the file `conf_double_pendulum.py`, which pemertte to change:
- values of M and N for the time horizon of the OCP
- simulation steps for MPC
- System weights (labeled with `w_`)
- Pendulum lengths (these only go to change at the results display level, in the simulation the pendulum lengths are taken in the URDF file)
- file save paths


Have fun and happy coding!