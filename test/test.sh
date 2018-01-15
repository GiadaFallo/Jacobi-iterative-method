#!/bin/bash

# PARAMS
MAX_ITER=50
VALS=(5000 10000 15000 25000 30000)
HEADER="latency_one latency_two latency_three   latency_four    latency_five    latency_six latency_seven latency_eight   latency_nine    latency_ten latency_eleven latency_twelve latency_thirteen  latency_fourteen    latency_fifteen latency_sixteen latency_seventeen latency_eighteen   latency_nineteen    latency_twenty   iteration_numbers   precision   error   threads"

# SETTINGS
if [[ "$1" == "MIC" ]]
    then
        echo  "Running on MIC"
        THREADS=240
        THREADS_STEPS=16
        EPS=0.00001
else 
    if [[ "$1" == "TITANIC" ]]
    then
        echo  "Running on TITANIC HOST"
        THREADS=24
        THREADS_STEPS=4
        EPS=0.0001
    else 
    if [[ "$1" == "TITANIC_CUDA" ]]
    then
        echo  "Running on TITANIC, CUDA"
        EPS=0.0001
    else
        echo  "Running on XEON HOST"
        THREADS=16
        THREADS_STEPS=4
        EPS=0.00001
    fi
    fi
fi


if [[ "$1" == "TITANIC_CUDA" ]]
    then
        for N in "${VALS[@]}"
        do
            RES_CU="results/titanic/cuda_titanic_$N.csv"
            echo  "Running cuda computation for matrix dimension $N"
            echo $HEADER  > $RES_CU
            ./bin/jacobi_cuda $N $MAX_ITER $EPS >> $RES_CU
            echo  "Done"
        done

else        
    for N in "${VALS[@]}"
    do    
        # FILES
        if [[ "$1" == "MIC" ]]
            then
                RES_FF="results/mic/ff_mic_$N.csv"
                RES_SEQ="results/mic/seq_mic_$N.csv"
                RES_TH="results/mic/th_mic_$N.csv"
        else 
            if [[ "$1" == "TITANIC" ]]
            then
                RES_FF="results/titanic/ff_titanic_$N.csv"
                RES_SEQ="results/titanic/seq_titanic_$N.csv"
                RES_TH="results/titanic/th_titanic_$N.csv"
        else
                RES_FF="results/host/ff_host_$N.csv"
                RES_SEQ="results/host/seq_host_$N.csv"
                RES_TH="results/host/th_host_$N.csv"
            fi
        fi
        
        echo  "Running sequential computation for matrix dimension $N"
        echo $HEADER  > $RES_SEQ

        if [[ "$1" == "MIC" ]]
            then
                ssh mic0 "./bin/jacobi_mic s $N $MAX_ITER $EPS" >> $RES_SEQ
        else 
            if [[ "$1" == "TITANIC" ]]
            then
                ./bin/jacobi_titanic s $N $MAX_ITER $EPS >> $RES_SEQ
            else 
                ./bin/jacobi_xeon s $N $MAX_ITER $EPS >> $RES_SEQ   
            fi
        fi
        echo  "Done"

        echo  "Running thread computation for matrix dimension $N"
        echo $HEADER > $RES_TH
        for i in 1 2 $(seq 4 $THREADS_STEPS $[THREADS-1]) $THREADS
            do
                echo  "Working with $i threads"
                if [[ "$1" == "MIC" ]]
                    then
                        ssh mic0 "./bin/jacobi_mic t $N $MAX_ITER $EPS $i" >> $RES_TH
                else 
                if [[ "$1" == "TITANIC" ]]
                    then
                        ./bin/jacobi_titanic t $N $MAX_ITER $EPS $i >> $RES_TH
                    else 
                        ./bin/jacobi_xeon t $N $MAX_ITER $EPS $i >> $RES_TH  
                fi
                fi
            echo  "Done"
        done
        
        echo  "Running fastflow computation for matrix dimension $N"
        echo $HEADER  > $RES_FF
        
        for i in 1 2 $(seq 4 $THREADS_STEPS $[THREADS-1]) $THREADS
            do
                echo  "Working with $i threads"
                if [[ "$1" == "MIC" ]]
                    then
                    ssh mic0 "./bin/jacobi_mic f $N $MAX_ITER $EPS $i 10" >> $RES_FF
                else 
                if [[ "$1" == "TITANIC" ]]
                    then
                        ./bin/jacobi_titanic f $N $MAX_ITER $EPS $i 10 >> $RES_FF
                    else 
                        ./bin/jacobi_xeon f $N $MAX_ITER $EPS $i 10 >> $RES_FF
                    fi
                fi
            echo  "Done"
        done
    done
fi