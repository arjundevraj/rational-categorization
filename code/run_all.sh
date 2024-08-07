#!/bin/bash

cd experiment1
echo "behavioral accuracy for control condition of experiment 1"
python behavioral_accuracy_control.py
echo "behavioral accuracy for experimental condition of experiment 1"
python behavioral_accuracy_experimental.py
echo "prototype and exemplar model fits for control condition of experiment 1"
python control_models.py
echo "prototype and exemplar model fits for experimental condition of experiment 1"
python experimental_models.py

cd ../experiment2
echo "behavioral accuracy for both conditions of experiment 2"
python behavioral_accuracy.py
for (( i=0; i <= 4; i+=2 ))
do
	echo "prototype and exemplar model fits for control condition of experiment 2 with $i exceptions"
    python control_models.py $i
	echo "prototype and exemplar model fits for experimental condition of experiment 2 with $i exceptions"
	python experimental_models.py $i
done

echo "Now performing the analysis from Appendix A"
cd ../appendix_a
echo "control condition with guessing-rate parameter"
python models.py control
echo "experimental condition with guessing-rate parameter"
python models.py experimental
echo "control condition with guessing-rate and response-scaling parameters"
python models.py control -r 
echo "experimental condition with guessing-rate and response-scaling parameters"
python models.py experimental -r 
echo "control condition with guessing-rate parameter and forgetting function"
python models.py control -f
echo "experimental condition with guessing-rate parameter and forgetting function"
python models.py experimental -f
echo "control condition with guessing-rate + response-scaling parameters and forgetting function"
python models.py control -r -f
echo "experimental condition with guessing-rate + response-scaling parameters and forgetting function"
python models.py experimental -r -f

echo "Now performing the analysis from Appendix B"
cd ../appendix_b
echo "control condition baseline"
python models_baseline.py control
echo "experimental condition baseline"
python models_baseline.py experimental
echo "control condition with guessing-rate parameter"
python models.py control
echo "experimental condition with guessing-rate parameter"
python models.py experimental
echo "control condition with guessing-rate and response-scaling parameters"
python models.py control -r 
echo "experimental condition with guessing-rate and response-scaling parameters"
python models.py experimental -r 
echo "control condition with guessing-rate parameter and forgetting function"
python models.py control -f
echo "experimental condition with guessing-rate parameter and forgetting function"
python models.py experimental -f
echo "control condition with guessing-rate + response-scaling parameters and forgetting function"
python models.py control -r -f
echo "experimental condition with guessing-rate + response-scaling parameters and forgetting function"
python models.py experimental -r -f

echo "Now performing the analysis from Appendix C"
cd ../appendix_c
echo "behavioral accuracy over trial segments segmented by stimulus"
python behavioral_results_per_stim.py

echo "Now performing the analysis from Appendix D"
cd ../appendix_d
echo "comparing model and observed responses for the control condition"
python responses.py control
echo "comparing model and observed responses for the experimental condition"
python responses.py experimental

echo "Now performing the analysis from Appendix E"
cd ../appendix_e
echo "model fits on simulated data generated from the exemplar model"
python models_on_simulated_data.py exemplar
echo "model fits on simulated data generated from the prototype model"
python models_on_simulated_data.py prototype
echo "model fits on simulated data generated from the exemplar model, only considering trials after the last exemplar has been seen"
python models_on_simulated_data.py exemplar -a
echo "model fits on simulated data generated from the prototype model, only considering trials after the last exemplar has been seen"
python models_on_simulated_data.py prototype -a
echo "model fits on for the experimental condition after fixing the number of exemplars seen"
python models_fixed_exemplar_count.py