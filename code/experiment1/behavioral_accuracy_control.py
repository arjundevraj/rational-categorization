# Usage: python behavioral_accuracy_control.py

import csv
import matplotlib.pyplot as plt
import numpy as np

responses = []
results = [[], [], [], [], [], [], [], [], [], [], []]
NUM_TRIALS = 616
NUM_TRIALS_PER_BIN = 56

# load all participant reponses
with open("../../data/experiment1_data.csv") as fp:
	reader = csv.reader(fp)
	for row in reader:
		responses.append([int(x) for x in row])

for response in responses:
	csv_num = response[0]
	# skip participants who received the experimental condition
	if csv_num % 2 != 0:
		continue
	stim_list = []
	filename = '../../stimuli/stim_list_' + str(csv_num) + '.csv'
	with open(filename, 'r') as f:
		for rec in csv.reader(f, delimiter=","):
			for x in rec:
				stim_list.append(int(x))
	
	vals = response[1:len(response)]
	# compute the participant accuracy in each bin/trial segment
	for start_trial in range(0, NUM_TRIALS, NUM_TRIALS_PER_BIN):
		num_correct = 0
		for i in range(start_trial, start_trial + NUM_TRIALS_PER_BIN):
			if (stim_list[i] < 7 and vals[i] == 1) or (stim_list[i] >= 7 and vals[i] == 2):
				num_correct += 1

		results[start_trial // NUM_TRIALS_PER_BIN].append(num_correct / NUM_TRIALS_PER_BIN)

# Graph the results
graph_data = []
errors = []
for result in results:
	graph_data.append(sum(result)/len(result))
	errors.append(np.std(result) / np.sqrt(len(result)))

trial_segs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
	"font.size": 13
})

# Shade the area between y1 and y2
y1 = [val - err for val, err in zip(graph_data, errors)]
y2 = [val + err for val, err in zip(graph_data, errors)]

plt.fill_between(trial_segs, y1, y2, facecolor="grey", color='grey', alpha=0.2)

plt.plot(trial_segs, graph_data,'k')

plt.xlabel('Trial Segment', fontsize=15)
plt.ylabel('Average Accuracy', fontsize=15)
plt.ylim(0.6, 0.95)
plt.xticks(trial_segs)
plt.show()