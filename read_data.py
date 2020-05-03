# nll - no gamblers
# gmblers - no gamblers for 5 epochs, then gamblers with o=1.9
# autosched - no gamblers for 5 epochs, then scheduling
import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = "run_data"
search_name = "nll"

for search_name in ["nllpair", "gmblerspair", "coteachingpair", "autoschedpair", "lqpair"]:
	print("test: {}".format(search_name))
	print("__________________________")
	files = list(os.listdir(data_dir))
	files.sort(key=lambda x: x.split('_')[1])
	for file in files:
		if file.split('_')[0] == search_name and file.split('_')[2] == "0.001" and file.endswith("test_acc.npy"):
			data = np.load("{}/{}".format(data_dir, file))
			print("noise rate: {}".format(file.split('_')[1]))
			print(np.mean(data[-5:]))
			# print(data[-5:])
			print(np.std(data[-5:]))
			print(np.amax(data))
			plt.plot(np.arange(0, len(data), 1), data, label=file.split('_')[1])
			# print(",".join([str(i) for i in data]))
			# plt.plot(a)
		# os.rename(wav_files[i], "time{:04d}.wav".format(i+1))
	plt.legend()

	plt.savefig('plot_{}.png'.format(search_name))
	plt.clf()
