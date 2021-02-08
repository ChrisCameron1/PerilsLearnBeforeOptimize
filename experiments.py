from train import train
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rhos= list(np.linspace(-1,1,11))

results_filename = 'results.csv'
with open(results_filename,'a') as f:
	f.write("rho,seed,two_stage,end_to_end,opt\n")


for seed in range(1,10):
	for rho in rhos:
		end_to_end, two_stage, opt = train(rho=rho, num_samples=1000, K=1, num_nodes=20, train_iters=500,lr=0.01,seed=seed)
		with open(results_filename,'a') as f:
			f.write("%f,%f,%f,%f,%f\n" % (rho, seed, np.mean(two_stage), np.mean(end_to_end), np.mean(opt)))



# Plot results

# plt.plot(rhos, two_stage_means, 'r',label='Two stage')
# plt.plot(rhos, end_to_end_means, 'g',label='End to end')
# plt.plot(rhos, opt_means, 'b',label='OPT')

# plt.xlabel('Rho')
# plt.ylabel('Facility Location Test Loss')
# plt.legend()

# plt.savefig('sample.png',dpi=500)



