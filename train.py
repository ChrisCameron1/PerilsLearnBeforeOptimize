import sys
import torch
import argparse
import numpy as np
import torch.optim as optim
import random
import pickle
import torch.nn as nn
from tqdm import tqdm
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from models import NNFacilityLocationLP, linear_program_avg_distance_facility_location_setup


def toy_distribution(num_customers=2, num_facilities=2, rho=0, num_samples=1000):


	std_dev = np.sqrt(20.25)
	sig_d = std_dev
	sig_t = std_dev
	sig_bad = 3.

	demands_all = []
	travel_times_all = []
	customer_facility_distances_all = []
	features_all = []
	conditional_features = []
	base_features = []
	demand_features = []


	half = int(num_facilities/2)

	feed_forward_matrix = np.ones((num_customers, num_facilities,2))
	feed_forward_matrix = torch.FloatTensor(feed_forward_matrix)

	zs = []

	for sample in range(num_samples):
		travel_times = np.zeros((num_customers, num_facilities))

		z = rho * np.random.randn(num_facilities) * std_dev # latent variable
		zs.append(z)
		features = np.ones((num_customers, num_facilities)) * z
		demand_features.append(torch.FloatTensor(np.ones((num_customers)) * z))
		base_features.append(torch.FloatTensor(features))
		conditional_features.append(torch.FloatTensor(features*z))

		demands = 5.5 + np.sign(rho)*z + np.sqrt(1-rho**2) * sig_d * np.random.randn(num_facilities)#np.maximum(v[:,1],0)
		demands = np.maximum(demands,0)
		times2 = np.maximum(6 + np.random.randn(half) * sig_bad,0)

		for customer in range(num_customers):
			correlated_customer_travel_time = np.maximum(5.5 + z[customer] + np.sqrt(1-rho**2) * sig_t * np.random.randn(1),0)#times1[customer]
			travel_times[customer,half:] = np.ones(half)*correlated_customer_travel_time
			travel_times[customer,:half] = times2


		customer_facility_distances = (travel_times.T * demands).T
		customer_facility_distances_all.append(customer_facility_distances)

		demands_all.append(demands)
		travel_times_all.append(travel_times)
		

	return demands_all, travel_times_all, feed_forward_matrix, customer_facility_distances_all, conditional_features, base_features, demand_features, zs




def train(no_cuda=True,
	seed=24,
	lr=0.001,
	weight_decay=5e-4,
	hidden=50,
	dropout=0.5,
	rho=0,
	rho_noise=1,
	embed_dim=50,
	K=1,
	train_iters=100,
	num_samples=100,
	num_nodes=4):


	cuda = not no_cuda and torch.cuda.is_available()

	np.random.seed(seed)
	torch.manual_seed(seed)
	if cuda:
	    torch.cuda.manual_seed(seed)

	demands, travel_times, feed_forward_matrix, customer_facility_distances, conditional_features, base_features, demand_features,z = toy_distribution(num_customers=num_nodes,num_facilities=num_nodes,rho=rho, num_samples=num_samples)


	num_customers = feed_forward_matrix.shape[0]
	num_facilities = feed_forward_matrix.shape[1]

	num_graphs = len(demands)
	test_instances = list(range(int(num_graphs*0.8), num_graphs))
	valid_instances = list(range(int(num_graphs*0.6), int(num_graphs*0.8)))
	train_instances = [x for x in range(num_graphs) if x not in test_instances and x not in valid_instances]

	nfeat = feed_forward_matrix.shape[2]

	model = NNFacilityLocationLP(num_nodes,
					nfeat=nfeat,
	                nhid=hidden,
	                nout=embed_dim,
	                dropout=dropout,
	                K = K, 
	                nlayers = 2)

	optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

	# optim.SGD(model.parameters(), lr = lr, weight_decay = 0.)#


	# Get two-stage distance expected values
	demand_sum = np.zeros_like(demands[0])
	travel_time_sum = np.zeros_like(travel_times[0])
	demand_travel_time_sum = np.zeros_like(travel_times[0])
	for i in train_instances:
		demand_sum += demands[i]
		travel_time_sum += travel_times[i]
		demand_travel_time_sum += (travel_times[i].T * demands[i]).T
	two_stage_demands = demand_sum / len(train_instances)
	two_stage_travel_times = travel_time_sum / len(train_instances)
	opt_dist = demand_travel_time_sum / len(train_instances)


	two_stage_expected_distances = (two_stage_travel_times.T * two_stage_demands).T

	# Get two-stage assignment
	problem, parameters, variables = linear_program_avg_distance_facility_location_setup(num_customers, num_facilities,mip=True)
	parameters[0].value = two_stage_expected_distances
	parameters[1].value = K
	solution = problem.solve(solver= cp.GLPK_MI, verbose=False) # GLPK_MI
	twostage_assignments = variables[1].value


	# Get OPT assignment
	problem, parameters, variables = linear_program_avg_distance_facility_location_setup(num_customers, num_facilities,mip=True)
	parameters[0].value = opt_dist
	parameters[1].value = K
	solution = problem.solve(solver= cp.GLPK_MI, verbose=False) # GLPK_MI
	opt_assignments = variables[1].value


	losses = []
	losses_test = []
	losses_test_two_stage = []
	batch_losses = []
	two_stage_losses = []
	opt_losses = []
	optimizer.zero_grad()
	for t in tqdm(range(train_iters),desc="Training..."):
		
		i = np.random.choice(train_instances)
		#ground_truth_distances_i = #ground_truth_distances[i]
		facilities, assignments, predicted_dists = model(base_features[i])#model(feed_forward_matrix) #model(features, adj)
		#print(predicted_dists)
		ground_truth_demand_weighted_distance = torch.FloatTensor(customer_facility_distances[i])
		loss = torch.sum(ground_truth_demand_weighted_distance * assignments)
		loss.backward()
		losses.append(loss.item())
		batch_losses.append(loss.item())

		two_stage_loss = torch.sum(ground_truth_demand_weighted_distance * twostage_assignments)
		two_stage_losses.append(two_stage_loss.item())

		opt_loss = torch.sum(ground_truth_demand_weighted_distance * opt_assignments)
		opt_losses.append(opt_loss.item())


		if t % 10 == 0:
			optimizer.step()
			optimizer.zero_grad()

			print('End2End:%f,2Stage:%f,OPT:%f' % (np.mean(batch_losses),np.mean(two_stage_losses),np.mean(opt_losses)))
			batch_losses = []
			two_stage_losses = []
			opt_losses = []



	# Get end-to-end assignment
	end2end_facilities, end2end_assignments, predicted_dists = model(feed_forward_matrix, mip=True)


	# Turn of model learning
	model.eval()
	torch.no_grad()
	opt_losses = []
	losses_test_two_stage = []
	losses_test_two_stage_conditional = []
	losses_test = []
	for i in tqdm(test_instances, desc='Testing'):
		
		ground_truth_demand_weighted_distance = torch.FloatTensor(customer_facility_distances[i])
		
		loss = torch.sum(ground_truth_demand_weighted_distance * end2end_assignments)
		if loss.item() < 0:
			print(ground_truth_demand_weighted_distance)
			print(assignments)
			raise Exception('Loss is zero but demand-weighted distances must be strictly positive.')
		losses_test.append(loss.item())

		# Two-stage
		loss = torch.sum(ground_truth_demand_weighted_distance * twostage_assignments)
		losses_test_two_stage.append(loss.item())

		# Optimal
		opt_loss = torch.sum(ground_truth_demand_weighted_distance * opt_assignments)
		opt_losses.append(opt_loss.item())


	print('Test:')
	print('End2End:%f,2Stage:%f,OPT:%f' % (np.mean(losses_test),np.mean(losses_test_two_stage),np.mean(opt_losses)))

	return losses_test, losses_test_two_stage, opt_losses


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--no-cuda', action='store_true', default=True,
	                    help='Disables CUDA training.')
	parser.add_argument('--seed', type=int, default=24, help='Random seed.')
	parser.add_argument('--lr', type=float, default=0.01,
	                    help='Initial learning rate.')
	parser.add_argument('--weight_decay', type=float, default=5e-4,
	                    help='Weight decay (L2 loss on parameters).')
	parser.add_argument('--hidden', type=int, default=50,
	                    help='Number of hidden units.')
	parser.add_argument('--dropout', type=float, default=0.5,
	                    help='Dropout rate (1 - keep probability).')
	parser.add_argument('--rho', type=float, default=0,
	                    help='Correlated between demand and travel times')
	parser.add_argument('--rho_noise', type=float, default=1.0,
	                    help='random factor rhi upweighted by')
	parser.add_argument('--embed_dim', type=int, default=50,
	                    help='Dimensionality of node embeddings')
	parser.add_argument('--K', type=int, default=1,
	                    help='How many facilities')
	parser.add_argument('--train_iters', type=int, default=100,
	                    help='number of training iterations')
	parser.add_argument('--num_samples', type=int, default=1000,
	                    help='number of training iterations')
	parser.add_argument('--num_nodes', type=int, default=4,
	                    help='number of training iterations')

	args = parser.parse_args()

	train(no_cuda=args.no_cuda,
		seed=args.seed,
		lr=args.lr,
		weight_decay=args.weight_decay,
		hidden=args.hidden,
		dropout=args.dropout,
		rho=args.rho,
		rho_noise=args.rho_noise,
		embed_dim=args.embed_dim,
		K=args.K,
		train_iters=args.train_iters,
		num_samples=args.num_samples,
		num_nodes=args.num_nodes)








