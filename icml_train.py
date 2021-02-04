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

from models import GCNFacilityLocationLP, linear_program_avg_distance_facility_location_setup


def correlated_time_demand_data(num_customers=2, num_facilities=2, rho=0, num_samples=1000, rho_noise=1):

	num_nodes = num_customers + num_facilities

	customers = list(range(num_nodes))[:num_customers]
	facilities = list(range(num_nodes))[num_customers:]

	# Uniform random 2d vector between 0 and 1
	num_features = 2
	features = 5*np.random.rand(num_nodes,num_features)
	mean_travel_times = np.zeros((num_customers, num_facilities))
	for i in range(num_customers):
		for j in range(num_facilities):
			mean_travel_times[i,j] = np.inner(features[i,:], features[j,:])


	print(mean_travel_times)
	#exit()

	features = torch.FloatTensor(features)

	mean_demands = np.zeros((num_customers,1))
	for i in range(num_customers):
		mean_demands[i] = np.inner(features[i,:], features[i,:])



	# Sample from a n+1 dimensional gaussian for every node. When 0th index is customer and 1-n+1 is travel_times to facilities. Travel_times are only correlated through effect on demand
	customer_info = []
	for customer in range(num_customers):

		mean = np.ones(num_facilities + 1)
		cov = np.ones((num_facilities+1, num_facilities+1))
		for index in range(num_facilities):
			rho_val = rho #* (1 + random.uniform(0,rho_noise))
			X = np.random.rand(num_facilities+1, num_facilities+1)
			X = np.matmul(X.T,X)
			rho_matrix = rho * np.ones((num_facilities+1, num_facilities+1)) 
			diag_ind = np.diag_indices_from(rho_matrix)
			rho_matrix[diag_ind[0], diag_ind[1]] = torch.ones(rho_matrix.shape[0])
			X[diag_ind[0], diag_ind[1]] = torch.ones(rho_matrix.shape[0])

			cov = X * rho_matrix



			min_eig = np.min(np.real(np.linalg.eigvals(cov)))
			if min_eig < 0:
				cov -= min_eig * np.eye(*cov.shape)

			# print(cov)
			# exit()
		
		info = {'mean': mean, 'cov':cov}
		customer_info.append(info)



	feed_forward_matrix = np.zeros((num_customers, num_facilities, num_features*2))
	for i, customer in enumerate(customers):
		for j, facility in enumerate(facilities):
			customer_features = features[customer]
			facility_features = features[facility]
			concat_features = np.concatenate([customer_features, facility_features])
			feed_forward_matrix[i,j,:] = concat_features

	feed_forward_matrix = torch.FloatTensor(feed_forward_matrix)


	demands_all = []
	travel_times_all = []
	ground_truth_distances_all = []
	customer_facility_distances_all = []
	for i in range(num_samples):

		demands = np.zeros_like(mean_demands)
		travel_times = np.zeros_like(mean_travel_times)

		for customer_index in range(num_customers):
			# Sample demand error and travel_time error for every connected facility
			error_sample = np.random.multivariate_normal(customer_info[customer_index]['mean'], customer_info[customer_index]['cov'],size=1)[0]
			# print(customer_info[customer_index]['cov'])
			# print(error_sample)
			# exit()
			demands[customer] = mean_demands[customer_index] + error_sample[0]
			travel_times[customer_index, :] = mean_travel_times[customer_index, :] + error_sample[1:]

		# demands = np.maximum(demands, 0.01)
		# travel_times = np.maximum(travel_times,0.01)


		demands_all.append(demands)
		travel_times_all.append(travel_times)

		customer_facility_distances = (travel_times.T * demands).T
		customer_facility_distances_all.append(customer_facility_distances)

	return demands_all, travel_times_all, feed_forward_matrix, customer_facility_distances_all




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

	demands, travel_times, feed_forward_matrix, customer_facility_distances = correlated_time_demand_data(num_customers=num_nodes,num_facilities=num_nodes,rho=rho, num_samples=num_samples, rho_noise=rho_noise)

	# print('Demands:%s' % demands[0])
	# print('Travel_times:%s' % travel_times[0])
	# print('Ground_truth_distances:%s' % ground_truth_distances[0])
	# exit() 

	num_customers = feed_forward_matrix.shape[0]
	num_facilities = feed_forward_matrix.shape[1]

	num_graphs = len(demands)
	test_instances = list(range(int(num_graphs*0.8), num_graphs))
	valid_instances = list(range(int(num_graphs*0.6), int(num_graphs*0.8)))
	train_instances = [x for x in range(num_graphs) if x not in test_instances and x not in valid_instances]

	nfeat = feed_forward_matrix.shape[2]

	model = GCNFacilityLocationLP(nfeat=nfeat,
	                nhid=hidden,
	                nout=embed_dim,
	                dropout=dropout,
	                K = K, 
	                nlayers = 2)

	optimizer = optim.Adam(model.parameters(), lr = lr, 
	                                   weight_decay = weight_decay)


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

	print('Two stage distances')
	print(two_stage_expected_distances)
	print('OPT distances')
	print(opt_dist)

	#exit()

	# Get two-stage assignment
	problem, parameters, variables = linear_program_avg_distance_facility_location_setup(num_customers, num_facilities,mip=True)
	parameters[0].value = two_stage_expected_distances
	parameters[1].value = K
	solution = problem.solve(solver= cp.GLPK_MI, verbose=False) # GLPK_MI
	twostage_assignments = variables[1].value

	# Get OPT
	problem, parameters, variables = linear_program_avg_distance_facility_location_setup(num_customers, num_facilities,mip=True)
	parameters[0].value = opt_dist
	parameters[1].value = K
	solution = problem.solve(solver= cp.GLPK_MI, verbose=False) # GLPK_MI
	opt_assignments = variables[1].value


	# print(customer_facility_distances[0])
	# exit()

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
		facilities, assignments, predicted_dists = model(feed_forward_matrix) #model(features, adj)
		#print(predicted_dists)
		ground_truth_demand_weighted_distance = torch.FloatTensor(customer_facility_distances[i])
		loss = torch.sum(ground_truth_demand_weighted_distance * assignments)
		loss.backward()
		losses.append(loss.item())
		batch_losses.append(loss.item())

		two_stage_loss = torch.sum(ground_truth_demand_weighted_distance * twostage_assignments)
		two_stage_losses.append(two_stage_loss.item())

		opt_loss = torch.sum(ground_truth_demand_weighted_distance * opt_assignments)
		opt_losses.append(two_stage_loss.item())


		if t % 10 == 0:
			optimizer.step()
			optimizer.zero_grad()

			print('End2End:%f,2Stage:%f,OPT:%f' % (np.mean(batch_losses),np.mean(two_stage_losses),np.mean(opt_losses)))
			#print(np.mean(batch_losses))
			batch_losses = []
			two_stage_losses = []
			opt_losses = []

		# print('predicted_dists...')
		# print(predicted_dists.detach())
		# print('true dists')
		# print(ground_truth_demand_weighted_distance.detach())
		# print('assignments...')
		# print(assignments.detach())
		# print(facilities.detach())



	#print('Training losses: %s' % (losses))

	#exit()

	# Get end-to-end assignment
	end2end_facilities, end2end_assignments, predicted_dists = model(feed_forward_matrix, mip=True)
	# print('End-to-end assignments')
	# print(predicted_dists)
	# print(end2end_facilities)
	# print(end2end_assignments)


	# print('two-stage assignments')
	# print(assignments)



	# Turn of model learning
	model.eval()
	torch.no_grad()
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

		#exit()



	print('End-to-end loss: %f. Complete losses: %s' % (np.mean(losses_test), losses_test))
	print('Two-stage loss: %f. Complete losses: %s' % (np.mean(losses_test_two_stage), losses_test_two_stage))

	return losses_test, losses_test_two_stage
# Plot rho vs two_stage-end2end gap


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








