import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import torch
import sklearn
import sklearn.cluster

class GCN(nn.Module):
    '''
    2-layer GCN with dropout
    '''
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


def linear_program_min_max_facility_location_setup(num_nodes,mip=False):

    # Set variables
    facilities = cp.Variable(shape=(num_nodes), boolean=mip)
    assignments = cp.Variable(shape=(num_nodes, num_nodes), boolean=mip)
    distance_slacks = cp.Variable(shape=(num_nodes))
    global_slack = cp.Variable()

    # Set parameters
    dists = cp.Parameter(shape=(num_nodes, num_nodes))
    K = cp.Parameter()

    #Constraint 1: Weighted sum of distances to assignments upper-bounded by slack variable
    row_sums = cp.sum(cp.atoms.affine.binary_operators.multiply(assignments, dists), axis=1) # row sum after element-wise product
    slack_constraint = row_sums <= distance_slacks

    # Constraint 2: Global slack greater than all node slack variables
    global_slack_constraint = distance_slacks <= global_slack

    # Constraint 3: Every node is assigned to precisely 1 facility
    node_facility_assignment = assignments @ np.ones(num_nodes) == 1

    # Constraint 4: Ensure each node-facility assignment upper bounded by facility (ensures facility assignments can only be made to designated facilties) 
    # Stack each facility num nodes times vertically
    tiled_facilities = []
    for i in range(num_nodes):
        facility = facilities[i]
        for j in range(num_nodes):
            if not tiled_facilities:
                tiled_facilities.append(cp.atoms.affine.vec.vec(facility))
            else:
                tiled_facilities.append(cp.atoms.affine.vec.vec(facility))

    tiled_facilities = cp.atoms.affine.hstack.hstack(tiled_facilities)


    diagonal_assignments = cp.atoms.affine.reshape.reshape(assignments, (num_nodes * num_nodes))
    facility_assignment = diagonal_assignments <= tiled_facilities.T

    # Constraint 5: At most K facilities
    facility_cap = facilities.T @ np.ones(num_nodes) <= K

    # Constraint 6: 0, 1 bounds
    facilities_upper_bound = facilities <= 1
    facilities_lower_bound = facilities >= 0
    assignments_upper_bound = diagonal_assignments <= 1
    assignments_lower_bound = diagonal_assignments >= 0

    constraints = [slack_constraint, 
                    global_slack_constraint,
                    node_facility_assignment,
                    facility_assignment, 
                    facilities_lower_bound, 
                    facilities_upper_bound, 
                    assignments_lower_bound, 
                    assignments_upper_bound,
                    facility_cap]

    objective = cp.Minimize(global_slack)
    problem = cp.Problem(objective, constraints)

    return problem, [dists, K], [facilities, assignments, distance_slacks, global_slack]


def linear_program_avg_distance_facility_location_setup(num_customers=None, num_facilities=None, mip=False):

    # Set variables
    facilities = cp.Variable(shape=(num_facilities), boolean=mip)
    assignments = cp.Variable(shape=(num_customers, num_facilities), boolean=mip)

    # Set parameters
    dists = cp.Parameter(shape=(num_customers, num_facilities))
    K = cp.Parameter()

    row_sums = cp.sum(cp.atoms.affine.binary_operators.multiply(assignments, dists), axis=1) # row sum after element-wise product
    loss = cp.sum(row_sums, axis=0)

    # Constraint 2: Every node is assigned to precisely 1 facility
    node_facility_assignment = assignments @ np.ones(num_facilities) == 1

    # Constraint 4: Ensure each node-facility assignment upper bounded by facility (ensures facility assignments can only be made to designated facilties) 
    # Stack each facility num nodes times vertically
    tiled_facilities = []
    for i in range(num_facilities):
        facility = facilities[i]
        for j in range(num_customers):
            if not tiled_facilities:
                tiled_facilities.append(cp.atoms.affine.vec.vec(facility))
            else:
                tiled_facilities.append(cp.atoms.affine.vec.vec(facility))
                # tiled_facilities = cp.atoms.affine.hstack.hstack((tiled_facilities,cp.atoms.affine.vec.vec(facility)))

    tiled_facilities = cp.atoms.affine.hstack.hstack(tiled_facilities)

    diagonal_assignments = cp.atoms.affine.reshape.reshape(assignments, (num_customers * num_facilities))
    facility_assignment = diagonal_assignments <= tiled_facilities.T

    # Constraint 5: At most K facilities
    facility_cap = facilities.T @ np.ones(num_facilities) <= K

    # Constraint 6: 0, 1 bounds
    facilities_upper_bound = facilities <= 1
    facilities_lower_bound = facilities >= 0
    assignments_upper_bound = diagonal_assignments <= 1
    assignments_lower_bound = diagonal_assignments >= 0

    constraints = [node_facility_assignment,
                    facility_assignment, 
                    facilities_lower_bound, 
                    facilities_upper_bound, 
                    assignments_lower_bound, 
                    assignments_upper_bound,
                    facility_cap]

    penalty_term_weight = 10.0
    # squared 2-norm of decision variables
    quadratic_penalty_term = cp.norm(assignments) * penalty_term_weight

    if mip:
        objective = cp.Minimize(loss)
    else:
        objective = cp.Minimize(loss + quadratic_penalty_term)
    problem = cp.Problem(objective, constraints)

    return problem, [dists, K], [facilities, assignments]

class GCNLP(nn.Module):
    '''
    The GCNLP architecture. The first step is a GCN to generate embeddings.
    The dot product of those embeddings give demand-weighted travel times predictions
    The output is the solution to LP given those parameers inputs
    
    The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
    adjacency matrix.
    '''
    def __init__(self, nfeat, nhid, nout, dropout, K, nlayers):
        super(GCNLP, self).__init__()

        self.GCN = GCNDeep(nfeat, nhid, nout, dropout, nlayers)
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
        self.K = torch.tensor(K).type(torch.float32).unsqueeze(0)
        self.init = torch.rand(K, nout)
        
    def forward(self, x, adj, mip=False):

        embeds = self.GCN(x, adj) # x are node features, adj are indices
        dists = torch.matmul(embeds, embeds.t())

        ind = np.diag_indices(dists.shape[0])
        dists[ind[0], ind[1]] = torch.zeros(dists.shape[0])

        num_nodes = x.shape[0]
        problem, parameters, variables = linear_program_kcenter_setup(num_nodes,mip=False)
        lp_layer = CvxpyLayer(problem, parameters=parameters, variables=variables)
        dists = dists.unsqueeze(0).type(torch.float32)
        facilities, assignments, distance_slacks, global_slack = lp_layer(dists, self.K)

        return facilities, assignments, embeds, distance_slacks


class NNFacilityLocationLP(nn.Module):
    '''
    The neural network facility location linear programming architecture. The first step is a NN to generate embeddings.
    The dot product of those embeddings give demand-weighted travel times predictions.
    The output is the solution to facility location LP given those parameter inputs.
    At test time, can set mip=True to do a forward pass optimally with integer constraints
    '''
    def __init__(self, num_nodes, nfeat, nhid, nout, dropout, K, nlayers):
        super(NNFacilityLocationLP, self).__init__()


        self.K = torch.tensor(K).type(torch.float32).unsqueeze(0)
        # self.ff1 = nn.Linear(nfeat,1)
        # torch.nn.init.xavier_normal(self.ff1.weight)
        # self.relu1 = torch.nn.ReLU()
        # self.ff2 = nn.Linear(nhid,1)
        # torch.nn.init.xavier_normal(self.ff2.weight)

        self.pars = torch.nn.Parameter(torch.randn(num_nodes,num_nodes) * 1e-2, requires_grad=True)
        
    def forward(self, x, mip=False):

        dists = self.pars

        num_customers = x.shape[0]
        num_facilities = x.shape[1]
        problem, parameters, variables = linear_program_avg_distance_facility_location_setup(num_customers,num_facilities,mip=mip)
        
        if mip:
            # No gradients
            parameters[0].value = dists.detach().numpy()
            parameters[1].value = int(self.K.detach().numpy())
            solution = problem.solve(solver= cp.GLPK_MI, verbose=False)
            facilities = variables[0].value
            assignments = variables[1].value
        else:
            lp_layer = CvxpyLayer(problem, parameters=parameters, variables=variables)
            dists = dists.unsqueeze(0).type(torch.float32)
            facilities, assignments = lp_layer(dists, self.K)

        return facilities, assignments, dists
    
class GCNDeep(nn.Module):
    '''
    A stack of nlayers GCNs. The first maps nfeat -> nhid features, the 
    middle layers all map nhid -> nhid, and the last maps nhid -> nout.
    '''
    def __init__(self, nfeat, nhid, nout, dropout, nlayers):
        super(GCNDeep, self).__init__()

        self.gcstart = GraphConvolution(nfeat, nhid)
        self.gcn_middle = []
        for i in range(nlayers-2):
            self.gcn_middle.append(GraphConvolution(nhid, nhid))
        self.gcend = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gcstart(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        for gc in self.gcn_middle:
            x = F.relu(gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcend(x, adj)

        return x
    

class GCNDeepSigmoid(nn.Module):
    '''
    Deep archicture that, instead of being intended to output a cluster membership
    for each node (as in GCNDeep), outputs instead a probability distribution over
    the nodes. Used for problems like facility location where the goal is to select
    a subset of K nodes. 
    '''
    def __init__(self, nfeat, nhid, nout, dropout, nlayers):
        super(GCNDeepSigmoid, self).__init__()

        self.gcstart = GraphConvolution(nfeat, nhid)
        self.gcn_middle = []
        for i in range(nlayers-2):
            self.gcn_middle.append(GraphConvolution(nhid, nhid))
        self.gcend = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gcstart(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        for gc in self.gcn_middle:
            x = F.relu(gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcend(x, adj)
        x = torch.nn.Sigmoid()(x).flatten()
        return x
