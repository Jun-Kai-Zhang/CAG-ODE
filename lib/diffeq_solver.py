import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import lib.utils as utils
import torch.nn.functional as F
from scipy.linalg import block_diag
from torch_scatter import scatter_add
import scipy.sparse as sp
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.inits import glorot


class TemporalEncoding(nn.Module):

    def __init__(self, d_hid):
        super(TemporalEncoding, self).__init__()
        self.d_hid = d_hid
        self.div_term = torch.FloatTensor([1 / np.power(10000, 2 * (hid_j // 2) / self.d_hid) for hid_j in range(self.d_hid)]) #[20]
        self.div_term = torch.reshape(self.div_term,(1,-1))
        self.div_term = nn.Parameter(self.div_term,requires_grad=False)

    def forward(self, t):
        '''

        :param t: [n,1]
        :return:
        '''
        t = t.view(-1,1)
        t = t
        position_term = torch.matmul(t,self.div_term)
        position_term[:,0::2] = torch.sin(position_term[:,0::2])
        position_term[:,1::2] = torch.cos(position_term[:,1::2])

        return position_term

def compute_edge_initials(first_point_enc, num_atoms,w_node_to_edge_initial):
    '''

    :param first_point_enc: [K*N,D]
    :return: [K*N*N,D']
    '''
    node_feature_num = first_point_enc.shape[1]
    fully_connected = np.ones([num_atoms, num_atoms])
    rel_send = np.array(utils.encode_onehot(np.where(fully_connected)[0]),
                        dtype=np.float32)  # every node as one-hot[10000], (N*N,N)
    rel_rec = np.array(utils.encode_onehot(np.where(fully_connected)[1]),
                       dtype=np.float32)  # every node as one-hot[10000], (N*N,N)

    rel_send = torch.FloatTensor(rel_send).to(first_point_enc.device)
    rel_rec = torch.FloatTensor(rel_rec).to(first_point_enc.device)

    first_point_enc = first_point_enc.view(-1, num_atoms, node_feature_num)  # [K,N,D]

    senders = torch.matmul(rel_send, first_point_enc)  # [K,N*N,D]
    receivers = torch.matmul(rel_rec, first_point_enc)  # [K,N*N,D]

    edge_initials = torch.cat([senders, receivers], dim=-1)  # [K,N*N,2D]
    edge_initials = F.gelu(w_node_to_edge_initial(edge_initials))  # [K,N*N,D_edge]
    edge_initials = edge_initials.view(-1, edge_initials.shape[2])  # [K*N*N,D_edge]

    return edge_initials



class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method,args,
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.device = device
        self.ode_func = ode_func
        self.args = args
        self.num_atoms = args.num_atoms

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol



    def forward(self, first_point,time_steps_to_predict,times_absolute,w_node_to_edge_initial):
        '''

        :param first_point:  [K*N,D]
        :param edge_initials: [K*N*N,D]
        :param time_steps_to_predict: [t] 1....14/14
        times_absolute [K*N,T]
        :return:
        '''

        # Node ODE Function
        n_traj,feature_node = first_point.size()[0], first_point.size()[1]  #[K*N,d]
        if self.args.augment_dim > 0:
            aug_node = torch.zeros(first_point.shape[0], self.args.augment_dim).to(self.device) #[K*N,D_aug]
            first_point = torch.cat([first_point, aug_node], 1) #[K*N,d+D_aug]
            feature_node += self.args.augment_dim

        # Edge initialization: h_ij = f([u_i,u_j])
        edge_initials = compute_edge_initials(first_point, self.num_atoms, w_node_to_edge_initial)  # [K*N*N,D_edge]
        assert (not torch.isnan(edge_initials).any())

        node_edge_initial = torch.cat([first_point,edge_initials],0)  #[K*N + K*N*N,D+D_aug]
        # Set index
        K_N = int(node_edge_initial.shape[0]/(self.num_atoms+1))
        K = K_N/self.num_atoms
        self.ode_func.set_index_and_graph(K_N,K,time_steps_to_predict)
        #Set Treatment
        self.ode_func.set_t_treatments(times_absolute)

        node_initial = node_edge_initial[:K_N,:]
        self.ode_func.set_initial_z0(node_initial)


        # Results
        self.ode_func.nfe = 0
        self.ode_func.t_index = 0
        pred_y = odeint(self.ode_func, node_edge_initial, time_steps_to_predict,
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method) #[time_length, K*N + K*N*N, D]

        pred_y = pred_y.permute(1,0,2) #[ K*N + K*N*N, time_length, d]

        if int(self.args.use_attention)==1 and int(self.args.use_onehot)==0:
            treatment_dim = self.args.treatment_dim
        else:
            treatment_dim = self.args.t_dim

        treatment_rep = torch.zeros([time_steps_to_predict.shape[0], K_N, treatment_dim]).to(self.device)

        if int(self.args.use_attention) == 1:
            for t in np.arange(len(time_steps_to_predict)):
                treatment_rep[t, :, :] = self.ode_func.calculate_merged_treatment(
                    self.ode_func.t_treatments[:, t, :], t, self.ode_func.w_treatment)
        else:
            for t in np.arange(len(time_steps_to_predict)):
                treatment_rep[t, :, :] = self.ode_func.t_treatments[:, t, :]



        treatment_rep = treatment_rep.permute(1, 0, 2)  # [ K*N, time_length, treatmen_dim]


        assert(pred_y.size()[0] == K_N*(self.num_atoms+1))

        if self.args.augment_dim > 0:
            pred_y = pred_y[:, :, :-self.args.augment_dim]

        return pred_y,K_N,treatment_rep

    def set_treatments(self, treatments):
        self.ode_func.set_treatments(treatments)

    def set_policy_starting_ending_points(self, policy_starting_ending_points):
        self.ode_func.set_policy_starting_ending_points(policy_starting_ending_points)



class CoupledODEFunc(nn.Module):
    def __init__(self, args,num_treatment,node_ode_func_net,edge_ode_func_net,num_atom, dropout,device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(CoupledODEFunc, self).__init__()


        self.args = args
        self.num_treatment = num_treatment
        self.device = device
        self.node_ode_func_net = node_ode_func_net  #input: x, edge_index
        self.edge_ode_func_net = edge_ode_func_net
        self.num_atom = num_atom
        self.nfe = 0
        self.t_index = 0
        self.dropout = nn.Dropout(dropout)

        self.w_treatment = nn.Parameter(
            torch.FloatTensor(self.num_treatment,
                              self.args.treatment_dim))  # TODO: changeg the dimnesion to arbitrary d.
        glorot(self.w_treatment)
        self.w_treatment_attention = nn.Linear(self.args.treatment_dim,
                                               self.args.treatment_dim)  # for calculating attention vector
        utils.init_network_weights(self.w_treatment_attention)
        self.temporal_net = TemporalEncoding(self.args.treatment_dim)

    def get_time_index(self,t_local):

        tmp = torch.nonzero(torch.where(t_local>=self.time_steps_to_predict,1,0))[-1]
        return tmp




    def forward(self, t_local, z, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        z:  [H,E] concat by axis0. H is [K*N,D], E is[K*N*N,D], z is [K*N + K*N*N, D]
        """
        self.nfe += 1

        t_index = self.get_time_index(t_local)




        node_attributes = z[:self.K_N,:]
        edge_attributes = z[self.K_N:,:]
        assert (not torch.isnan(node_attributes).any())
        assert (not torch.isnan(edge_attributes).any())

        # Calculate the aggregaated treament representation
        if int(self.args.use_attention) == 1:  # with attention
            treatment_curret_t_merged = self.calculate_merged_treatment(self.t_treatments[:, t_index, :],
                                                                        t_index,
                                                                        self.w_treatment)
            cat_node_attributes = torch.cat((node_attributes, treatment_curret_t_merged), dim=1)

        else:
            cat_node_attributes = torch.cat((node_attributes, self.t_treatments[:, t_index, :].squeeze()), dim=1)

        #grad_edge, edge_value = self.edge_ode_func_net(node_attributes,self.num_atom) # [K*N*N,D],[K,N*N], edge value are non-negative by using relu.
        grad_edge, edge_value = self.edge_ode_func_net(cat_node_attributes, edge_attributes,
                                                       self.num_atom)  # [K*N*N,D],[K,N*N], edge value are non-negative by using relu.todo:with self-evolution

        edge_value = self.normalize_graph(edge_value, self.K_N)
        assert (not torch.isnan(edge_value).any())
        grad_node = self.node_ode_func_net(cat_node_attributes, edge_value, self.node_z0)  # [K*N,D]
        assert (not torch.isnan(grad_node).any())
        assert (not torch.isinf(grad_edge).any())

        assert (not torch.isnan(grad_node).any())
        assert (not torch.isinf(grad_edge).any())

        # Concat two grad
        grad = self.dropout(torch.cat([grad_node, grad_edge], 0))  # [K*N + K*N*N, D]



        return grad


    def set_index_and_graph(self,K_N,K,time_steps_to_predict):
        '''

        :param K_N: index for separating node and edge matrixs.
        :return:
        '''
        self.K_N = K_N
        self.K = int(K)
        self.K_N_N = self.K_N*self.num_atom
        self.nfe = 0
        self.time_steps_to_predict = time_steps_to_predict

        # Step1: Concat into big graph: which is set by default.diagonal matrix
        edge_each = np.ones((self.num_atom, self.num_atom))
        edge_whole= block_diag(*([edge_each] * self.K))
        edge_index,_ = utils.convert_sparse(edge_whole)
        self.edge_index = torch.LongTensor(edge_index).to(self.device)

    def set_initial_z0(self,node_z0):
        self.node_z0 = node_z0

    def normalize_graph(self,edge_weight,num_nodes):
      '''
      For asymmetric graph
      :param edge_index: [num_edges,2], torch.LongTensor
      :param edge_weight: [K,N*N] ->[num_edges]
      :param num_nodes:
      :return:
      '''
      assert (not torch.isnan(edge_weight).any())
      assert (torch.sum(edge_weight<0)==0)
      edge_weight_flatten = edge_weight.view(-1)  #[K*N*N]

      row, col = self.edge_index[0], self.edge_index[1]
      deg = scatter_add(edge_weight_flatten, row, dim=0, dim_size=num_nodes) #[K*N]
      deg_inv_sqrt = deg.pow_(-1)
      deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
      if torch.isnan(deg_inv_sqrt).any():
          assert (torch.sum(deg == 0) == 0)
          assert (torch.sum(deg < 0) == 0)

      #assert (not torch.isnan(deg_inv_sqrt).any())


      edge_weight_normalized = deg_inv_sqrt[row] * edge_weight_flatten   #[k*N*N]
      assert (torch.sum(edge_weight_normalized < 0) == 0) and (torch.sum(edge_weight_normalized > 1) == 0)

      # Reshape back

      edge_weight_normalized = torch.reshape(edge_weight_normalized,(self.K,-1)) #[K,N*N]
      assert (not torch.isnan(edge_weight_normalized).any())

      return edge_weight_normalized

    def set_t_treatments(self, time_steps_to_predict):
        '''

        :param time_steps_to_predict: [K*N,T]
        :return:
        '''
        device = time_steps_to_predict.device
        T2 = time_steps_to_predict.shape[1]
        t_dim = self.treatments.shape[2]
        t_treatments = torch.zeros((self.K_N, T2, t_dim)).to(device)

        for k in range(self.K):
            for i in range(self.num_atom):
                    t_treatments[k * self.num_atom + i, :, :] = self.treatments[i,
                                                            time_steps_to_predict[k * self.num_atom + i].long(), :]
        self.t_treatments = t_treatments
        self.time_absolute = time_steps_to_predict

    def set_treatments(self, treatments):
        """
        treatments [N, T, t_dim]
        """
        self.treatments = treatments

    def set_policy_starting_ending_points(self, policy_starting_ending_points):
        """
        policy_starting_ending_points [t_dim, 2], containing the starting point and ending point of each policy
        """
        self.policy_starting_ending_points = policy_starting_ending_points

    def weather_multiple_treatment(self, t_all_index):
        '''
        t_allindex: [num_treatment,2] (0:k*n node index, also group index, 1: treatment_id)]
        output: [num_treatment], 1 indicating the treatment has corrresponding other to form a group, 0 means at t only one treatment.
        '''
        _, inverse_indices, counts = torch.unique(t_all_index[:, 0], return_inverse=True, return_counts=True)
        counts = counts - 1
        counts_indexes_nonzero = torch.nonzero(counts)
        returned_bool = torch.where(torch.isin(inverse_indices, counts_indexes_nonzero), 1, 0).to(t_all_index.device)

        return returned_bool

    def one_hot_encode(self, input_matrix, num_classes):
        """
        Convert a matrix into one-hot representations.

        :param matrix: numpy array of size [k,1] with integer values from 1 to K
        :param num_classes: number of classes (K)
        :return: numpy array of size [k,K] with one-hot representations
        """
        one_hot = torch.zeros((input_matrix.shape[0], num_classes)).to(input_matrix.device)
        one_hot[
            torch.arange(input_matrix.shape[0]).to(input_matrix.device), input_matrix.flatten()] = torch.FloatTensor(
            [1]).to(input_matrix.device)
        return one_hot

    def calculate_merged_treatment(self, t_treatment, t_current, w_treatment):
        '''
        t_treatment:[k*n,t_dim]
        t_current: scaler, should be a non-normalized one. TODO: revise if normaized it into [0,1]
        '''
        t_all_index = torch.nonzero(t_treatment).to(
            t_treatment.device)  # [num_treatment, 2(0:k*n node index, also group index, 1: treatment_id)]

        if int(self.args.use_onehot) == 1:
            t_treatment_embeddings_original = self.one_hot_encode(t_all_index[:, 1].view(-1, 1),
                                                                  self.num_treatment)  # [num_treatment,t_dim]

        t_treatment_embeddings = torch.index_select(w_treatment, 0, t_all_index[:, 1])  # [num_treatment,d]
        # t_treatment_embeddings.requires_grad_(True)

        # Add positional encoding here.
        # Step1: get the state id for each treatment.
        num_k_list = t_all_index[:, 0] // self.num_atom
        num_n_list = (t_all_index[:,
                      0] - num_k_list * self.num_atom)  # [num_treatment], stored the state id for each treatment
        t_start_treatment_list = torch.index_select(self.policy_starting_ending_points, 0, num_n_list)[:, :,
                                 0]  # [num_treatment,t_dim]
        # Step2: get the start time for the real policy
        t_start_treatment_list = t_start_treatment_list[
            torch.arange(t_treatment_embeddings.shape[0]).to(t_treatment.device), t_all_index[:,
                                                                                  1].squeeze()]  # [num_treatament], the time is absolute time


        delta_t_treament = self.time_absolute[t_all_index[:,0],t_current] - t_start_treatment_list  # [num_treatment]

        # TODO: if there's only one treatment, we get rid of the attention. therefor also no positional encoding
        if int(self.args.mask_single_treatment) == 1:
            treatment_group_bool = self.weather_multiple_treatment(t_all_index)  # [num_treatment]
        else:
            treatment_group_bool = torch.ones_like((t_all_index[:, 0]))

        t_treatment_embeddings += self.temporal_net(
            delta_t_treament.to(torch.float).to(t_treatment.device)) * treatment_group_bool.view(-1, 1)

        # Calculate attention vector per group

        attention_vector_group = F.gelu(self.w_treatment_attention(
            global_mean_pool(t_treatment_embeddings, t_all_index[:, 0])))  # [num_treatment_group,d]
        # Expand attention vector to [num_treatment, d]
        attention_vector_group_expanded = torch.index_select(attention_vector_group, 0,
                                                             t_all_index[:, 0])  # [num_treatment, d]
        # Calculate attention score:
        attention_treatments = torch.sigmoid(
            torch.squeeze(torch.bmm(torch.unsqueeze(attention_vector_group_expanded, 1),
                                    torch.unsqueeze(t_treatment_embeddings, 2)))).view(-1, 1)  # [num_treatments]

        if int(self.args.use_onehot) == 0:
            t_treatment_att_embeddings = torch.where(treatment_group_bool.view(-1, 1) == 1, attention_treatments,
                                                     torch.FloatTensor([1]).to(
                                                         attention_treatments.device)) * t_treatment_embeddings  # [num_treatment,d]
        else:
            try:
                assert torch.where(treatment_group_bool.view(-1, 1) == 1, attention_treatments,
                                   torch.FloatTensor([1]).to(attention_treatments.device)).shape[0] == \
                       t_treatment_embeddings_original.shape[0]
            except AssertionError as error:
                print(torch.where(treatment_group_bool.view(-1, 1) == 1, attention_treatments,
                                  torch.FloatTensor([1]).to(attention_treatments.device)).shape)
                print(t_treatment_embeddings_original.shape)
            t_treatment_att_embeddings = torch.where(treatment_group_bool.view(-1, 1) == 1, attention_treatments,
                                                     torch.FloatTensor([1]).to(
                                                         attention_treatments.device)) * t_treatment_embeddings_original  # [num_treatment,d]

        t_treatment_embeddings_merged = global_mean_pool(t_treatment_att_embeddings, t_all_index[:,
                                                                                     0])  # [num_treatment_group,d] without activation

        # convert back to [k*n,d]
        t_treatment_back = torch.zeros((t_treatment.shape[0], t_treatment_embeddings_merged.shape[1])).to(
            t_treatment.device)  # [k*n,d]
        kn_indexes = torch.unique(
            t_all_index[:, 0])  # [num_treament_group], value indicates the state id for each treatment
        t_treatment_back[kn_indexes] = torch.index_select(t_treatment_embeddings_merged, 0, kn_indexes)
        # t_treatment_back.requires_grad_(True)

        return t_treatment_back  # [k*n,d]

    def set_time_base(self, input_t_base):
        self.input_t_base = input_t_base.long()














