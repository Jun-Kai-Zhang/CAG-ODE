import numpy as np
import torch
from torch_geometric.data import DataLoader,Data
from torch.utils.data import DataLoader as Loader
from tqdm import tqdm
import math
from scipy.linalg import block_diag
import lib.utils as utils
import pandas as pd


weights = [10,1,10,1,100000]

class ParseData(object):

    def __init__(self,args):
        self.args = args
        self.datapath = args.datapath
        self.dataset = args.dataset
        self.random_seed = args.random_seed
        self.pred_length = args.pred_length
        self.condition_length = args.condition_length
        self.batch_size = args.batch_size

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


    def load_train_data(self,is_train = True):

        # Loading Data. N is state number, T is number of days. D is feature number.

        features = torch.load(self.args.datapath + self.args.dataset + '/features.pt').permute(1,0,2).numpy()  # [N,T,D]
        graphs = torch.load(self.args.datapath + self.args.dataset + '/graphs.pt').numpy()
        self.num_states = features.shape[0]


        # Feature normalization
        # nomalization mean
        features_mean = np.mean(features, axis=(0, 1))
        features_std = np.std(features, axis=(0, 1))

        # TODO: Maybe we can normalize the graph to [0,1]
        graphs = graphs / weights[-1]

        norm_info = {
            'features_mean': features_mean,
            'features_std': features_std,
        }

        for d in range(features.shape[2]):
            features[:, :, d] = (features[:, :, d] - features_mean[d]) / (features_std[d] + 1)


        self.num_features = features.shape[2]


        # Graph Preprocessing: remain self-loop and take log
        graphs = self.graph_preprocessing(graphs,method = 'norm_const', is_self_loop = True)  #[T,N,N]

        treatments = torch.tensor(torch.load(self.args.datapath + self.args.dataset + '/' + self.args.treatment_name + ".pt"))
        treatments = treatments.permute(1, 0, 2)  # [N, T, t_dim]

        interference = torch.load(self.args.datapath + self.args.dataset + '/interference.pt')
        interference = interference.permute(1, 0, 2)  # [N, T, t_dim]

        # Split Training Samples
        features,graphs,time_absolute_split,treatments_split, interference_split = self.generateTrainSamples(features,graphs,treatments,interference) #[K,N,T,D], [K,T,N,N], [K,N,T]

        if is_train:
            features = features[:-5, :, :, :]
            graphs = graphs[:-5, :, :, :]
            time_absolute_split = time_absolute_split[:-5,:,:]
            treatments_split = treatments_split[:-5,:,:,:]
            interference_split = interference_split[:-5,:,:,:]
        else:
            features = features[-5:, :, :, :]
            graphs = graphs[-5:, :, :, :]
            time_absolute_split = time_absolute_split[-5:,:,:]
            treatments_split = treatments_split[-5:, :, :,:]
            interference_split = interference_split[-5:, :, :,:]



        encoder_data_loader, decoder_data_loader, decoder_graph_loader, num_batch, self.num_states = self.generate_train_val_dataloader(features,graphs,time_absolute_split,treatments_split,interference_split,is_train)


        return encoder_data_loader, decoder_data_loader, decoder_graph_loader, num_batch, self.num_states, norm_info


    def generate_train_val_dataloader(self,features,graphs,time_absolute_split,treatment_split,interference_split,is_train = True):
        # Split data for encoder and decoder dataloader
        '''

        :param features:
        :param graphs:
        :param is_train:
        time_absolute_split: [K,N,T]
        :return:
        '''
        feature_observed, times_observed, series_decoder_cat, times_extrap, treatment_series_decoder, interference_series_decoder = self.split_data(
            features,time_absolute_split,treatment_split,interference_split)  # series_decoder[K*N,T2,D] # time_absolute_split_extrap [K*N,T_extrap]
        self.times_extrap = times_extrap

        # Generate Encoder data
        encoder_data_loader = self.transfer_data(feature_observed, graphs, times_observed, self.batch_size)



        series_decoder_all = [(series_decoder_cat[i, :, :], treatment_series_decoder[i,:,:],interference_series_decoder[i,:,:]) for i in
                              range(series_decoder_cat.shape[0])]

        decoder_data_loader = Loader(series_decoder_all, batch_size=self.batch_size * self.num_states, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                         batch))  # num_graph*num_ball [tt,vals,masks]

        graph_decoder = graphs[:, self.args.condition_length:, :, :]  # [K,T2,N,N]
        decoder_graph_loader = Loader(graph_decoder, batch_size=self.batch_size, shuffle=False)



        num_batch = len(decoder_data_loader)
        assert len(decoder_data_loader) == len(decoder_graph_loader)

        # Inf-Generator
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        decoder_graph_loader = utils.inf_generator(decoder_graph_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return encoder_data_loader, decoder_data_loader, decoder_graph_loader, num_batch, self.num_states


    def graph_preprocessing(self,graph_input, method = 'norm_const', is_self_loop = True):
        '''
                :param graph_input: [T,N,N]
                :param method: norm--norm by rows: G[i,j] is the outflow from i to j.
                :param is_self_loop: True to remain self-loop, otherwise no self-loop
                :return: [T,N,N]
                '''
        if not is_self_loop:
            num_days = graph_input.shape[0]
            num_states = graph_input.shape[1]
            graph_output = np.ones_like(graph_input)
            for i in range(num_days):
                graph_output[i] = graph_input[i] * (1 - np.identity(num_states))
        else:
            graph_output = graph_input

        if method == "log":
            graph_output = np.log(graph_output + 1)  # 0 remains 0
        elif method == "norm_const":
            graph_output = graph_output/weights[-1]

        return graph_output

    def generateTrainSamples(self,features, graphs,treatments,interference):
        '''
        Split training data into several overlapping series.
        :param features: [N,T,D]
        :param graphs: [T,N,N]
        :param interval: 3
        :return: transform feature into [K,N,T,D], transform graph into [K,T,N,N]
        '''
        interval = self.args.split_interval
        each_length = self.args.pred_length + self.args.condition_length
        num_batch = math.floor((features.shape[1] - each_length) / interval) + 1
        num_states = features.shape[0]
        num_features = features.shape[2]
        features_split = np.zeros((num_batch, num_states, each_length, num_features))
        time_absolute_split = np.zeros((num_batch, num_states, each_length))  #[K,N,T]
        graphs_split = np.zeros((num_batch, each_length, num_states, num_states))
        batch_num = 0

        num_treatments = treatments.shape[2]
        num_interference = interference.shape[2]
        treatments_split = np.zeros((num_batch, num_states, each_length, num_treatments))
        interference_split = np.zeros((num_batch, num_states, each_length, num_interference))

        for i in range(0, features.shape[1] - each_length+1, interval):
            assert i + each_length <= features.shape[1]
            features_split[batch_num] = features[:, i:i + each_length, :]
            treatments_split[batch_num] = treatments[:, i:i + each_length, :]
            interference_split[batch_num] = interference[:, i:i + each_length, :]
            time_absolute_split[batch_num,:,:] = np.arange(i,i + each_length)
            graphs_split[batch_num] = graphs[i:i + each_length, :, :]
            batch_num += 1



        return features_split, graphs_split, time_absolute_split,treatments_split, interference_split  # [K,N,T,D], [K,T,N,N]

    def split_data(self, feature,time_absolute_split,treatment_split,interference_split):
        '''
               Generate encoder data (need further preprocess) and decoder data
               :param feature: [K,N,T,D], T=T1+T2
               :param time_absolute_split: [K,N,T]
               :return:
               '''

        feature_observed = feature[:, :, :self.args.condition_length, :]
        # select corresponding features
        feature_out_index = self.args.feature_out_index
        feature_extrap = feature[:, :, self.args.condition_length:, feature_out_index]
        assert feature_extrap.shape[-1] == len(feature_out_index)
        times = np.asarray([i / feature.shape[2] for i in range(feature.shape[2])])  # normalized in [0,1] T
        times_observed = times[:self.args.condition_length]  # [T1]
        times_extrap = times[self.args.condition_length:] - times[
            self.args.condition_length]  # [T2] making starting time of T2 be 0.
        assert times_extrap[0] == 0

        time_absolute_split_extrap = time_absolute_split[:,:,self.args.condition_length:]  #[K,N,T_extrap]
        time_absolute_split_extrap = np.reshape(time_absolute_split_extrap,(-1,len(times_extrap),1)) #[K*N,T_extrap]
        series_decoder = np.reshape(feature_extrap, (-1, len(times_extrap), len(feature_out_index)))  # [K*N,T2,D]

        series_decoder_cat = np.concatenate([series_decoder,time_absolute_split_extrap],axis=-1) # [K*N,T2,D]

        treatment_extrap = treatment_split[:, :, self.args.condition_length:, :]
        interference_extrap = interference_split[:, :, self.args.condition_length:, :]
        treatment_series_decoder = np.reshape(treatment_extrap,(-1, len(times_extrap), treatment_extrap.shape[-1]))  # [K*N,T2,D]
        interference_series_decoder = np.reshape(interference_extrap, (
        -1, len(times_extrap), interference_extrap.shape[-1]))  # [K*N,T2,D]

        return feature_observed, times_observed, series_decoder_cat, times_extrap, treatment_series_decoder, interference_series_decoder

    def transfer_data(self, feature, edges, times,batch_size):
        '''

        :param feature: #[K,N,T1,D]
        :param edges: #[K,T,N,N], with self-loop
        :param times: #[T1]
        :param time_begin: 1
        :return:
        '''
        data_list = []
        edge_size_list = []

        num_samples = feature.shape[0]

        for i in tqdm(range(num_samples)):
            data_per_graph, edge_size = self.transfer_one_graph(feature[i], edges[i], times)
            data_list.append(data_per_graph)
            edge_size_list.append(edge_size)

        print("average number of edges per graph is %.4f" % np.mean(np.asarray(edge_size_list)))
        data_loader = DataLoader(data_list, batch_size=batch_size,shuffle=False)

        return data_loader



    def feature_normalization(self,features, feature_ID, method='None', is_inc=False):
        '''
        normalize one single feature.
        :param features: [N,T,D]
        :param feature_ID:
        :param method:
        :param is_inc:
        :return: [N,T]
        '''
        # method: log, norm, log_norm, None, norm is to normalize within [1,2]
        if is_inc:
            one_feature = np.ones_like(features[:, :, feature_ID])  # [N,T]
            one_feature[:, 1:] = features[:, 1:, feature_ID] - features[:, :-1, feature_ID]
        else:
            one_feature = features[:, :, feature_ID]

        if method == "log":
            one_feature = np.log(one_feature + 1)
            return one_feature
        elif method == 'norm_const':
            one_feature = one_feature/weights[feature_ID]
            return one_feature
        elif method == "None":
            return one_feature

    def transfer_one_graph(self,feature, edge, time):
        '''f

        :param feature: [N,T1,D]
        :param edge: [T,N,N]  (needs to transfer into [T1,N,N] first, already with self-loop)
        :param time: [T1]
        :return:
            1. x : [N*T1,D]: feature for each node.
            2. edge_index [2,num_edge]: edges including cross-time
            3. edge_weight [num_edge]: edge weights
            4. y: [N], value= num_steps: number of timestamps for each state node.
            5. x_pos 【N*T1】: timestamp for each node
            6. edge_time [num_edge]: edge relative time.
        '''

        ########## Getting and setting hyperparameters:
        num_states = feature.shape[0]
        T1 = self.args.condition_length
        each_gap = 1/ edge.shape[0]
        edge = edge[:T1,:,:]
        time = np.reshape(time,(-1,1))

        ########## Compute Node related data:  x,y,x_pos
        # [Num_states],value is the number of timestamp for each state in the encoder, == args.condition_length
        y = self.args.condition_length*np.ones(num_states)
        # [Num_states*T1,D]
        x = np.reshape(feature,(-1,feature.shape[2]))
        # [Num_states*T1,1] node timestamp
        x_pos = np.concatenate([time for i in range(num_states)],axis=0)
        assert len(x_pos) == feature.shape[0]*feature.shape[1]

        ########## Compute edge related data
        edge_time_matrix = np.concatenate([np.asarray(x_pos).reshape(-1, 1) for _ in range(len(x_pos))],
                                          axis=1) - np.concatenate(
            [np.asarray(x_pos).reshape(1, -1) for _ in range(len(x_pos))], axis=0)  # [N*T1,N*T1], SAME TIME = 0

        edge_exist_matrix = np.ones((len(x_pos), len(x_pos)))  # [N*T1,N*T1] NO-EDGE = 0, depends on both edge weight and time matrix

        # Step1: Construct edge_weight_matrix [N*T1,N*T1]
        edge_repeat = np.repeat(edge, self.args.condition_length, axis=2)  # [T1,N,NT1]
        edge_repeat = np.transpose(edge_repeat, (1, 0, 2))  # [N,T1,NT1]
        edge_weight_matrix = np.reshape(edge_repeat, (-1, edge_repeat.shape[2]))  # [N*T1,N*T1]

        # mask out cross_time edges of different state nodes.
        a = np.identity(T1)  # [T,T]
        b = np.concatenate([a for i in range(num_states)], axis=0)  # [N*T,T]
        c = np.concatenate([b for i in range(num_states)], axis=1)  # [N*T,N*T]

        a = np.ones((T1, T1))
        d = block_diag(*([a] * num_states))
        edge_weight_mask = (1 - d) * c + d
        edge_weight_matrix = edge_weight_matrix * edge_weight_mask  # [N*T1,N*T1]

        max_gap = each_gap


        # Step2: Construct edge_exist_matrix [N*T1,N*T1]: depending on both time and weight.
        edge_exist_matrix = np.where(
            (edge_time_matrix <= 0) & (abs(edge_time_matrix) <= max_gap) & (edge_weight_matrix != 0),
            edge_exist_matrix, 0)


        edge_weight_matrix = edge_weight_matrix * edge_exist_matrix
        edge_index, edge_weight_attr = utils.convert_sparse(edge_weight_matrix)
        assert np.sum(edge_weight_matrix!=0)!=0  #at least one edge weight (one edge) exists.

        edge_time_matrix = (edge_time_matrix + 3) * edge_exist_matrix # padding 2 to avoid equal time been seen as not exists.
        _, edge_time_attr = utils.convert_sparse(edge_time_matrix)
        edge_time_attr -= 3

        # converting to tensor
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_weight_attr = torch.FloatTensor(edge_weight_attr)
        edge_time_attr = torch.FloatTensor(edge_time_attr)
        y = torch.LongTensor(y)
        x_pos = torch.FloatTensor(x_pos)


        graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight_attr, y=y, pos=x_pos, edge_time = edge_time_attr)
        edge_num = edge_index.shape[1]

        return graph_data,edge_num

    def variable_time_collate_fn_activity(self,batch):
        """
        Expects a batch of
            - (feature0,feaure_gt) [K*N, T2, D]
        """
        # Extract corrsponding deaths or cases
        combined_vals = np.concatenate([np.expand_dims(ex[0],axis=0) for ex in batch],axis=0)
        combined_treatment = np.concatenate([np.expand_dims(ex[1], axis=0) for ex in batch], axis=0)
        combined_interference = np.concatenate([np.expand_dims(ex[2], axis=0) for ex in batch], axis=0)




        combined_vals = torch.FloatTensor(combined_vals) #[M,T2,D]
        combined_vals_treatment = torch.FloatTensor(combined_treatment)  # [M,T2,D]
        combined_vals_interference = torch.FloatTensor(combined_interference)  # [M,T2,D]


        combined_tt = torch.FloatTensor(self.times_extrap)

        data_dict = {
            "data": combined_vals[:,:,:-1],
            "time_steps": combined_tt,
            "treatment": combined_vals_treatment,
            "interference": combined_vals_interference,
            "time_absolute": combined_vals[:,:,-1] #[K*N,T]
            }
        return data_dict



