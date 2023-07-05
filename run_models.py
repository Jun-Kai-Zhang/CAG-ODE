import os
import sys
from lib.load_data_covid import ParseData
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import lib.utils as utils
from torch.distributions.normal import Normal
from lib.create_coupled_ode_model import create_CoupledODE_model
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser('Coupled ODE')

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('--dataset', type=str, default='Dec', help="Dec")
parser.add_argument('--datapath', type=str, default='data/', help="default data path")
parser.add_argument('--pred_length', type=int, default=14, help="Number of days to predict ")
parser.add_argument('--condition_length', type=int, default=7, help="Number days to condition on")
parser.add_argument('--features', type=str,
                    default="Confirmed,Deaths,Recovered,Mortality_Rate,Testing_Rate",
                    help="selected features")
parser.add_argument('--split_interval', type=int, default=3,
                    help="number of days between two adjacent starting date of two series.")
parser.add_argument('--feature_out', type=str, default='Confirmed',
                    help="Confirmed, Deaths, or Confirmed and deaths")
parser.add_argument('--treatment_dim', type=int, default=5, help="treatment embedding dim ")
parser.add_argument('--treatment_name', type=str, default="treatments", help="treatment embedding dim ")

parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-3, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=8)
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--edge_lamda', type=float, default=0.5, help='edge weight')

parser.add_argument('--z0-encoder', type=str, default='GTrans', help="GTrans")
parser.add_argument('--rec-dims', type=int, default= 64, help="Dimensionality of the recognition model .")
parser.add_argument('--ode-dims', type=int, default= 20, help="Dimensionality of the ODE func for edge and node (must be the same)")
parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in recognition model ")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers  ODE func ")

parser.add_argument('--augment_dim', type=int, default=0, help='augmented dimension')
parser.add_argument('--solver', type=str, default="rk4", help='dopri5,rk4,euler')

parser.add_argument('--alias', type=str, default="run_trail")

# dataset
parser.add_argument('--type_y', type=str, default='confirmed', choices=['confirmed', 'death'])
parser.add_argument('--type_net', type=str, default='dist', choices=['dist', 'mob', 'no'])
parser.add_argument('--start_time', type=str, default='2020-01-22')  # start time: XXXX-XX-XX
parser.add_argument('--end_time', type=str, default='2020-12-31')  # start time: XXXX-XX-XX
parser.add_argument('--time_interval', type=int, default=1, help='interval between time steps (days)')
parser.add_argument('--cuda', type=int, default=1, help='cuda training')



# balancing
parser.add_argument('--alpha', type=float, default= 0, help='weight of treatment prediction loss.')
parser.add_argument('--beta', type=float, default= 0, help='weight of inference prediction loss.')
parser.add_argument('--use_attention', type=int, default=1, help="weather to use attention")
parser.add_argument('--use_onehot', type=int, default=1, help="weather to use onehot representaton of treatment")
parser.add_argument('--mask_single_treatment', type=int, default=0, help="weather cancel the aattention if there is only one treatment")

# model type
parser.add_argument('--encoder_add_treat', type=float, default= 0, help='whether we include treatment for encoding process')
parser.add_argument('--ode_add_treat', type=float, default= 1, help='whether we include treatment for ode process')

parser.add_argument('--logdir', type=str, default='log_zijie/run_trail', help='log directory')
parser.add_argument('--device', type=str, default="cuda:0", help='choose cuda number')
args = parser.parse_args()



data_path = {
        'path_y': 'dataset/confirm_death_matrix_new.csv',
        'path_t': 'dataset/state_policies.csv',
        'path_trend': 'dataset/GoogleTrend/',
        'path_jhu': 'dataset/time_series_covid19_confirmed_US.csv',
        'path_dist': 'dataset/dis_ad_matrix_v3.csv',
        'path_mob': 'dataset/Population_mobility/'}
    # ==========



############ CPU AND GPU related
if torch.cuda.is_available():
	print("Using GPU" + "-"*80)
	device = torch.device("cuda:0")
else:
	print("Using CPU" + "-" * 80)
	device = torch.device("cpu")

###########  feature related:
if args.feature_out == "Confirmed":
    args.output_dim = 1
    args.is_cases = 1
    args.feature_out_index = [0]
    print("CONFIRMED!")
elif args.feature_out == "Deaths":
    args.output_dim = 1
    args.feature_out_index = [1]
    args.is_cases = 0
    print("Deaths!")
else:
    args.output_dim = 2
    args.feature_out_index = [0, 1]
    print("Joint!")


#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    writer = SummaryWriter(log_dir=args.logdir, flush_secs=20)
    np.random.seed(args.random_seed)

    #Saving Path
    utils.makedirs(args.save)
    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)


    #Command Log
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)


    #Loading Data
    print("predicting data at: %s" % args.dataset)
    dataloader = ParseData(args =args)
    train_encoder, train_decoder, train_graph, train_batch, num_atoms,norm_info = dataloader.load_train_data(is_train=True)
    norm_info['features_std'] =norm_info['features_std'][args.feature_out_index]
    val_encoder, val_decoder, val_graph, val_batch, _,_ = dataloader.load_train_data(is_train=False)
    args.num_atoms = num_atoms
    input_dim = dataloader.num_features

    #Loading Treatment
    treatments = torch.tensor(torch.load(args.datapath + args.dataset + '/' + args.treatment_name + ".pt"))
    treatments = treatments.to(device)
    treatments = treatments.permute(1, 0, 2)  # [N, T, t_dim]
    args.t_dim = treatments.shape[2]  # dimension of treatment

    # Model Setup
    # Create the model
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    model = create_CoupledODE_model(args, input_dim, z0_prior, obsrv_std, device)

    # Load checkpoint and evaluate the model
    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        utils.get_ckpt_model(ckpt_path, model, device)



    # Training Setup
    log_path = "logs/" + args.alias +"_" + args.dataset +  "_Con_"  + str(args.condition_length) +  "_Pre_" + str(args.pred_length) + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    logger.info(args.alias)

    # Optimizer
    if args.optimizer == "AdamW":
        optimizer =optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)


    wait_until_kl_inc = 10
    best_test_RMSE = np.inf
    n_iters_to_viz = 1

    # Set treatments
    model.set_treatments(treatments)

    policy_starting_ending_points = torch.tensor(
        torch.load(args.datapath + args.dataset + '/policy_starting_points.pt')).to(device)  # [N, t_dim,2]
    model.set_policy_starting_ending_points(policy_starting_ending_points)


    def train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef, itr):

        optimizer.zero_grad()
        train_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,args.num_atoms,edge_lamda = args.edge_lamda, alpha = args.alpha, beta = args.beta, kl_coef=kl_coef,istest=False)

        loss = train_res["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        loss_value = loss.data.item()

        del loss
        torch.cuda.empty_cache()
        # train_res, loss
        return loss_value,train_res['MSE'],train_res["likelihood_node"],train_res["likelihood_edge"],train_res["treatment_balancing"],train_res["interference_balancing"]




    def train_epoch(epo):
        model.train()
        loss_list = []
        MSE_list = []
        likelihood_node_list = []
        likelihood_edge_list = []
        loss_treatment_list = []
        loss_interference_list = []

        std_first_p_list = []

        torch.cuda.empty_cache()

        for itr in tqdm(range(train_batch)):

            #utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=args.lr / 10)
            wait_until_kl_inc = 1000

            if itr < wait_until_kl_inc:
                kl_coef = 1
            else:
                kl_coef = 1*(1 - 0.99 ** (itr - wait_until_kl_inc))

            batch_dict_encoder = utils.get_next_batch_new(train_encoder, device)
            batch_dict_graph = utils.get_next_batch_new(train_graph, device)
            batch_dict_decoder = utils.get_next_batch(train_decoder, device)

            loss, MSE,likelihood_node,likelihood_edge,loss_treatment, loss_interference = train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef,itr)

            #saving results
            loss_list.append(loss), MSE_list.append(MSE),likelihood_node_list.append(likelihood_node),likelihood_edge_list.append(likelihood_edge),
            loss_treatment_list.append(loss_treatment),loss_interference_list.append(loss_interference)

            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder
                #train_res, loss
            torch.cuda.empty_cache()

        scheduler.step()
        return kl_coef, np.mean(loss_list), np.sqrt(np.mean(MSE_list)), np.mean(likelihood_node_list), np.mean(
            likelihood_edge_list), np.mean(loss_treatment_list), np.mean(loss_interference_list)



    def val_epoch(epo,kl_coef):
        model.eval()
        loss_list = []
        MSE_list = []
        likelihood_node_list = []
        likelihood_edge_list = []
        loss_treatment_list = []
        loss_interference_list = []
        torch.cuda.empty_cache()

        for itr in tqdm(range(val_batch)):
            batch_dict_encoder = utils.get_next_batch_new(val_encoder, device)
            batch_dict_graph = utils.get_next_batch_new(val_graph, device)
            batch_dict_decoder = utils.get_next_batch(val_decoder, device)

            val_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                                                 args.num_atoms, edge_lamda=args.edge_lamda, alpha = args.alpha, beta = args.beta, kl_coef=kl_coef,
                                                 istest=False)

            loss_list.append(val_res['loss'].item())
            MSE_list.append(val_res['MSE'])
            likelihood_node_list.append(val_res['likelihood_node']), likelihood_edge_list.append(val_res['likelihood_edge']),
            loss_treatment_list.append(val_res['treatment_balancing']), loss_interference_list.append(val_res['interference_balancing'])
            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder
            # train_res, loss
            torch.cuda.empty_cache()



        return np.mean(loss_list), np.sqrt(np.mean(MSE_list)), np.mean(likelihood_node_list),np.mean(likelihood_edge_list),np.mean(loss_treatment_list),np.mean(loss_interference_list)


    for epo in range(1, args.niters + 1):
        kl_coef, rec_loss_train, RMSE_train, loss_node_train, loss_edge_train, loss_treat_train, loss_inter_train = train_epoch(
            epo)
        rec_loss_val, RMSE_val, loss_node_val, loss_edge_val, loss_treat_val, loss_inter_val = val_epoch(epo, kl_coef)

        RMSE_train_unnorm = RMSE_train * (norm_info['features_std'] + 1)
        RMSE_val_unnorm = RMSE_val * (norm_info['features_std'] + 1)

        if RMSE_val_unnorm < best_test_RMSE:
            best_test_RMSE = RMSE_val_unnorm

            ckpt_path = os.path.join(args.save, args.alias + "_experiment_" + str(
                experimentID) + "_" + args.dataset[:-1] + "_" + "_epoch_" + str(epo) + "_Rmse_" + str(
                best_test_RMSE) + '.ckpt')
            torch.save({
                'args': args,
                'state_dict': model.state_dict(),
            }, ckpt_path)


        logger.info("Experiment " + str(experimentID))
        logger.info('Epoch {:04d} [Train seq (cond on sampled tp)] |  RMSE {:.6F} | LOSS {:.6F} | LOSS_node {:.6F} | LOSS_edge {:.6F} | LOSS_treat {:.6F} | LOSS_inter {:.6F}'.format(
                epo, RMSE_train_unnorm.item(), rec_loss_train, loss_node_train, loss_edge_train, loss_treat_train,
                loss_inter_train))
        logger.info('Epoch {:04d} [Val seq (cond on sampled tp)] |  RMSE {:.6F} | LOSS {:.6F} | LOSS_node {:.6F} | LOSS_edge {:.6F} | LOSS_treat {:.6F} | LOSS_inter {:.6F}'.format(
                epo, RMSE_val_unnorm.item(), rec_loss_val, loss_node_val, loss_edge_val, loss_treat_val,
                loss_inter_val))


        # print(
        #     'Epoch {:04d} [Train seq (cond on sampled tp)] |  RMSE {:.6F} | LOSS {:.6F} | LOSS_node {:.6F} | LOSS_edge {:.6F} | LOSS_treat {:.6F} | LOSS_inter {:.6F}'.format(
        #         epo, RMSE_train_unnorm.item(), rec_loss_train, loss_node_train, loss_edge_train, loss_treat_train,
        #         loss_inter_train))
        #
        # print(
        #     'Epoch {:04d} [Val seq (cond on sampled tp)] |  RMSE {:.6F} | LOSS {:.6F} | LOSS_node {:.6F} | LOSS_edge {:.6F} | LOSS_treat {:.6F} | LOSS_inter {:.6F}'.format(
        #         epo, RMSE_val_unnorm.item(), rec_loss_val, loss_node_val, loss_edge_val, loss_treat_val,
        #         loss_inter_val))

        writer.add_scalars(f"loss/Reconstruction Loss", {
            "train": rec_loss_train,
            "val": rec_loss_val,
        }, epo)

        writer.add_scalars(f"loss/RMSE", {
            "train": RMSE_train_unnorm,
            "val": RMSE_val_unnorm,
        }, epo)

        writer.add_scalars(f"loss/Node Edge Loss", {
            "node": loss_node_val,
            "edge": loss_edge_val,
        }, epo)

        writer.add_scalars(f"loss/Banlancing Loss", {
            "treatment": loss_treat_val,
            "interference": loss_inter_val,
        }, epo)

    writer.flush()
    logger.info(best_test_RMSE)
            















