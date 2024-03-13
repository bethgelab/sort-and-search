import argparse, os, pickle, gc, copy
import numpy as np
import pandas as pd
from utils import seed_everything
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Search and Sort Framework')
    parser.add_argument('--data_dir', type=str, default='../results/', help='save files directory')
    parser.add_argument('--dataset_name', type=str, default='lifelong-imagenet', choices=['lifelong-imagenet', 'lifelong-cifar10'])
    parser.add_argument('--A_matrix_name', type=str, default='../feats/A.npy', help='file name of the A matrix')
    parser.add_argument('--rank_number', type=int, default=6000, help='Number of points used for ranking (rest used for querying)')
    parser.add_argument('--transpose', action='store_true', help='Transpose A matrix, invert the sampling and querying problem')
    parser.add_argument('--ranking_mode', type=str, default='sum', choices=['sum','recursive_sum'], help='Choose a ranking algorithm')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for experiments')
    args = parser.parse_args()
    return args

def uniform_sampling(query_len, num_queries):
    # Calculate step size and start point
    step = query_len//num_queries
    start = step//2 if query_len==step*num_queries else (query_len - step*num_queries)
    
    # Sample K points
    sampled_points = np.arange(start, query_len, step)
    assert(len(sampled_points)==num_queries)
    return sampled_points


def random_sampling(query_len, num_queries):
    sampled_points = np.random.choice(query_len, num_queries, replace=False)
    sampled_points = np.sort(sampled_points)
    assert(len(sampled_points)==num_queries)
    return sampled_points


def dynamic_programming_threshold(A):
    Aopt = copy.deepcopy(A)
    Aopt[Aopt==0] = -1
    idx = np.argmax(np.cumsum(Aopt, axis=1), axis=1)
    # dim m
    return idx


def create_pred_array(A, idx):
    # setting the first k elements of each row to be 1, where k comes from the thresholds `idx`
    pred_array = np.zeros_like(A)
    mask = np.arange(A.shape[1]) < idx[:, None]
    pred_array[mask] = 1
    return pred_array


def sum_ranking(A, idx):
    sum_ranking = A[idx].sum(axis=0)
    order = np.flip(np.argsort(sum_ranking))
    return order


def recursive_sum_ranking(A, idx):
    # Two step approximation works well enough
    sum_bins = A[idx].sum(axis=0)
    order = np.flip(np.argsort(sum_bins))

    # an array of size m --> indexes of thresholds for each model in the ordered matrix 
    thresh_ordered = dynamic_programming_threshold(A[:, order])
    # permute to fix ordering
    sum_bins_ordered = sum_bins[order]
    uniq_bins = np.unique(sum_bins_ordered)

    # we look at each bin, take all the thresholds that lie within each bin, and take the model sums for those thresholds, and then compute the sum again
    for bin in uniq_bins:
        idx = np.nonzero(sum_bins_ordered==bin)[0]
        # find thresh in this idx
        # across models
        thresh_idx = np.nonzero(np.all([[thresh_ordered >= idx.min()], [thresh_ordered <= idx.max()]], axis=0))[1]
        A_New = A[thresh_idx][:, order[idx]]
        improved_bins = A_New.sum(axis=0)
        # new ordering within current bin
        new_order = np.flip(np.argsort(improved_bins))
        # order withing current bin
        order[idx] = order[idx[new_order]]
    return order
    

def do_search_subsampled(A, order, query_size, mode='uniform'):
    A = A[:, order]
    if mode=='uniform':
        samples = uniform_sampling(A.shape[1], query_size)
    elif mode=='random':
        samples = random_sampling(A.shape[1], query_size)
    idx = A[:, samples].sum(axis=1).astype(int)
    # ensure that the indices don't shoot beyond the length of the sub-sampled samples
    idx[idx>=samples.shape[0]] = samples.shape[0] - 1
    Apred = create_pred_array(A, samples[idx])
    global_dataset_diff = np.abs(A-Apred).mean(axis=1)*100
    global_dataset_accpred = Apred.mean(axis=1)*100

    idx = dynamic_programming_threshold(A[:,samples])
    Apred = create_pred_array(A, samples[idx])
    sample_level_diff = np.abs(A-Apred).mean(axis=1)*100
    sample_level_accpred = Apred.mean(axis=1)*100
    return global_dataset_diff, global_dataset_accpred, sample_level_diff, sample_level_accpred


def do_searching(A, order, query_array):
    diffs, accpreds = {}, {}
    accpreds_gt = A.mean(axis=1)*100
    diff_gt = np.zeros_like(accpreds_gt)
    diffs['GT'] = diff_gt
    accpreds['GT'] = accpreds_gt

    global_diff_opt, global_accpred_opt,  sample_diff_opt, sample_accpred_opt = do_search_subsampled(A, order, A.shape[1], mode='uniform')
    diffs['Optimal_global'] = global_diff_opt
    accpreds['Optimal_global'] = global_accpred_opt
    diffs['Optimal_sample'] = sample_diff_opt
    accpreds['Optimal_sample'] = sample_accpred_opt

    for query in tqdm(query_array, ascii=True, total=len(query_array)):
        global_diff, global_accpred, sample_diff, sample_accpred = do_search_subsampled(A, order, query, mode='uniform')
        diffs['Uniform_global_'+str(query)] = global_diff
        accpreds['Uniform_global_'+str(query)] = global_accpred
        diffs['Uniform_sample_'+str(query)] = sample_diff
        accpreds['Uniform_sample_'+str(query)] = sample_accpred

        global_diff, global_accpred, sample_diff, sample_accpred = do_search_subsampled(A, order, query, mode='random')
        diffs['Random_global_'+str(query)] = global_diff
        accpreds['Random_global_'+str(query)] = global_accpred
        diffs['Random_sample_'+str(query)] = sample_diff
        accpreds['Random_sample_'+str(query)] = sample_accpred
        
    return diffs, accpreds


if __name__ == '__main__': 
    args = parse_args()

    # seed everything
    seed_everything(args.seed)

     # A --> m x n
    A = np.load(os.path.join(args.data_dir, args.A_matrix_name))
    if args.transpose:
        A = A.T
    
    transpose_str = 'sample_eval' if args.transpose == True else 'model_eval'
    # getting ranking and search idxes 
    if os.path.exists(os.path.join(args.data_dir, 'rank_idx_{}_{}_{}_{}_{}_.npy'.format(args.dataset_name, transpose_str, args.rank_number, args.ranking_mode, args.seed))):
        idx = np.load(os.path.join(args.data_dir, 'rank_idx_{}_{}_{}_{}_{}_.npy'.format(args.dataset_name, transpose_str, args.rank_number, args.ranking_mode, args.seed)))
    else:
        if args.transpose:
            # to ensure that we always eval on the held-out test split (assumes the test-datasets come last in the A columns)
            idx = np.arange(args.rank_number)
        else:
            # get random K models for ranking
            idx = np.random.choice(A.shape[0], args.rank_number, replace=False)
        np.save(os.path.join(args.data_dir, 'rank_idx_{}_{}_{}_{}_{}_.npy'.format(args.dataset_name, transpose_str, args.rank_number, args.ranking_mode, args.seed)), idx)
    
    # held-out test set
    search_idx = np.setdiff1d(np.arange(A.shape[0]), idx)
    
    # getting ranking order of size m (if transpose) / n (if not transpose)
    if os.path.exists(os.path.join(args.data_dir, 'rank_order_{}_{}_{}_{}_{}.npy'.format(args.dataset_name, transpose_str, args.rank_number, args.ranking_mode, args.seed))):
        order = np.load(os.path.join(args.data_dir, 'rank_order_{}_{}_{}_{}_{}.npy'.format(args.dataset_name, transpose_str, args.rank_number, args.ranking_mode, args.seed)))
    else:
        if args.ranking_mode=='sum':
            order = sum_ranking(A, idx)
        elif args.ranking_mode=='recursive_sum':
            assert(not args.transpose)
            order = recursive_sum_ranking(A, idx)
        assert(args.ranking_mode in ['sum', 'recursive_sum'])
        np.save(os.path.join(args.data_dir, 'rank_order_{}_{}_{}_{}_{}.npy'.format(args.dataset_name, transpose_str, args.rank_number, args.ranking_mode, args.seed)), order)


    
    if args.dataset_name == 'lifelong-imagenet':
        queries = [8, 16, 32] if args.transpose else [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    elif args.dataset_name == 'lifelong-cifar10':
        queries = [8, 16, 32, 64, 128, 256, 512, 1024, 2048] if args.transpose else [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    else:
        raise ValueError('Eval-setting unknown')

   
    # getting search results
    A = A[search_idx]
    diffs, accpreds = do_searching(A, order, query_array=queries)

    # Save results as dataframes
    df_diffs = pd.DataFrame(diffs)
    df_accpreds = pd.DataFrame(accpreds)
    df_diffs.to_csv(os.path.join(args.data_dir, 'mae_{}_{}_{}_{}_{}.csv'.format(args.dataset_name, transpose_str, args.rank_number, args.ranking_mode, args.seed)))
    df_accpreds.to_csv(os.path.join(args.data_dir, 'accs_{}_{}_{}_{}_{}.csv'.format(args.dataset_name, transpose_str, args.rank_number, args.ranking_mode, args.seed)))
