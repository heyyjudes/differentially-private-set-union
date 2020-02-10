import re
import scipy
import argparse
import numpy as np
import pandas as pd
import agm_example as agm
import scipy.stats

# Local Libs
import histogram as hgram


def test_histogram(input_df, Delta_0, n, distribution, algorithm, num_iter, delta=np.e**(-10), eps=3, passes=1, alpha=2,
                   save_hist=False):
    '''
    method for testing various histogram generation algorithms
    :param input_df: input dataframe of clean, tokenized data
    :param Delta_0: budget parameter
    :param n: n gram n
    :param distribution: noise distribution
    :param algorithm: set union / histogram algorithm
    :param num_iter: number of random shuffles to build set union over
    :param delta: dp delta parameter
    :param eps: dp epsilon parameter
    :param save_hist: whether histogram should be saved
    :return: list of release counts
    '''
    output_arr = []
    for i in range(num_iter):
        print("generating {}-gram histogram, iteration {}".format(n, i))
        new_hist = hgram.Histogram(n, input_df.sample(frac=1), 'askreddit')
        if distribution == hgram.Noise.LAPLACE:
            # calculating LaPlace parameter: using Delta_0/eps if Delta not provided (same in hist gen methods
            if algorithm == hgram.Algorithm.COUNT:
                l_param, l_rho = calculate_threshold(algorithm, distribution, eps, delta, Delta_0)
                new_hist.generate_delta_hist(delta_0=Delta_0)
                if save_hist:
                    new_hist.save_hist('count_laplace', str(Delta_0))
            elif algorithm == hgram.Algorithm.WEIGHTED:
                l_param, l_rho = calculate_threshold(algorithm, distribution, eps, delta, Delta_0)
                new_hist.generate_weighted_hist(delta_0=Delta_0, weighted_dist=distribution)
                if save_hist:
                    save_str = str(Delta_0)
                    new_hist.save_hist("weighted_laplace", save_str)
            elif algorithm == hgram.Algorithm.POLICY:
                l_param, l_rho = calculate_threshold(algorithm, distribution, eps, delta, Delta_0)
                new_hist.generate_policy_laplace_hist(delta_0=Delta_0, Gamma=l_rho + alpha*l_param, passes=passes)
                if save_hist:
                    save_str = str(Delta_0)
                    new_hist.save_hist("policy_laplace", save_str)
            elif algorithm == hgram.Algorithm.GREEDY:
                l_param, l_rho = calculate_threshold(algorithm, distribution, eps, delta, Delta_0)
                new_hist.generate_policy_greedy_hist(delta_0=Delta_0, Gamma=l_rho + alpha*l_param)
                if save_hist:
                    save_str = str(Delta_0)
                    new_hist.save_hist("greedy_laplace", save_str)
            else:
                print('Error check input algorithm input')

            output_vocab = {}
            for key, val in new_hist.ngram_hist.items():
                nval = val + np.random.laplace(0, l_param)
                if nval > l_rho:
                    output_vocab[key] = val
            output_arr.append(len(output_vocab))

        elif distribution == hgram.Noise.GAUSSIAN:

            if algorithm == hgram.Algorithm.COUNT:
                g_param, g_rho = calculate_threshold(algorithm, distribution, eps, delta, Delta_0)
                new_hist.generate_delta_hist(delta_0=Delta_0)
                if save_hist:
                    new_hist.save_hist('count_gaussian', str(Delta_0))
            elif algorithm == hgram.Algorithm.WEIGHTED:
                g_param, g_rho = calculate_threshold(algorithm, distribution, eps, delta, Delta_0)
                new_hist.generate_weighted_hist(delta_0=Delta_0, weighted_dist=distribution)
                if save_hist:
                    save_str = str(Delta_0)
                    new_hist.save_hist("weighted_gaussian", save_str)
            elif algorithm == hgram.Algorithm.POLICY:
                g_param, g_rho = calculate_threshold(algorithm, distribution, eps, delta, Delta_0)
                new_hist.generate_policy_gaussian_hist(delta_0=Delta_0, Gamma=g_rho + alpha*g_param, passes=passes)
                if save_hist:
                    save_str = str(Delta_0)
                    new_hist.save_hist("policy_gaussian", save_str)
            else:
                print('Error check input algorithm string')
                return

            output_vocab = {}
            for key, val in new_hist.ngram_hist.items():
                nval = val + np.random.normal(0, g_param)
                if nval > g_rho:
                    output_vocab[key] = val
            output_arr.append(len(output_vocab))

        else:
            print("Error check input distribution string")
    return output_arr


def calculate_threshold(algorithm, noise, eps, delta, Delta_0):
    '''
    method for threshold and parameter for each algorithm and noise
    :param algorithm: set union / histogram algorithm
    :param noise: added noise
    :param eps: dp epsilon parameter
    :param delta: dp delta parameter
    :param Delta_0: budget parameter
    :return:
    '''
    if noise == hgram.Noise.LAPLACE:
        if algorithm == hgram.Algorithm.COUNT:
            l_param = Delta_0/eps
            l_rho = 1 + (Delta_0/eps)*scipy.log(1/(2*(1-(1-delta)**(1/Delta_0))))
        elif algorithm == hgram.Algorithm.WEIGHTED or algorithm == hgram.Algorithm.POLICY:
            l_param = 1 / eps
            F_l_rho = lambda t: 1 / t + (1 / eps) * scipy.log(1 / (2 * (1 - (1 - delta) ** (1 / t))))
            l_rho = max([F_l_rho(t) for t in range(1, Delta_0 + 1)])
        elif algorithm == hgram.Algorithm.GREEDY:
            l_param = 1 / eps
            F_l_rho = lambda t: 1 / t + (1 / eps) * scipy.log(1 / (2 * (1 - (1 - delta) ** (1 / t))))
            l_rho = max([F_l_rho(t) for t in range(1, Delta_0 + 1)])
        else:
            raise Exception("Invalid algorithm for laplace noise")
        return l_param, l_rho
    elif noise == hgram.Noise.GAUSSIAN:
        if algorithm == hgram.Algorithm.COUNT:
            g_param = agm.calibrate_analytic_gaussian_mechanism(epsilon=eps, delta=delta / 2,
                                                                GS=scipy.sqrt(Delta_0), tol=1.e-12)
            g_rho = 1 + g_param * scipy.stats.norm.ppf((1 - delta / 2) ** (1 / Delta_0))
        elif algorithm == hgram.Algorithm.WEIGHTED or algorithm == hgram.Algorithm.POLICY:
            g_param = agm.calibrate_analytic_gaussian_mechanism(epsilon=eps, delta=delta / 2, GS=1, tol=1.e-12)
            F_g_rho = lambda t: 1 / scipy.sqrt(t) + g_param * scipy.stats.norm.ppf((1 - delta / 2) ** (1 / t))
            g_rho = max([F_g_rho(t) for t in range(1, Delta_0 + 1)])
        else:
            raise Exception("Invalid algorithm for gaussian noise")
        return g_param, g_rho
    else:
        raise Exception("invalid noise and algorithm combination {} {}".format(algorithm, noise))


def main():
    parser = argparse.ArgumentParser(description="DP Set Union Example")
    parser.add_argument("--D0",
                        type=int,
                        default=10,
                        help="input sensitivity")

    parser.add_argument("--eps",
                        type=float,
                        default=3,
                        help="epsilon dp parameter")

    parser.add_argument("--delta",
                        type=float,
                        default=np.e**(-10),
                        help="threshold parameter")

    parser.add_argument("--alg",
                        type=str,
                        default="policy",
                        help="algorithm type: count, weighted, policy")

    parser.add_argument("--alpha",
                        type=float,
                        default=5,
                        help="delta dp parameter")

    parser.add_argument("--noise",
                        type=str,
                        default="gaussian",
                        help="noise type: laplace, gaussian, rdp")

    parser.add_argument('--ngram',
                        type=int,
                        default=1,
                        help="n for histogram ngrams")

    parser.add_argument('--trials',
                        type=int,
                        default=1,
                        help="number of trials to run for release count average")

    parser.add_argument('--passes',
                        type=int,
                        default=1,
                        help='number of passes made through user list')

    parser.add_argument('--dataset',
                        type=str,
                        default="data/clean_askreddit.csv",
                        help='path to dataset in .csv format with "clean_data" column')

    parser.add_argument("--save_histogram",
                        action="store_true",
                        default=False,
                        help="For saving current histogram")

    args = parser.parse_args()

    # Check inputs and convert to ENUMS
    if re.match(args.noise,'laplace', re.IGNORECASE):
        dist = hgram.Noise.LAPLACE
    elif re.match(args.noise,'gaussian', re.IGNORECASE):
        dist = hgram.Noise.GAUSSIAN
    else:
        raise Exception("Please enter 'laplace', 'gaussian' or 'rdp' as --noise parameter")

    if re.match(args.alg,'count', re.IGNORECASE):
        alg = hgram.Algorithm.COUNT
    elif re.match(args.alg,'weighted', re.IGNORECASE):
        alg = hgram.Algorithm.WEIGHTED
    elif re.match(args.alg,'policy', re.IGNORECASE):
        alg = hgram.Algorithm.POLICY
    elif re.match(args.alg, 'greedy', re.IGNORECASE):
        alg = hgram.Algorithm.GREEDY
    else:
        raise Exception("Please enter 'count', 'weighted', 'policy', or 'greedy' as --alg parameter")

    reddit_df = pd.read_csv(args.dataset, index_col=0)
    reddit_df = reddit_df.dropna()

    result_arr = test_histogram(input_df=reddit_df, Delta_0=args.D0, n=args.ngram, distribution=dist, algorithm=alg,
                                eps=args.eps, delta=args.delta, num_iter=args.trials, passes=args.passes,
                                alpha=args.alpha, save_hist=args.save_histogram)

    print("Output for {} {} with {} trials for alpha {} run".format(alg, dist, args.trials, args.alpha))
    print("Mean released ngram count:", np.mean(result_arr))
    print("STD of released ngram count: ", np.std(result_arr))


if __name__ == "__main__":
    np.random.seed(42)
    main()


