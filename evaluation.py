from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.autograd import Variable
import torch
import numpy as np


def evaluate_pred(model, test_data, batch_num, N, N_t, ts):
    model.eval()

    T = 102998.97
    true_lab = []
    predict_lab = []
    mae = []
    rmse = []
    node_list = [a for a in range(100)]
    timept = 0

    for i in range(batch_num):
        t_next = 0.0
        mini_batch = Variable(test_data[i])
        inp_tuple = mini_batch[0]
        v1 = int(inp_tuple.data[0])
        v2 = int(inp_tuple.data[1])
        curr_time = inp_tuple.data[2]
        mini_batch_node_list = [a for a in node_list if (a != v1 and a != v2)]
        true_lab.append(inp_tuple.data[5])

        cur_intensity, cur_survival = model(mini_batch, mini_batch_node_list, N, 1)

        for k in range(N_t):
            t_rand = np.random.uniform(curr_time, T)
            mini_batch[0].data[2] = t_rand
            output_intensity, output_survival = model(mini_batch, mini_batch_node_list, N, 2)
            survival = torch.exp(-output_survival)
            f = output_intensity * survival
            density = f.data.numpy()
            t_next += t_rand * density

        t_predict = curr_time + (t_next / N_t)
        predict_lab.append(t_predict.reshape(1)[0])

    true_lab = np.array(true_lab, np.float)
    predict_lab = np.array(predict_lab, np.float)
    mae_temp = mean_absolute_error(true_lab, predict_lab)
    rmse_temp = np.sqrt(mean_squared_error(true_lab, predict_lab))
    mae.append(mae_temp)
    rmse.append(rmse_temp)

    return mae, rmse
