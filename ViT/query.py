import numpy as np
import torch
import time
'''
   drop 0-127 tokens on first layer, and check each layer's time consuming
   use layer_timing.py to generate

'''

import random


class fake_oracle():
    def fit_for_Xeon_E5_2690_v4_2dot60GHz_one_thread(self, profiling_filename):

        self.total_token_num = 197
        prof = torch.load(profiling_filename)
        attn_time_consumed = prof['attn'].permute(1, 0)
        MLP_time_consumed = prof['MLP'].permute(1, 0)
        encoder_time_consumed = prof['encoder']
        endphase_time_consumed = prof['endphase']
        total_time_consumed = prof['total']
        prepare_time_consumed = prof['prepare']

        block_time_consumed = attn_time_consumed + MLP_time_consumed
        estimated_remaining_time = MLP_time_consumed + endphase_time_consumed

        for i in range(10, -1, -1):
            estimated_remaining_time[i] += estimated_remaining_time[i + 1]
            estimated_remaining_time[i] += attn_time_consumed[i + 1]

        # the shapes are (12,128) because I only droped upto 128...
        self.estimated_remaining_time = estimated_remaining_time
        self.attn_time_consumed = attn_time_consumed
        self.MLP_time_consumed = MLP_time_consumed
        self.prepare_time_consumed = prepare_time_consumed
        self.block_time_consumed = block_time_consumed
        self.encoder_time_consumed = encoder_time_consumed
        self.endphase_time_consumed = endphase_time_consumed
        self.total_time_consumed = total_time_consumed

        self.max_drop_num = estimated_remaining_time.shape[1] - 1
        token_number = torch.Tensor([self.total_token_num - i for i in range(self.max_drop_num + 1)])

        self.per_layer_attn_fit = []
        self.per_layer_MLP_fit = []
        self.remaining_time_fit = []

        # please use coeffs attribute to get the coefficients
        for i in range(12):
            self.per_layer_attn_fit.append(np.poly1d(np.polyfit(token_number, attn_time_consumed[i], 2)))
            self.per_layer_MLP_fit.append(np.poly1d(np.polyfit(token_number, MLP_time_consumed[i], 1)))
            self.remaining_time_fit.append(np.poly1d(np.polyfit(token_number, estimated_remaining_time[i], 2)))

        # bigger means that we lose more accuracy
        self.drop_punishment_coefficient = [50, 35, 18, 17, 16, 15, 14, 13, 12, 10, 10, 10]
        self.decision_factor = 13
        self.latency_reduce_gain_coefficient = [self.remaining_time_fit[i].coeffs[1] for i in range(12)]
        self.attn_score = None

    def make_drop_v1(self, layer_index, start_time, budget, last_layer_MLP_time, attn_matrix):
        token_num_left = int(attn_matrix.shape[-1])
        token_num_droped = self.total_token_num - token_num_left

        if token_num_droped >= self.max_drop_num:
            return None

        if layer_index > 0:
            env_factor = 1 + 0.5 * (last_layer_MLP_time / self.MLP_time_consumed[layer_index - 1][token_num_droped] - 1)
        else:
            env_factor = 1
        budget_left = budget - (time.time() - start_time)
        estimate_time = env_factor * self.estimated_remaining_time[layer_index][token_num_droped]

        if estimate_time < budget_left:
            # self.log[layer_index]['drop_type'] = 'no drop'
            return None
        low = token_num_droped + 1
        high = self.max_drop_num
        # satisfied = False

        while (low <= high):
            cur = int((low + high) >> 1)
            if env_factor * self.estimated_remaining_time[layer_index][cur] < budget_left:
                # satisfied = True
                high = cur - 1
            else:
                low = cur + 1
        min_satisfied_drop = cur - token_num_droped
        drop_candidates = [i for i in range(1, token_num_left - 1)]
        drop_indexes = random.sample(drop_candidates, min_satisfied_drop)
        candidates = [i for i in range(token_num_left)]
        for idx in drop_indexes:
            candidates.remove(idx)
        return candidates

    def make_drop_v0(self, layer_index, start_time, budget, last_layer_MLP_time, attn_matrix):
        '''
            first we check if the budget is already satisfied for this layer
        '''

        # attn_matrix_shape : batch, layer_num, token, token
        token_num_left = int(attn_matrix.shape[-1])
        token_num_droped = self.total_token_num - token_num_left

        if token_num_droped >= self.max_drop_num:
            return None

        if layer_index > 0:
            env_factor = 1 + 0.5 * (last_layer_MLP_time / self.MLP_time_consumed[layer_index - 1][token_num_droped] - 1)
        else:
            env_factor = 1
        budget_left = budget - (time.time() - start_time)

        estimate_time = env_factor * self.estimated_remaining_time[layer_index][token_num_droped]
        # self.log[layer_index]['env_factor'] = env_factor
        # self.log[layer_index]['estimate_time'] = estimate_time
        # self.log[layer_index]['budget_left'] = budget_left
        # self.log[layer_index]['last_layer_MLP_time'] = last_layer_MLP_time

        # attn_score_for_this_layer = (attn_matrix.sum(dim=-2).sum(dim=1).squeeze(dim=0))
        # if layer_index==0:
        #     self.attn_score = attn_score_for_this_layer
        # else:
        #     self.attn_score = 0.5* ( self.attn_score + attn_score_for_this_layer)

        # attn_score_for_this_layer = self.attn_score

        if estimate_time < budget_left:
            # self.log[layer_index]['drop_type'] = 'no drop'
            return None

        else:
            # self.log[layer_index]['drop_type'] = 'drop'
            attn_score_for_this_layer = (attn_matrix.sum(dim=-2).sum(dim=1).squeeze(dim=0))[1:]
            sorted_attn_score, sorted_indexes = attn_score_for_this_layer.sort()
            head_num = attn_matrix.shape[1]
            attn_score_sum = head_num * (token_num_left - 1)

            for i in range(1, token_num_left - 1):
                sorted_attn_score[i] += sorted_attn_score[i - 1]

            low = token_num_droped + 1
            high = self.max_drop_num

            satisfied = False

            while (low <= high):
                cur = int((low + high) >> 1)
                if env_factor * self.estimated_remaining_time[layer_index][cur] < budget_left:
                    satisfied = True
                    high = cur - 1
                else:
                    low = cur + 1

            min_satisfied_drop = cur - token_num_droped

            low = token_num_droped + 1
            high = self.max_drop_num

            satisfied = False
            while (low <= high):
                cur = int((low + high) >> 1)
                latency_gain = (env_factor * self.estimated_remaining_time[layer_index][cur] / budget_left
                                ) * env_factor * (self.estimated_remaining_time[layer_index][token_num_droped] -
                                                  self.estimated_remaining_time[layer_index][cur])
                acc_punish = (sorted_attn_score[cur - token_num_droped - 1] /
                              attn_score_sum) * self.drop_punishment_coefficient[layer_index]
                if latency_gain * self.decision_factor > acc_punish:
                    low = cur + 1
                    satisfied = True
                else:
                    high = cur - 1

            if not satisfied:
                return None

            dest_drop = cur - token_num_droped
            if dest_drop > min_satisfied_drop:
                dest_drop = min_satisfied_drop

            if dest_drop + token_num_droped > self.max_drop_num:
                dest_drop = self.max_drop_num - token_num_droped

            # drop_ratio = dest_drop / (token_num_left -1 )
            # drop_ratio_factor = 1-(layer_index/12)
            # bar = int((token_num_left-1)*0.03)
            # if drop_ratio*drop_ratio_factor > 0.03:
            #     dest_drop = int(bar + 0.3*(dest_drop-bar))

            if dest_drop > 12:
                dest_drop = 8 + int((dest_drop - 8) * 0.3)
            elif dest_drop > 6:
                dest_drop = int(dest_drop * 0.8)

            # self.log[layer_index]['drop_num'] = dest_drop
            # self.log[layer_index]['min_satisfied'] = min_satisfied_drop
            # self.log[layer_index]['estimate_time'] = estimate_time
            # self.log[layer_index]['budget_left'] = budget_left
            # self.log[layer_index]['last_layer_time'] = last_layer_MLP_time

            self.log[layer_index]['drop_indexes'] = sorted_indexes[:dest_drop]

            indexes2ret = torch.cat((torch.Tensor([0]).int(), sorted_indexes[dest_drop:]))

            # self.attn_score = self.attn_score[indexes2ret]
            return indexes2ret

    def make_drop(self, layer_index, start_time, budget, last_layer_MLP_time, attn_matrix):
        return self.make_drop_v1(layer_index, start_time, budget, last_layer_MLP_time, attn_matrix)

    def __init__(self, filename):
        self.fit_for_Xeon_E5_2690_v4_2dot60GHz_one_thread(filename)
        self.log = [{} for i in range(12)]

    def print_log(self):
        for i in range(12):
            print('layer ', i)
            for k, val in self.log[i].items():
                print(k, val)
            print('')


if __name__ == '__main__':
    my_oracle = fake_oracle('profiling/profiling_result/profiling_result')
    print(my_oracle.total_time_consumed[0])
    print(my_oracle.prepare_time_consumed[0])
    # print(my_oracle.attn_time_consumed[0][0])
    # print(my_oracle.MLP_time_consumed[0][0])
    print(my_oracle.block_time_consumed[:, 0])
    # print(my_oracle.block_time_consumed.sum(dim=0)[0])
    print(my_oracle.endphase_time_consumed[0])