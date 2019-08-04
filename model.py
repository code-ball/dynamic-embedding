import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class dynemb(nn.Module):
    def __init__(self, nsize, em_size, num_dynamics, eval_batch_size):
        super(dynemb, self).__init__()
        self.em_size = em_size
        self.num_dynamics = num_dynamics
        self.eval_batch_size = eval_batch_size
        self.initial_embeddings = nn.Embedding(nsize, em_size)
        self.latest_embeddings = Variable(torch.zeros(nsize, em_size), requirev1_grad=False)
        self.embed_available = Variable(torch.zeros(nsize), requirev1_grad=False)
        self.g_0 = nn.Linear( 2 *em_size, 1)
        self.g_1 = nn.Linear(2 * em_size, 1)
        self.psi = Variable(torch.ones(num_dynamics, 1), requirev1_grad=True)
        self.relu_act = nn.ReLU()

    def computeScore(self, v1_emb, v2_emb, k):
        if k == 0:
            g_inp = torch.cat((v1_emb, v2_emb), 2)
            score = self.g_0(g_inp)
        if k == 1:
            g_inp = torch.cat((v1_emb, v2_emb), 2)
            score = self.g_1(g_inp)
        return score

    def intensityActivation(self, sc, param):
        intensity = (param) * torch.log(1.0 + torch.exp(sc / param))
        return intensity

    def computeSurvival(self, N, node_list, v1_curr, v1_curr_em, v2_curr, v2_curr_em):
        l_surv = 0.0
        v1_surv = 0.0
        v2_surv = 0.0
        i = 0

        while i < N:
            v1_other = random.choice(node_list)
            v2_other = random.choice(node_list)
            if v1_other in [v1_curr, v2_curr] or v2_other in [v1_curr, v2_curr]:
                continue
            i += 1
            if self.embed_available.data[v1_other]:
                v1_other_emb = self.latest_embeddings[v1_other].view(1, 1, self.em_size)
            else:
                emb_inp = Variable(torch.LongTensor([[v1_other]]))
                v1_other_emb = self.initial_embeddings(emb_inp)
                self.embed_available.data[v1_other] = 1

            if self.embed_available.data[v2_other]:
                v2_other_emb = self.latest_embeddings[v2_other].view(1, 1, self.em_size)
            else:
                emb_inp = Variable(torch.LongTensor([[v2_other]]))
                v2_other_emb = self.initial_embeddings(emb_inp)
                self.embed_available.data[v2_other] = 1

            for k in range(self.num_dynamics):
                g_v1_score = self.computeScore(v1_curr_em, v2_other_emb, k)
                v1_surv += self.intensityActivation(g_v1_score, self.psi[k])

                g_v2_score = self.computeScore(v1_other_emb, v2_curr_em, k)
                v2_surv += self.intensityActivation(g_v2_score, self.psi[k])

        l_surv += (v1_surv + v2_surv) / N
        return l_surv


    def forward(self, input, node_list, N, eval_phase=0):
        outputv1_intent = []
        outputv1_surv = []

        for i in range(input.size(0)):
            inp_tuple = input[i]

            v1 = int(inp_tuple.data[0])
            v2 = int(inp_tuple.data[1])
            l = int(inp_tuple.data[3])
            k = int(inp_tuple.data[4])

            if self.embed_available.data[int(inp_tuple.data[0])]:
                v1_emb = self.latest_embeddings[int(inp_tuple.data[0])].view(1, 1, self.em_size)
            else:
                emb_inp = Variable(torch.LongTensor([[int(inp_tuple.data[0])]]))
                v1_emb = self.initial_embeddings(emb_inp)
                self.embed_available.data[int(inp_tuple.data[0])] = 1

            if self.embed_available.data[int(inp_tuple.data[1])]:
                v2_emb = self.latest_embeddings[int(inp_tuple.data[1])].view(1, 1, self.em_size)
            else:
                emb_inp = Variable(torch.LongTensor([[int(inp_tuple.data[1])]]))
                v2_emb = self.initial_embeddings(emb_inp)
                self.embed_available.data[int(inp_tuple.data[1])] = 1

            # Compute the intensity for the current pair and event type

            g_score = self.computeScore(v1_emb, v2_emb, k)
            g_intensity = self.intensityActivation(g_score, self.psi[k])
            outputv1_intent.append(g_intensity)

            # Compute the survival prob corresponding to current event

            g_surv = self.computeSurvival(N, node_list, v1, v1_emb, v2, v2_emb)
            outputv1_surv.append(g_surv)

            if (eval_phase == 1 and i != self.eval_batch_size - 1) or \
                    eval_phase == 2:
                continue

            self.latest_embeddings.data[int(inp_tuple.data[0])] = v1.data
            self.latest_embeddings.data[int(inp_tuple.data[1])] = v2.data
            self.embed_available[int(inp_tuple.data[0])] = 1
            self.embed_available[int(inp_tuple.data[1])] = 1

        outputv1_intensity = torch.stack(outputv1_intent, 1).squeeze(2)
        outputv1_surv = torch.stack(outputv1_surv, 1).squeeze(2)
        return outputv1_intensity, outputv1_surv
