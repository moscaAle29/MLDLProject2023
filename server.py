import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from utils.utils import set_up_logger, get_checkpoint_path
import os

count_train=0
E=1

class Server:

    def __init__(self, args, single_client, train_clients, test_clients, model, metrics):
        self.args = args
        self.single_client = single_client
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.teacher_params_dict = None
        self.teacher = None
        
        self.output_file= open("results.txt", "a")
        
        if args.test is False:
            self.logger = set_up_logger(self.args)
            for test_client in test_clients:
                test_client.logger = self.logger
                
        self.teacher_kd_params_dict = None
        self.teacher_kd = None

        self.checkpoint_round = 0
    
    def set_teacher(self, teacher):
        self.teacher = teacher
        self.teacher_params_dict = copy.deepcopy(self.teacher.state_dict())

    def set_teacher_kd(self, teacher_kd):
        self.teacher_kd = teacher_kd
        self.teacher_kd_params_dict = copy.deepcopy(self.teacher_kd.state_dict())

    
    def save_model(self, round, save_eval = True):
        if save_eval:
            dir = get_checkpoint_path(self.args)
            name = f'round{round}.ckpt'

            path = os.path.join(dir, name)

            state = {
                "round": round,
                "model_state": self.model_params_dict
            }

            torch.save(state, path)
            self.logger.save(path)
        else:
            dir = get_checkpoint_path(self.args)
            name = f'last_point.ckpt'

            path = os.path.join(dir, name)

            state = {
                "round": round,
                "model_state": self.model_params_dict
            }

            torch.save(state, path)
            self.logger.save(path)


        torch.save(state, path)
        if self.args.test is False:
            self.logger.save(path)
        
    #save the model's checkpoint for a given epoch and number of clients
    def save_model_client_epochs(self):
        dir = os.path.join('checkpoints','task2')
        name = f'{self.args.clients_per_round}clients_{self.args.num_epochs}epochs.ckpt'
        path = os.path.join(dir, name)

        state = {
            "num_clients": self.args.clients_per_round,
            "epochs":self.args.num_epochs,
            "rounds":self.args.num_rounds,
            "model_state": self.model_params_dict
        }

        torch.save(state, path)
        if self.args.test is False:
            self.logger.save(path)
            self.output_file.write(f"n_clients:{self.args.clients_per_round}, n_epochs:{self.args.num_epochs}")

    def select_clients(self):
        if self.args.setting == 'federated':
            num_clients = min(self.args.clients_per_round, len(self.train_clients))
            return np.random.choice(self.train_clients, num_clients, replace=False)
        elif self.args.setting == 'centralized':
            return [self.single_client]
        else:
            raise NotImplemented

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        global count_train, E
        count_train+=1
        updates = []
        #for all the clients
        for client in clients:
            print(f'client-{client.name}')
            #update the client's model with the one saved in the server
            client.model.load_state_dict(self.model_params_dict)
            #if we use pseudolabels do the same with the teacher
            if self.args.self_supervised is True:
                client.set_teacher(self.teacher)
            #train the client and return the number of datapoints it trained on and the updated weights
            update = client.train()
            updates.append(update)
        #launch the selected aggregation algorithm
        if self.args.FedBN is True:
            if count_train % E == 0:
                self.fedBN(updates)
        else:
            self.fedAvg(updates)
    #use the fedAvg algorithm for aggregation
    def fedAvg(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        global_num_samples = 0
        global_param = OrderedDict()
        #count the total number of datapoint in all the clients
        for local_num_samples,_ in updates:
            global_num_samples += local_num_samples

        for local_num_samples, local_param in updates:
            #compute the percentage of datapoints used by the current client and use it as the weight
            weight = local_num_samples / global_num_samples

            for key, value in local_param.items():
                #get the old value
                old_value = global_param.get(key, 0)
                #if the value is integer => we don't have a weight
                if type(old_value) == int:
                    new_value = weight * value
                else:
                    new_value = old_value + weight * value
                #save the new parameters 
                global_param[key] = new_value.to('cuda')
        #load the new weights in the server's model
        self.model.load_state_dict(global_param)
        #save the parameters
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

    #use the fedBN algorithm for aggregation
    def fedBN(self, updates):
        global_param = OrderedDict()
        n_clients=len(updates)
        
        for key in self.model_params_dict:
            if 'bn' not in key: 
                tmp = torch.zeros_like(self.model_params_dict[key], dtype=torch.float32)
                
                for client_id in range(n_clients):
                    #print(type(updates[client_id][1][key]))
                    tmp+=torch.float32(1/n_clients) * torch.float32(updates[client_id][1][key])
                global_param[key] = tmp.to('cuda')
        
        self.model.load_state_dict(global_param)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())


    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        print("-------------------------START TRAINING-------------------------")

        for r in range(self.checkpoint_round,self.args.num_rounds):
            print(f'ROUND-{r+1}')
            selected_clients = self.select_clients()

            #train model for one round with all selected clients and update the model
            self.train_round(selected_clients)

            if (r+1) % 10 == 0:
                print("-------------------------EVALUATION ON TRAIN DATASET-------------------------")
                #erase the old results before evaluating the updated model
                self.metrics['eval_train'].reset()

                #evaluate the current model before updating
                self.eval_train()

                #get the train evaluation
                train_score = self.metrics['eval_train'].get_results()
                
                #log the evaluation

                if self.args.test is False:
                    self.logger.log_metrics({'Train Mean IoU': train_score['Mean IoU']}, step=r + 1)

                self.metrics['eval_train'].reset()

                #eval on single client
                if self.single_client is not None:
                    self.single_client.eval_train(self.metrics['eval_train'])
                    train_score = self.metrics['eval_train'].get_results()
                    if self.args.test is False:
                        self.logger.log_metrics({f'Train Mean IoU{self.args.dataset}': train_score['Mean IoU']}, step=r + 1)


                #erase the old results before evaluating the updated model
                self.metrics['eval_train'].reset()
                print("FINISH EVALUATION")

                print("-------------------------EVALUATION ON TEST DATASET-------------------------")

                self.metrics['test_same_dom'].reset()
                self.metrics['test_diff_dom'].reset()
                self.test(test_phase= False, step= r + 1)

                print("FINISH EVALUATION")

                if self.args.test is False:
                    self.save_model(round = r+1)
            if self.args.test is False:
                self.save_model(round = r+1, save_eval=False)
            
            #if self_supervised is True, update teacher after some intervals or never update
            if self.args.self_supervised is True:
                if self.args.update_interval != 0:
                    if (r+1) % self.args.update_interval == 0:
                        self.teacher.load_state_dict(self.model_params_dict)
        #at the end of the training save the model we have adn the checkpoint
        if self.args.task_2_data_collection is True:
            print("-------------------------SAVING CHECKPOINT-------------------------")
            self.save_model_client_epochs()
            self.output_file.write(f"Eval MIoU(train):{train_score['Mean IoU']}, ")

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        for c in self.train_clients:
            c.eval_train(self.metrics['eval_train'])
        

    def test(self, test_phase = True, step = None, final_test=False):
        """
            This method handles the test on the test clients
        """
        if test_phase:
            print("-------------------------START TESTING-------------------------")
            step = self.args.num_rounds + 1

        same_dom_scores=[]
        diff_dom_scores=[]
        
        for c in self.test_clients:
            if c.name == 'test_same_dom':
                print("SAME DOM")

                c.test(self.metrics['test_same_dom'], test_phase)

                test_score = self.metrics['test_same_dom'].get_results()
                #for testing purposes
                if self.args.test is True:
                    print(f"same domain: {test_score['Mean IoU']}")
                else:
                    self.logger.log_metrics({'Test Same Dom Mean IoU': test_score['Mean IoU']}, step = step)
                #save the scores on a list
                if self.args.task_2_data_collection is True:
                    same_dom_scores.append(test_score['Mean IoU'])
            else:
                print("DIFF DOM")

                c.test(self.metrics['test_diff_dom'],test_phase)

                test_score = self.metrics['test_diff_dom'].get_results()
                #for testing purposes
                if self.args.test is True:
                    print(f"different domain: {test_score['Mean IoU']}")
                else:
                    self.logger.log_metrics({'Test Diff Dom Mean IoU': test_score['Mean IoU']}, step = step)
                #save the scores on a list
                if self.args.task_2_data_collection is True:
                    diff_dom_scores.append(test_score['Mean IoU'])
        
        #for sata collection we need to save the values of the chosen metrics           
        if self.args.task_2_data_collection is True and final_test is True:
            self.output_file.write(f"Test same domain:{np.max(same_dom_scores)}, ")
            self.output_file.write(f"Test different domain:{np.max(diff_dom_scores)}\n")
            self.output_file.flush()
        

