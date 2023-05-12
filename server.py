import copy
from collections import OrderedDict

import numpy as np
import torch

from utils.utils import set_up_logger


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        self.logger = set_up_logger()

        for test_client in test_clients:
            test_client.logger = self.logger

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []

        for client in clients:
            print(f'client-{client.name}')

            client.model.load_state_dict(self.model_params_dict)

            update = client.train()
            updates.append(update)

        self.aggregate(updates)

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        global_num_samples = 0
        global_param = OrderedDict()

        for local_num_samples,_ in updates:
            global_num_samples += local_num_samples

        for local_num_samples, local_param in updates:
            weight = local_num_samples / global_num_samples

            for key, value in local_param.items():
                old_value = global_param.get(key, 0)
                if type(old_value) == int:
                    new_value = weight * (value.type(torch.FloatTensor).cpu())
                else:
                    new_value = old_value.cpu() + weight * (value.type(torch.FloatTensor).cpu())

                global_param[key] = new_value.to('cuda')
        
        self.model.load_state_dict(global_param)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())


    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        print("-------------------------START TRAINING-------------------------")

        for r in range(self.args.num_rounds):
            print(f'ROUND-{r}')
            
            if r % 10 == 0:
                selected_clients = self.select_clients()

                #evaluate the current model before updating
                self.eval_train()

                #get the train evaluation
                train_score = self.metrics['eval_train'].get_results()
                
                #log the evaluation
                self.logger.log_metrics({'Train Mean IoU': train_score['Mean IoU']}, step=r + 1)
                
                #erase the old results before evaluate the updated model
                self.metrics['eval_train'].reset()

                print("FINISH TRAIN EVALUATION")

            #train model for one round with all selected clients and update the model
            self.train_round(selected_clients)

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        print("-------------------------EVALUATION METRICS-------------------------")
        for c in self.train_clients:
            c.eval_train(self.metrics['eval_train'])
        

    def test(self):
        """
            This method handles the test on the test clients
        """
        print("-------------------------START TESTING-------------------------")

        for c in self.test_clients:
            if c.name == 'test_same_dom':
                print("-------------------------SAME DOM-------------------------")

                c.test(self.metrics['test_same_dom'])

                test_score = self.metrics['test_same_dom'].get_results()
                self.logger.log_metrics({'Test Same Dom Mean IoU': test_score['Mean IoU']})
            else:
                print("-------------------------DIFF DOM-------------------------")

                c.test(self.metrics['test_diff_dom'])
                self.logger.log_metrics({'Test Diff Dom Mean IoU': test_score['Mean IoU']})

