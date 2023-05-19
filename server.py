import copy
from collections import OrderedDict

import numpy as np
import torch

from utils.utils import set_up_logger, get_checkpoint_path

class Server:

    def __init__(self, args, single_client, train_clients, test_clients, model, metrics):
        self.args = args
        self.single_client = single_client
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        self.logger = set_up_logger(self.args)

        for test_client in test_clients:
            test_client.logger = self.logger
    
    def save_model(self, round):
        path = get_checkpoint_path(self.args)

        state = {
            "round": round,
            "model_state": self.model_params_dict
        }

        torch.save(state, path)

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
                    new_value = weight * value
                else:
                    new_value = old_value + weight * value

                global_param[key] = new_value.to('cuda')
        
        self.model.load_state_dict(global_param)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())


    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        print("-------------------------START TRAINING-------------------------")

        for r in range(self.args.num_rounds):
            print(f'ROUND-{r+1}')
            selected_clients = self.select_clients()

            if (r+1) % 10 == 0:
                print("-------------------------EVALUATION ON TRAIN DATASET-------------------------")
                #evaluate the current model before updating
                self.eval_train()

                #get the train evaluation
                train_score = self.metrics['eval_train'].get_results()
                
                #log the evaluation
                self.logger.log_metrics({'Train Mean IoU': train_score['Mean IoU']}, step=r + 1)
                
                #erase the old results before evaluating the updated model
                self.metrics['eval_train'].reset()

                print("FINISH EVALUATION")

                print("-------------------------EVALUATION ON TEST DATASET-------------------------")

                self.test(test_phase= False, step= r + 1)
                self.metrics['test_same_dom'].reset()
                self.metrics['test_diff_dom'].reset()

                print("FINISH EVALUATION")

                self.save_model(round = r+1)

            #train model for one round with all selected clients and update the model
            self.train_round(selected_clients)

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        for c in self.train_clients:
            c.eval_train(self.metrics['eval_train'])
        

    def test(self, test_phase = True, step = None):
        """
            This method handles the test on the test clients
        """
        if test_phase:
            print("-------------------------START TESTING-------------------------")
            step = self.args.num_rounds + 1

        for c in self.test_clients:
            if c.name == 'test_same_dom':
                print("SAME DOM")

                c.test(self.metrics['test_same_dom'], test_phase)

                test_score = self.metrics['test_same_dom'].get_results()
                self.logger.log_metrics({'Test Same Dom Mean IoU': test_score['Mean IoU']}, step = step)
            else:
                print("DIFF DOM")

                c.test(self.metrics['test_diff_dom'],test_phase)

                test_score = self.metrics['test_diff_dom'].get_results()
                self.logger.log_metrics({'Test Diff Dom Mean IoU': test_score['Mean IoU']}, step = step)

