# Towards Real World Federated Learning
### Machine Learning and Deep Learning 2023
Giacomo Bastiani s303217
Alessandro Mosca s309595
Minh Triet Ngo s309062

#### Politecnico di Torino
 Federated Learning (FL) is a highly promising approach in machine learning that effectively tackles the challenges of data privacy and sharing. In the context of autonomous driving, a paper called FFREEDA designs a FL framework and acknowledges the realistic assumption that clients (e.g., vehicles) lack access to ground-truth labels. A proposed solution named LADD within the FFREEDA framework  leverages a combination of techniques with a particular focus on client clustering. This clustering of clients plays a crucial role in facilitating the FL process for autonomous driving applications. In our work, we present an alternative approach to client clustering within the FL framework by utilizing variational autoencoders (VAEs).

## How to run
The ```main.py``` orchestrates training. All arguments need to be specified through the ```args``` parameter (options can be found in ```utils/args.py```).
Example of FedAvg experiment 
```bash
python main.py --dataset idda --model deeplabv3_mobilenetv2 --num_rounds 200 --num_epochs 2 --clients_per_round 8 
```
## How to test
In order to test the algorithm you need to checkout the test branch using 
```bash
git clone -b test https://github.com/moscaAle29/MLDLProject2023.git 
```
Launching the main.py function will start the training, if your goal is just to evaluate the result please select a number of clients and epochs and run the test.py file specifying the configuration you want to evaluate using the ```args``` parameter.
#### task 1
In the first task we are asked to run experiments in order to find the better configuration, we included the checkpoint for the best configuration, to test run 
```bash
python test.py --task 1  --setting centralized --dataset idda --dataset2 idda --model deeplabv3_mobilenetv2 --num_rounds 100
```
#### task 2
In task 2 we are asked to move to a federated learning setting using the data augmentation techniques found in the previous steps and selecting the number of clients and epochs:
- Number of clients: [2 4 8]
- Number of epochs: [1 3 6 9 12]
```bash
python test.py --task 2 --setting federated --dataset idda --dataset2 idda --model deeplabv3_mobilenetv2 --num_rounds 100 --num_epochs *selected_value* --clients_per_round *selected_value*
```
#### task 3.2
In this task we are asked to move to an unlabeled dataset using a model trained on a labeled one choosing batch size and learning rate, you can choose 
- Batch size: 2, Learning Rate:0.01
- Batch size: 4, Learning Rate:0.01
- Batch size: 6, Learning Rate:0.01
- Batch size: 8, Learning Rate:0.01
- Batch size: 10, Learning Rate:0.01
- Batch size: 4, Learning Rate:0.03
- Batch size: 8, Learning Rate:0.03
- Batch size: 4, Learning Rate:0.05
```bash
python test.py --task 3.2 --setting centralized --dataset gta5 --dataset2 idda --model deeplabv3_mobilenetv2 --num_rounds 100 --lr *selected_value* --bs *selected_value*
```
#### task 3.4
We now want to use a domain adaptation technique called FDA, choosing Batch size and the value of alpha for FDA
- Batch size: 4
- Alpha value: [0.1, 0.05, 0.01,0.005]
```bash
python test.py --task 3.4 --setting centralized --dataset gta5 --dataset2 idda --model deeplabv3_mobilenetv2 --num_rounds 100 --fda_alpha *selected_value* --bs 4
```

#### task 4.2
In this task we want to use a self-training technique using a teacher model loading checkpoints from *task 3.2* specifying number of clients and update interval
- Number of Clients: [2, 8]
- Number of local epochs: [0, 1]
*update interval 0 equals never update*
```bash
python test.py --task 4.2 --setting federated --dataset gta5 --dataset2 idda --model deeplabv3_mobilenetv2 --num_rounds 100 --clients_per_round *selected_value* --update_interval *selected_value*
```

#### task 4.3
In this task we want to use a self-training technique using a teacher model loading checkpoints from *task 3.4* specifying number of clients and update interval
- Number of Clients: [2, 8]
- Number of local epochs: [0, 1]
*update interval 0 equals never update*
```bash
python test.py --task 4.3 --setting federated --dataset gta5 --dataset2 idda --model deeplabv3_mobilenetv2 --num_rounds 100 --clients_per_round *selected_value* --update_interval *selected_value*
```
#### task 5
In this task we implemented a VAE to cluster the clients, for the ablation study you can choose to test using Knowledge Distillation and Sthocastic Weight Averaging
- Knowledge Distillation: --kd
- Sthocastic Weight Averaging: --swa
```bash
python test.py --task 5 --setting federated --dataset gta5 --dataset2 idda --model deeplabv3_mobilenetv2 --num_rounds 100 --kd --swa
```
   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
