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
Launching the main.py function will start the training, if your goal is just to evaluate the result please select a number of clients and epochs and run the test.py file specifying
- Number of clients: [2 4 8]
- Number of epochs: [1 3 6 9 12]
```bash
python test.py --task_2_test --dataset idda --model deeplabv3_mobilenetv2 --num_rounds 100 --num_epochs *Number of epochs* --clients_per_round *Number of Clients*
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