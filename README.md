# Self-adaptive in Apache Storm
This version is under development. It is an extension of the v3.1 branch, adding the functionality to select the best predictor based on a reinforcement learning model. Although a first version was used, there are still improvements.
Implementation of MAPE Model for Storm (2.8.0) extension. This project presents the self-adaptive system for to modify the number of active/inactive replicas for each pool of operators in the SPS application. The goal is to analyse differences metrics (i.e. input, executed time, queue) and to plan the changes necessary for to process all input events in the SPS.   

## Configuration
The config file '[config.yaml](configs/config.yaml)' has three principals parameters: `nimbus`, `redis`, `storm`. 

The parameter `nimbus` is related to Nimbus component in Storm. The variables `host` and `port` are the IP location of Nimbus.

The parameter `redis` is related to Redis cache. The variables `host` and `port` are the IP location of Redis.

The parameter `predictor` is related to Predictor API. The variables `host` and `port` are the IP location of Predictor API.

The params `storm` is related to Apache Storm.

The variable `deploy` is related to application deployment. 
- `duration` is the time of the experiment
- `script` is the app script that Apache will deploy
- `analyze` is the parameters if the system adapts (or not) the Storm application.  

The variable `adaptive` is related to self-adaptive system.
- `time_window_size` size of the time period where a sample is obtained. Its value is in seconds.
- `benchmark_samples` numbers of samples used by the benchmark.
- `analyze_samples` numbers of samples by MAPE model.
- `preditive_model` model used by input prediction. it's possible variables: `basic`, `linear_regression`, `fft`, `ann`, `random_forest`, `svg`, `svm`, `ridge`, `bayesian`, `multi`
- `prediction_samples`  number of samples used by predictive model
- `prediction_number`  number of predictions made by predictive model
- `selector_model`: model used by selection. `rmse`, `bandit-greedy`, `bandit-ucb`, `bandit`.
- `planning_samples`: numbers of samples used by planning
- `limit_repicas`  limit of number of pool replicas

The variable `csv` is the folder where the system saves the statistics.

## Requisites
For compile this project you need `go` and `redis`, and of course, `storm`. Please refer to you platform's/OS' documentation for support.

The `go` last version used was 1.2.1  (see the <a href="https://go.dev/doc/install">go installation instructions</a> for details). For `redis`, the last version used was 6.x. And `storm`, the last version used was `2.6.2` version (see <a href="https://storm.apache.org/2024/04/05/storm262-released.html">storm release</a>).

## Deploy
Before starting the application, it is necessary to deploy `storm`, run `redis` and the REST app (Flask) from the `py` folder.
The main file is `initSps.sh` which is responsible for run the monitor. If the machine has no Golang installed, so you should comment line 4 `go build`, because this linea compile again the Go project. It's mandatory create the `\stats` folder in the project. And the `scripts` folder has Storm applications that the system can use. Each script is the commands for deploy Storm app, so you must change the Storm directory is necessary.

## References
<a id="1">[1]</a> Daniel Wladdimiro, Nicolás Hidalgo, Alessio Pagliari, Luciana Arantes, Pierre Sens, Erika Rosas, Victor Reyes. A Multi-Model Predictive Framework for Adaptive Resource Management in Stream Processing Systems. Future Generation Computer Systems, (In Review). 