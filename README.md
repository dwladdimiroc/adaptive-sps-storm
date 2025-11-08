# Self-adaptive in Apache Storm
This work presents a new version of the MAPE model implementation for the Storm (version 2.8.0) extension, updating and improving upon the version described in [1]. In this project, we introduce a self-adaptive system designed to dynamically adjust the number of active/inactive replicas for each operator pool in the SPS application, now leveraging the latest version of Apache Storm. The system analyzes key metrics (e.g., input rate, execution time, queue length) and plans the necessary adjustments to ensure that all incoming events in the SPS are efficiently processed.

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
- `time_window_size` size of the time period (seconds) where a sample is obtained. Equivalent to monitor module time window.
- `benchmark_samples` numbers of samples used by the benchmark.
- `analyze_samples` analyze module time window.
- `preditive_model` model used by input prediction. it's possible variables: `basic`, `linear_regression`, `fft`, `ann`, `random_forest`, `svg`, `svm`, `ridge`, `bayesian`.
- `prediction_samples`  number of samples used by predictive model.
- `prediction_number`  number of predictions made by predictive model.
- `planning_samples` plan module time window.
- `limit_repicas`  limit of number of pool replicas.

The `rest_metric` is the REST app parameters to obtain the stats in the topology. The variable `port` is the REST App port.

The variable `csv` is the folder where the system saves the statistics.

## Requisites
For compile this project you need `go` and `redis`, and of course, `storm`. Please refer to you platform's/OS' documentation for support.

The `go` last version used was 1.23.0  (see the <a href="https://go.dev/doc/install">go installation instructions</a> for details). For `redis`, the last version used was 6.x. And `storm`, the last version used was `2.8.0` version (see <a href="https://storm.apache.org/2024/04/05/storm262-released.html">storm release</a>).

## Deploy
Before starting the application, it is necessary to deploy `storm`, run `redis` and the REST app (Flask) from the `py` folder.
The main file is `initSps.sh` which is responsible for run the monitor. If the machine has no Golang installed, so you should comment line 4 `go build`, because this linea compile again the Go project. It's mandatory create the `\stats` folder in the project. And the `scripts` folder has Storm applications that the system can use. Each script is the commands for deploy Storm app, so you must change the Storm directory is necessary.