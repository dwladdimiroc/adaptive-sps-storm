# Self-adaptive in Apache Storm
Implementation of MAPE Model for Storm extension in the article [[1]](#1). This project presents the self-adaptive system for to modify the number of active/inactive replicas for each pool of operators in the SPS application. The goal is to analyse differences metrics (i.e. input, executed time, queue) and to plan the changes necessary for to process all input events in the SPS.

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
- `preditive_model` model used by input prediction. it's possible variables: `basic`, `linear_regression`, `fft`, `ann`, `random_forest`
- `prediction_samples`  number of samples used by predictive model
- `prediction_number`  number of predictions made by predictive model
- `limit_repicas`  limit of number of pool replicas
- `throughput_weight` multi-metric throughput weight
- `latency_weight` multi-metric latency weight
- `latency_limit` execution time degradation threshold
- `queue_weight` multi-metric queue weight

The variable `csv` is the folder where the system saves the statistics.

## Requisites
For compile this project you need `go` and `redis`, and of course, `storm`. Please refer to you platform's/OS' documentation for support.

The `go` last version used was 1.2.2  (see the <a href="https://go.dev/doc/install">go installation instructions</a> for details). For `redis`, the last version used was 6.x. And `storm`, you must use the `2.x` version (see <a href="https://github.com/apache/storm/tree/2.x-branch">storm branch</a>).

## Deploy

The main file is `initSps.sh` which is responsible for run the monitor. If the machine has no Golang installed, so you should comment line 4 `go build`, because this linea compile again the Go project. It's mandatory create the `\stats` folder in the project. And the `scripts` folder has Storm applications that the system can use. Each script is the commands for deploy Storm app, so you must change the Storm directory is necessary.

## References
<a id="1" href="https://www.sciencedirect.com/science/article/pii/S0743731524001047">[1]</a>
Daniel Wladdimiro, Luciana Arantes, Nicolas Hidalgo, Pierre Sens. PA-SPS: A predictive adaptive approach for an elastic stream processing system.
Journal of Parallel and Distributed Computing, Volume 192, 2024, 104940, ISSN 0743-7315. 