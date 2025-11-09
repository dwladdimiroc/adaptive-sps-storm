echo 'upload branch adaptive storm v3.1'
go build
gcloud compute scp sps-storm storm-node-central:~/sps-storm
