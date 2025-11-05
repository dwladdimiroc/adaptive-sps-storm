echo 'upload branch adaptive storm dev'
go build
gcloud compute scp sps-storm storm-node-central:~/sps-storm
