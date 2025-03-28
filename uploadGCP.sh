echo 'upload branch adaptive storm serverless'
go build
gcloud compute scp sps-storm sps-adaptation:~/sps-storm
