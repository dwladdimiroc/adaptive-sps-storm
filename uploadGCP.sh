echo 'upload branch adaptive storm dev'
go build
gcloud compute scp sps-storm sps-storm-nimbus:~/sps-storm
