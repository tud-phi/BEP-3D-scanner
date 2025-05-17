docker build -t bep25:1.0 .
docker run --mount type=bind,src="${PWD}\datasets\docker_test",dst="/app/datasets/docker_test/" --name "scanny_pear"  bep25:1.0 