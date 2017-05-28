# dev workflow with aws

1. get an instance up and running
1. allow ssh on port 22
1. disable ssh password auth and install key pair
1. use `Makefile` commands to start/stop/ssh into the machine
  * this will forward the remote `8888` port to the local machine
1. when on the machine start the notebook
  * access the notebook locally at [http://localhost:8888](http://localhost:8888)
1. after everything is done `make syncdown` to sync the remote contents to local file
