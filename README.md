mumax3+GNEB 
======

GPU accelerated micromagnetic simulator.


Downloads and documentation
---------------------------

http://kuchkin.github.io


Paper
-----

Coming soon!


Tools
-----

https://godoc.org/github.com/mumax/3/cmd


Building from source (for linux)
--------------------

Consider downloading a pre-compiled binary. If you want to compile nevertheless:

  * install the nvidia proprietary driver, if not yet present.
   - if unsure, it's probably already there
   - version 440.44 recommended
  * install Go 
    - https://golang.org/dl/
    - set $GOPATH
  * install CUDA 
    - https://developer.nvidia.com/cuda-downloads (pick default installation path)
    - or `sudo apt-get install nvidia-cuda-toolkit`
  * install a C compiler
    - on Ubuntu: `sudo apt-get install gcc`
  * if you have git installed: 
    - `go get github.com/kuchkin/mumax3-gneb/cmd/mumax3`
  * optional: install gnuplot if you want pretty graphs
    - on ubuntu: `sudo apt-get install gnuplot`
  * use the Makefile if there is a need to recompile the cuda kernels
    - `make realclean && make`

Your binary is now at `$GOPATH/bin/mumax3`

Contributing
------------

Contributions are gratefully accepted. To contribute code, fork our repo on github and send a pull request.
