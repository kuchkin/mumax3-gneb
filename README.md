mumax3+GNEB and regularized LLG
======

GPU accelerated micromagnetic simulator.


Downloads, documentation, and instructions for building from source (for linux)
---------------------------

http://kuchkin.github.io/download.html

Prerequisites:

To run mumax3.10 you need
An NVIDIA GPU with at least a compute capability 3.0
An up to date NVIDIA driver (compatible versions given below)
Optional: gnuplot for plots in the web GUI

Setup Commands for mumax3-gneb:

sudo apt install golang-go
(Installs the Go programming language on your system.)

mkdir -p ~/go/src/github.com/kuchkin
(Creates the necessary directory structure for Go workspace following the standard Go project layout.)

cd ~/go/src/github.com/kuchkin/
(Changes the current directory to the newly created path to prepare for cloning the repository.)

git clone https://github.com/kuchkin/mumax3-gneb.git
(Clones the mumax3-gneb repository from GitHub into the current directory.)

cd mumax3-gneb/
(Changes the current directory to the newly created one.)

go mod init github.com/kuchkin/mumax3-gneb
(Initializes a new Go module for dependency management inside the project directory.)

make realclean && make
(Cleans up previous build files and compiles the project using the Makefile.)


Tools
-----

https://godoc.org/github.com/mumax/3/cmd


Contributing
------------

Contributions are gratefully accepted. To contribute code, fork our repo on github and send a pull request.
