To compile "code.cpp" use:
g++ -pthread -o <output-name> code.cpp
It is important to include "-pthread" because the program uses threads and it won't compile without that argument.
The program is meant to be run with 12 threads.