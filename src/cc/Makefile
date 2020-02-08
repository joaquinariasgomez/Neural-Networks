CC=g++
CFLAGS= -std=c++14 -c -Wall

main: main.o Perceptron.o
	$(CC) -o main main.o Perceptron.o
main.o: main.cc
	$(CC) $(CFLAGS) main.cc
Perceptron.o: Perceptron.cc
	$(CC) $(CFLAGS) Perceptron.cc
clean :
	rm -rf *.o
realclean: clean
	rm -rf main
