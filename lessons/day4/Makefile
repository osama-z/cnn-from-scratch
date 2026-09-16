CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

all: conv2d cnn_forward

conv2d: conv2d.c
	$(CC) $(CFLAGS) -o conv2d conv2d.c

cnn_forward: cnn_forward.c
	$(CC) $(CFLAGS) -o cnn_forward cnn_forward.c $(LDFLAGS)

clean:
	rm -f conv2d cnn_forward
