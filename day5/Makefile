CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

all: fixed_point

fixed_point: fixed_point.c
	$(CC) $(CFLAGS) -o fixed_point fixed_point.c $(LDFLAGS)

clean:
	rm -f fixed_point
