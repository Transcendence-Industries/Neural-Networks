CC = gcc
CPPFLAGS = -Iinclude
WARN = -Wall -Wextra -Werror -pedantic
STD = -std=c11
OPT = -O2
DBG = -O0 -g
COMMON_SRC = src/config.c src/tensor.c src/accuracy_funcs.c src/activation_funcs.c src/loss_funcs.c src/layers.c src/calc.c src/network.c src/io.c src/dataset.c
MAIN_SRC = src/main.c
TEST_SRC = tests/test_main.c
TARGET = neural-networks
TEST_TARGET = neural-networks-tests
ARTIFACTS_DIR = artifacts

all: release

release: CFLAGS = $(STD) $(WARN) $(OPT)
release: $(TARGET)

debug: CFLAGS = $(STD) $(WARN) $(DBG)
debug: clean $(TARGET)

san: CFLAGS = $(STD) $(WARN) $(DBG) -fsanitize=address,undefined
san: LDFLAGS = -fsanitize=address,undefined
san: clean $(TARGET) $(TEST_TARGET)
	./$(TEST_TARGET)

$(TARGET): $(COMMON_SRC:.c=.o) $(MAIN_SRC:.c=.o)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lm

$(TEST_TARGET): $(COMMON_SRC:.c=.o) $(TEST_SRC:.c=.o)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lm

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

test: CFLAGS = $(STD) $(WARN) $(DBG)
test: clean $(TEST_TARGET)
	./$(TEST_TARGET)

run-demo: release
	mkdir -p $(ARTIFACTS_DIR)
	./$(TARGET)

clean:
	rm -f *.o src/*.o tests/*.o $(TARGET) $(TEST_TARGET) neural-networks neural-networks-tests $(ARTIFACTS_DIR)/*.nn data/model.nn data/test_model.nn

.PHONY: all release debug san test run-demo clean
