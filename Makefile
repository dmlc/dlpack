.PHONY: clean all test doc

all: mock

LDFLAGS =
CFLAGS =  -std=c++11 -Wall -O3 -Iinclude

SRC = $(wildcard src/*.cc src/*/*.cc src/*/*/*.cc)
ALL_OBJ = $(patsubst src/%.cc, build/%.o, $(SRC))

doc:
	doxygen docs/Doxyfile

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

mock: $(ALL_OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

clean:
	$(RM) -rf build  */*/*/*~ */*.o */*/*.o */*/*/*.o */*.d */*/*.d */*/*/*.d
