all: lib tests

lib:
	$(MAKE) -C lib/

tests:
	$(MAKE) -C tests/

gen:
	$(MAKE) -C lib/ gen

clean:
	$(MAKE) -C lib/ clean
	$(MAKE) -C tests/ clean

.PHONY: all lib tests gen clean
