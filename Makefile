all: make.inc lib tests

make.inc:
	@echo 'Before building, please create make.inc from make.inc.example'
	@exit 1

lib:
	$(MAKE) -C lib/

tests:
	$(MAKE) -C tests/

tune:
	$(MAKE) -C tests/ tune
	@echo "Autotuning completed successfully. " \
	      "Please type 'make' to rebuild library."

gen:
	$(MAKE) -C lib/ gen

clean:
	$(MAKE) -C lib/ clean
	$(MAKE) -C tests/ clean

.PHONY: all lib tests tune gen clean
