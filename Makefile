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
numa:
	$(MAKE) -C tools/
	tools/gpu_affinity_test > tools/gpu_numa_config.txt
	echo The generated numa file is 
	cat tools/gpu_numa_config.txt
gen:
	$(MAKE) -C lib/ gen

clean:
	$(MAKE) -C lib/ clean
	$(MAKE) -C tests/ clean
	rm -rf ./config.log ./config.status ./autom4te.cache
.PHONY: all lib tests tune numa gen clean
