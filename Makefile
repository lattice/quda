all: make.inc lib tests

make.inc:
	@echo 'Please run configure to create make.inc before building.'
	@exit 1

lib:
	$(MAKE) -C lib/

tests: lib
	$(MAKE) -C tests/

fortran: lib 
	$(MAKE) -C lib/ quda_fortran.o

tune:
	@echo "Manual tuning is no longer required.  Please see README file."

gen:
	$(MAKE) -C lib/ gen

clean:
	$(MAKE) -C lib/ clean
	$(MAKE) -C tests/ clean
	rm -rf ./config.log ./config.status ./autom4te.cache

.PHONY: all lib tests fortran tune gen clean
