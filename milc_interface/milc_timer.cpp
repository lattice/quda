#include <iostream>
#include <iomanip>
#include "include/milc_timer.h"
#include <util_quda.h>
#include <string.h>
#include <cstring>



namespace milc_interface {

Timer::Timer(const std::string& timer_tag) : tag(timer_tag), _mute(false){
  init_time = ((double)clock());
  this_time = last_time = init_time;
}


Timer::~Timer(){
  if(!_mute){
    this_time = ((double)clock());
	  printfQuda("%s : time = %e seconds\n", tag.c_str(), (this_time - init_time)/CLOCKS_PER_SEC);
  }
}





void Timer::check(){
  if(!_mute){
    last_time = this_time;
    this_time = ((double)clock());
  }
}


void Timer::check(const std::string& statement){
  if(!_mute){
    last_time = this_time;
	  this_time = ((double)clock());
	  printfQuda("%s::%s : time = %e seconds\n", tag.c_str(), statement.c_str(), (this_time - last_time)/CLOCKS_PER_SEC);
  }
}


void Timer::mute(){
  _mute = true;
}

void Timer::stop(){
	  if(!_mute){
	    this_time = ((double)clock());
	    printfQuda("%s : time = %e seconds\n", tag.c_str(), (this_time - init_time)/CLOCKS_PER_SEC);
	    _mute = true;
    }
}


}; // namespace milc_interface
