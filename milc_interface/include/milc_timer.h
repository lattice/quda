#ifndef _TIMER_H_
#define _TIMER_H_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <time.h>


namespace milc_interface {

class Timer{

  public:
    Timer(const std::string& timer_tag);
    ~Timer();
     void check();
     void check(const std::string& statement);
     void mute();
     void stop();    

  private:
    std::string tag;
    double init_time;
    double last_time;
    double this_time;
    bool _mute;

};

} // namespace milc_interface


#endif  // _TIMER_H_
