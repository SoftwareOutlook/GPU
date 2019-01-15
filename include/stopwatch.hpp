#ifndef STOPWATCH_HPP
#define STOPWATCH_HPP
#include <boost/chrono/chrono.hpp>

class stopwatch{
private:    
  boost::chrono::system_clock::time_point t_beginning, t_end;
public:
  inline void start(){
    t_beginning=boost::chrono::system_clock::now();
  }
  inline void stop(){
    t_end=boost::chrono::system_clock::now();
  }
  inline double get_ns() const {
    return (boost::chrono::duration_cast<boost::chrono::nanoseconds>(t_end-t_beginning)).count();
  }
  inline double get() const {
    return get_ns()/1e9;   
  }
};
#endif
