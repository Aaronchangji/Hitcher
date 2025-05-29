#include <faiss/hitcher/client.h>

namespace faiss{

namespace hitcher {

Client::Client(StoragePtr storage_ptr, int offset, int num_query, double qps) {
  std::mt19937 rng(0);
  auto arrival_distribution = ScheduleDistribution(qps);
  PerfClock::time_point schedule_time = PerfClock::now();
  for (int i=offset; i<offset + num_query; i++) {
    storage_ptr->addQuery(i);
    schedule_time += arrival_distribution(rng);
    if (PerfClock::now() < schedule_time) {
      std::this_thread::sleep_until(schedule_time);
    }
  }
}


}

}