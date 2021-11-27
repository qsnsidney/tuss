#include <thread>
#include <type_traits>

namespace CPUSIM
{
    // Function signature: void(int i)
    template <typename Function>
    void parallel_for(int n_thread, int begin, int end, Function &&f)
    {
        typename std::decay_t<Function> f_copy = std::forward<Function>(f);
        auto worker_thread = [&f_copy](int i_begin, int i_end)
        {
            for (int i = i_begin; i < i_end; i++)
            {
                f_copy(i);
            }
        };

        int count = end - begin;
        int count_per_thread = (count - 1) / n_thread + 1;

        std::vector<std::thread> threads;
        threads.reserve(n_thread);

        int count_remaining = count;
        for (int thread_idx = 0; thread_idx < n_thread; thread_idx++)
        {
            int i_begin = thread_idx * count_per_thread;
            int i_size = std::min(count_per_thread, count_remaining);
            int i_end = i_begin + i_size;
            count_remaining -= i_size;
            threads.emplace_back(worker_thread, i_begin, i_end);
        }

        for (auto &thread : threads)
        {
            thread.join();
        }
    }
}