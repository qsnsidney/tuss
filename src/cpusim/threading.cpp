#include "threading.h"

namespace CPUSIM
{
    THREAD_POOL::THREAD_POOL(size_t n_thread)
    {
        resize(n_thread);
    }

    void THREAD_POOL::reset()
    {
        auto thread_quit_event = []()
        { return false; };

        const size_t n_thread = size();
        for (size_t thread_id = 0; thread_id < n_thread; thread_id++)
        {
            bool is_sent = threads_launch_channel_[thread_id]
                               .try_send(thread_quit_event);
            ASSERT(is_sent);
            threads_[thread_id].join();
        }
        threads_.clear();
        threads_launch_channel_.clear();
    }

    void THREAD_POOL::resize(size_t n_thread)
    {
        reset();

        if (n_thread == 0)
        {
            return;
        }

        threads_launch_channel_.resize(n_thread);
        threads_.reserve(n_thread);
        for (size_t thread_id = 0; thread_id < n_thread; thread_id++)
        {
            auto thread_worker = [&ch = threads_launch_channel_[thread_id]]()
            {
                while (true)
                {
                    thread_event_type event = ch.receive(); // Blocking wait
                    bool should_continue = event();
                    if (!should_continue)
                    {
                        break;
                    }
                }
            };
            threads_.emplace_back(std::move(thread_worker));
        }
        ASSERT(threads_launch_channel_.size() == threads_.size());
    }

}