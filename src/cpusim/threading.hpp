#include <thread>
#include <type_traits>
#include <optional>
#include <mutex>
#include <condition_variable>

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

    /// A channel with capacity 1 that supports backpressure.
    /// 1. The channel has a capacity of 1, i.e., no buffering.
    /// 2. A send call fulfills the channel, and holds the content until
    ///    a receive call happens.
    ///    The try_send call is non-blocking:
    ///    If the channel is empty, it populates the channel right away;
    ///    If the channel is full, it immediately fails.
    /// 3. A receive call clears out the channel.
    ///    The receive call is blocking:
    ///    If the channel is full, it clears the channel right away;
    ///    If the channel is empty, it blocks and waits until the channel is full.
    /// 4. A single send call pairs with a single receive call.
    /// std::condition_variable: https://en.cppreference.com/w/cpp/thread/condition_variable
    template <typename T>
    class CHANNEL_LITE
    {
    public:
        // False if the channel is already full; True if successfully sent
        bool try_send(T data)
        {
            {
                std::lock_guard<std::mutex> lock_guard(mutex_);
                if (parcel_.has_value())
                {
                    return false;
                }
                else
                {
                    parcel_.emplace(std::move(data));
                }
            }
            cv_has_parcel_.notify_one();
            return true;
        }

        T receive()
        {
            std::unique_lock<std::mutex> unique_lock(mutex_);
            cv_has_parcel_.wait(unique_lock, [this]()
                                { return parcel_.has_value(); });
            T data = std::move(parcel_.value());
            parcel_.reset();
            return data;
        }

    private:
        std::optional<T> parcel_;
        std::mutex mutex_;
        std::condition_variable cv_has_parcel_;
    };
}