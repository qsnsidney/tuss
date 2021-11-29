#include <thread>
#include <type_traits>
#include <optional>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <functional>
#include <memory>
#include "core/macros.hpp"

namespace CPUSIM
{
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
        CHANNEL_LITE() : state_ptr_(std::make_unique<STATE>()) {}

        // False if the channel is already full; True if successfully sent
        [[nodiscard]] bool try_send(T data);
        T receive();

    private:
        struct STATE
        {
            std::optional<T> parcel;
            std::mutex mutex;
            std::condition_variable cv_has_parcel;
        };
        std::unique_ptr<STATE> state_ptr_;
    };

    class THREAD_POOL
    {
    public:
        THREAD_POOL() = default;
        explicit THREAD_POOL(size_t n_thread);

        // Function signature: void(size_t thread_id)
        template <typename Function>
        void run(Function &&f);

        size_t size() const { return threads_.size(); }

        // Remove all worker threads
        void reset();
        // Does a reset and then reset to n_thread
        void resize(size_t n_thread);

    private:
        using thread_event_type = std::function<bool()>; // true to continue thread event loop; false to terminate
        using thread_event_launch_channel = CHANNEL_LITE<thread_event_type>;
        std::vector<std::thread> threads_;
        std::vector<thread_event_launch_channel> threads_launch_channel_;
    };

    /// Function signature: void(int i)
    template <typename Function>
    void parallel_for(size_t n_thread, int begin, int end, Function &&f);

    /// Implementation

    template <typename T>
    bool CHANNEL_LITE<T>::try_send(T data)
    {
        {
            std::lock_guard<std::mutex> lock_guard(state_ptr_->mutex);
            if (state_ptr_->parcel.has_value())
            {
                return false;
            }
            else
            {
                state_ptr_->parcel.emplace(std::move(data));
            }
        }
        state_ptr_->cv_has_parcel.notify_one();
        return true;
    }

    template <typename T>
    T CHANNEL_LITE<T>::receive()
    {
        std::unique_lock<std::mutex> unique_lock(state_ptr_->mutex);
        state_ptr_->cv_has_parcel
            .wait(unique_lock, [this]()
                  { return state_ptr_->parcel.has_value(); });
        T data = std::move(state_ptr_->parcel.value());
        state_ptr_->parcel.reset();
        return data;
    }

    template <typename Function>
    void THREAD_POOL::run(Function &&f)
    {
        typename std::decay_t<Function> f_copy = std::forward<Function>(f);
        const size_t n_thread = size();
        std::vector<CHANNEL_LITE<int> > finish_synchs(n_thread); // The actual data does not matter

        // Launch
        for (size_t thread_id = 0; thread_id < n_thread; thread_id++)
        {
            bool is_sent =
                threads_launch_channel_[thread_id]
                    .try_send([f_copy, thread_id, &finish_synch = finish_synchs[thread_id]]
                              {
                                  f_copy(thread_id);
                                  bool is_sent = finish_synch.try_send(1);
                                  ASSERT(is_sent);
                                  return true;
                              });
            ASSERT(is_sent);
        }

        // Synchronize
        for (auto &finish_synch : finish_synchs)
        {
            finish_synch.receive(); // Blocking wait
        }
    }

    /// Prepare the Function f to be ready to run on multiple threads.
    /// Function signature: void(int i)
    ///     The main body of the task to be run in a for loop.
    /// Executor signature: void Executor::operator()(typename ThreadTask)
    ///     The actual execution of the threadTask.
    /// ThreadTask signature: void(size_t thread_id)
    ///     A task that can be run on a thread directly.
    template <typename Executor, typename Function>
    void parallel_for_impl(Executor &executor, size_t n_thread, int begin, int end, Function &&f)
    {
        int count = end - begin;
        int count_per_thread = (count - 1) / n_thread + 1;

        // Launch and synchronize
        executor([f = std::forward<Function>(f), begin, end, n_thread, count_per_thread](size_t thread_id)
                 {
                     int i_begin = begin + thread_id * count_per_thread;
                     int i_end = (thread_id == n_thread - 1) ? end : (i_begin + count_per_thread);
                     for (int i = i_begin; i < i_end; i++)
                     {
                         f(i);
                     }
                 });
    }

    template <typename Function>
    void parallel_for(size_t n_thread, int begin, int end, Function &&f)
    {
        // Define an executor
        auto executor = [n_thread](auto &&thread_task)
        {
            using ThreadTask = decltype(thread_task);
            typename std::decay_t<ThreadTask> thread_task_copy = std::forward<ThreadTask>(thread_task);

            // Launch
            std::vector<std::thread> threads;
            threads.reserve(n_thread);
            for (int thread_id = 0; thread_id < n_thread; thread_id++)
            {
                threads.emplace_back(thread_task_copy, thread_id);
            }

            // Synchronize
            for (auto &thread : threads)
            {
                thread.join();
            }
        };

        parallel_for_impl(executor, n_thread, begin, end, std::forward<Function>(f));
    }
}