#include "simple_engine.cuh"
#include "core/timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <iostream>

#include "core/physics.hpp"
#include "core/serde.h"
#include "helper.h"
#include "data_t.h"
#include "constant.h"
#include "basic_kernel.h"

#define SIM_TIME 10
#define STEP_SIZE 1
#define DEFAULT_BLOCK_SIZE 32
// namespace
// {
//     struct BUFFER
//     {
//         std::vector<CORE::POS> pos;
//         std::vector<CORE::VEL> vel;
//         std::vector<CORE::ACC> acc;

//         BUFFER(int n_body) : pos(n_body, {0, 0, 0}), vel(n_body, {0, 0, 0}), acc(n_body, {0, 0, 0}) {}
//     };

//     std::ostream &operator<<(std::ostream &os, const BUFFER &buf)
//     {
//         int counter = 0;
//         os << "POS: ";
//         for (auto p : buf.pos)
//         {
//             os << "[" << counter << "] " << p << ", ";
//             counter++;
//         }
//         os << std::endl;

//         counter = 0;
//         os << "VEL: ";
//         for (auto v : buf.vel)
//         {
//             os << "[" << counter << "] " << v << ", ";
//             counter++;
//         }
//         os << std::endl;

//         counter = 0;
//         os << "ACC: ";
//         for (auto a : buf.acc)
//         {
//             os << "[" << counter << "] " << a << ", ";
//             counter++;
//         }
//         os << std::endl;

//         return os;
//     }

//     CORE::BODY_STATE_VEC generate_body_state_vec(const BUFFER &buffer, const std::vector<CORE::MASS> &mass)
//     {
//         CORE::BODY_STATE_VEC body_states;
//         body_states.reserve(mass.size());
//         for (size_t i_body = 0; i_body < mass.size(); i_body++)
//         {
//             body_states.emplace_back(buffer.pos[i_body], buffer.vel[i_body], mass[i_body]);
//         }
//         return body_states;
//     }

//     void debug_workspace(const BUFFER &buffer, const std::vector<CORE::MASS> &mass)
//     {
//         int counter = 0;
//         std::cout << "MASS: ";
//         for (auto m : mass)
//         {
//             std::cout << "[" << counter << "] " << m << ", ";
//             counter++;
//         }
//         std::cout << std::endl;

//         std::cout << buffer;

//         std::cout << std::endl;
//     }
// }

namespace TUS
{
   SIMPLE_ENGINE::SIMPLE_ENGINE(CORE::BODY_STATE_VEC body_states_ic,
                                CORE::DT dt,
                                int n_threads,
                                std::optional<std::string> body_states_log_dir_opt) : ENGINE(std::move(body_states_ic), dt, std::move(body_states_log_dir_opt)),
                                                                                      n_threads_(n_threads)
   {
   }

   CORE::BODY_STATE_VEC SIMPLE_ENGINE::execute(int n_iter)
   {
      unsigned simulation_time = SIM_TIME;
      unsigned step_size = STEP_SIZE;

      CORE::TIMER timer("cuda program");

      /* BIN file of initial conditions */
      const auto &ic = body_states_ic();

      // TODO: get better debug message.
      size_t nBody = ic.size();

      // random initializer just for now
      srand(time(NULL));
      size_t vector_size = sizeof(data_t_3d) * nBody;
      size_t data_size = sizeof(data_t) * nBody;

      /*
     *   host side memory allocation
     */
      data_t_3d *h_X, *h_A, *h_V, *h_output_X;
      data_t *h_M;
      host_malloc_helper((void **)&h_X, vector_size);
      host_malloc_helper((void **)&h_A, vector_size);
      host_malloc_helper((void **)&h_V, vector_size);
      host_malloc_helper((void **)&h_output_X, vector_size);
      host_malloc_helper((void **)&h_M, data_size);
      timer.elapsed_previous("allocated host side memory");
      /*
     *   input randome initialize
     */

      parse_ic(h_X, h_V, h_M, ic);
      timer.elapsed_previous("deserialize_body_state_vec_from_csv");

      /*
     *  mass 
     */
      data_t *d_M;
      gpuErrchk(cudaMalloc((void **)&d_M, data_size));
      /*
     *   create double buffer on device side
     */
      data_t_3d **d_X, **d_A, **d_V;
      unsigned src_index = 0;
      unsigned dest_index = 1;
      d_X = (data_t_3d **)malloc(2 * sizeof(data_t_3d *));
      gpuErrchk(cudaMalloc((void **)&d_X[src_index], vector_size));
      gpuErrchk(cudaMalloc((void **)&d_X[dest_index], vector_size));

      d_A = (data_t_3d **)malloc(2 * sizeof(data_t_3d *));
      gpuErrchk(cudaMalloc((void **)&d_A[src_index], vector_size));
      gpuErrchk(cudaMalloc((void **)&d_A[dest_index], vector_size));

      d_V = (data_t_3d **)malloc(2 * sizeof(data_t_3d *));
      gpuErrchk(cudaMalloc((void **)&d_V[src_index], vector_size));
      gpuErrchk(cudaMalloc((void **)&d_V[dest_index], vector_size));

      timer.elapsed_previous("allocated device memory");
      /*
     *   create double buffer on device side
     */
      // cudaMemcpy(d_A[0], h_A, vector_size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_X[src_index], h_X, vector_size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_V[src_index], h_V, vector_size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_M, h_M, data_size, cudaMemcpyHostToDevice);
      timer.elapsed_previous("copied input data from host to device");

      // nthread is assigned to either 32 by default or set to a custom power of 2 by user
      std::cout << "Set thread_per_block to " << n_threads_ << std::endl;
      unsigned nblocks = (nBody + n_threads_ - 1) / n_threads_;

      // calculate the initialia acceleration
      calculate_acceleration<<<nblocks, n_threads_>>>(nBody, d_X[src_index], d_M, d_A[src_index]);
      timer.elapsed_previous("Calculated initial acceleration");

      std::cout << "Start Computation\n";

      for (unsigned step = 0; step < simulation_time; step += step_size)
      {

         // There should be more than one ways to do synchronization. I temporarily randomly choosed one
         calculate_acceleration<<<nblocks, n_threads_>>>(nBody, d_X[src_index], d_M,                                                          //input
                                                         d_A[dest_index]);                                                                    // output
         update_step<<<nblocks, n_threads_>>>(nBody, (data_t)step_size, d_X[src_index], d_V[src_index], d_A[src_index], d_M, d_A[dest_index], //input
                                              d_X[dest_index], d_V[dest_index]);                                                              // output

         // we don't have to synchronize here but this gices a better visualization on how fast / slow the program is
         std::cout << "epoch " << step << std::endl;
         cudaDeviceSynchronize();

         swap(src_index, dest_index);
      }
      cudaDeviceSynchronize();
      timer.elapsed_previous("Finished computation");
      // at end, the final data is actually at src_index because the last swap
      cudaMemcpy(h_output_X, d_X[src_index], vector_size, cudaMemcpyDeviceToHost);
      timer.elapsed_previous("copied output back to host");
      // Just for debug purpose on small inputs
      for (unsigned i = 0; i < nBody; i++)
      {
         //printf("object = %d, %f, %f, %f\n", i, h_output_X[i].x, h_output_X[i].y, h_output_X[i].z);
      }

      return body_states_ic();
   }
   // {
   // CORE::TIMER timer(std::string("execute(") + std::to_string(dt()) + "*" + std::to_string(n_iter) + ")");
   // const int n_body = body_states_ic().size();

   // std::vector<CORE::MASS> mass(n_body, 0);
   // BUFFER buf_in(n_body);
   // // Step 1: Prepare ic
   // for (int i_body = 0; i_body < n_body; i_body++)
   // {
   //     const auto &[body_pos, body_vel, body_mass] = body_states_ic()[i_body];
   //     buf_in.pos[i_body] = body_pos;
   //     buf_in.vel[i_body] = body_vel;
   //     mass[i_body] = body_mass;
   // }
   // timer.elapsed_previous("step1");

   // // Step 2: Prepare acceleration for ic
   // for (int i_target_body = 0; i_target_body < n_body; i_target_body++)
   // {
   //     buf_in.acc[i_target_body].reset();
   //     for (int j_source_body = 0; j_source_body < n_body; j_source_body++)
   //     {
   //         if (i_target_body != j_source_body)
   //         {
   //             buf_in.acc[i_target_body] += CORE::ACC::from_gravity(buf_in.pos[j_source_body], mass[j_source_body], buf_in.pos[i_target_body]);
   //         }
   //     }
   // }
   // timer.elapsed_previous("step2");

   // BUFFER buf_out(n_body);
   // std::vector<CORE::VEL> vel_tmp(n_body);
   // // Core iteration loop
   // for (int i_iter = 0; i_iter < n_iter; i_iter++)
   // {
   //     if (false)
   //     {
   //         debug_workspace(buf_in, mass);
   //     }

   //     for (int i_target_body = 0; i_target_body < n_body; i_target_body++)
   //     {
   //         // Step 3: Compute temp velocity
   //         vel_tmp[i_target_body] = CORE::VEL::updated(buf_in.vel[i_target_body], buf_in.acc[i_target_body], dt());

   //         // Step 4: Update position
   //         buf_out.pos[i_target_body] = CORE::POS::updated(buf_in.pos[i_target_body], buf_in.vel[i_target_body], buf_in.acc[i_target_body], dt());
   //     }

   //     for (int i_target_body = 0; i_target_body < n_body; i_target_body++)
   //     {
   //         buf_out.acc[i_target_body].reset();
   //         // Step 5: Compute acceleration
   //         for (int j_source_body = 0; j_source_body < n_body; j_source_body++)
   //         {
   //             if (i_target_body != j_source_body)
   //             {
   //                 buf_out.acc[i_target_body] += CORE::ACC::from_gravity(buf_out.pos[j_source_body], mass[j_source_body], buf_out.pos[i_target_body]);
   //             }
   //         }

   //         // Step 6: Update velocity
   //         buf_out.vel[i_target_body] = CORE::VEL::updated(vel_tmp[i_target_body], buf_out.acc[i_target_body], dt());
   //     }

   //     // Write BODY_STATE_VEC to log
   //     if (i_iter == 0)
   //     {
   //         push_body_states_to_log([&]()
   //                                 { return generate_body_state_vec(buf_in, mass); });
   //     }
   //     push_body_states_to_log([&]()
   //                             { return generate_body_state_vec(buf_out, mass); });
   //     if (i_iter % 10 == 0)
   //     {
   //         serialize_body_states_log();
   //     }

   //     // Prepare for next iteration
   //     std::swap(buf_in, buf_out);

   //     timer.elapsed_previous(std::string("iter") + std::to_string(i_iter));
   // }

   // return generate_body_state_vec(buf_in, mass);
   // }
}